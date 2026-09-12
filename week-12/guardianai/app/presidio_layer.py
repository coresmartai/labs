"""Presidio-backed PII scrubbing.

Three call sites, one library, one audit format:
  - `scrub_chunk(text)`  : INGESTION. Runs before the chunk is embedded and
                           written to Qdrant. Unredacted PII never lands in
                           the store.
  - `scrub_input(text)`  : on the way in, before the model or any log sees it.
  - `scrub_output(text)` : on buffered output, before the SSE emit.

Two custom recognisers (TICKET_ID, CUSTOMER_ID) are registered at analyzer
construction - about fifteen lines per entity. Two thresholds: above the
redact threshold the operator fires; between log-only and redact the detection
is recorded for human review but nothing is transformed.

The NLP engine is built EXPLICITLY, via NlpEngineProvider, for two reasons.
First, it is the only way `SPACY_MODEL` actually reaches Presidio - the bare
`AnalyzerEngine()` constructor ignores it and loads Presidio's own default.
Second, the bare constructor will try to pip-download a missing spaCy model,
which throws from deep inside Presidio long before our friendly error message
gets a chance to run - and needs the network. We check `spacy.util.is_package`
first, so a missing model is a clean, offline, explainable condition.
"""
from __future__ import annotations

import logging

import spacy
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.predefined_recognizers import PhoneRecognizer
from presidio_analyzer.nlp_engine import NlpEngine, NlpEngineProvider, SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.config import get_settings
from app.schemas import ScrubDetection, ScrubResult

logger = logging.getLogger(__name__)

REDACTION_TOKEN = "<REDACTED>"

# The entities we actually scan for. Presidio ships dozens of recognisers, and
# left unbounded they produce noise that discredits the layer - a US phone
# number matches the UK NHS checksum, "Escalation" matches an Indian PAN. An
# explicit allowlist is Presidio's built-in coverage we actually want (person,
# email, phone, credit card, IBAN, IP) plus our two domain entities.
ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "TICKET_ID",
    "CUSTOMER_ID",
]

# Presidio's built-in phone recogniser scores EVERY match at 0.40 - below our
# 0.65 redact threshold. Out of the box, therefore, phone numbers are logged
# and never redacted. Presidio's context enhancer can lift a match to 0.75 when
# a word like "phone" happens to sit next to the number, but that depends both
# on the spaCy lemmatiser and on the document phrasing. A privacy control does
# not get to depend on phrasing. We re-register the built-in recogniser with a
# score that clears the threshold on its own, and we write the number down here
# so the choice is reviewable.
PHONE_SCORE = 0.85


# ---- NLP engine ----

def spacy_model_installed(model_name: str) -> bool:
    """True iff the spaCy model is importable as an installed package."""
    return bool(spacy.util.is_package(model_name))


class _LoadedSpacyNlpEngine(SpacyNlpEngine):
    """Presidio's spaCy engine, handed a pipeline we loaded ourselves.

    Presidio's own loader will pip-download a missing model. We never want
    that: it needs the network, and it buries the real problem in a stack
    trace. So we always load (or blank-out) the pipeline first and inject it.
    """

    def __init__(self, loaded_pipeline) -> None:
        super().__init__()
        self.nlp = {"en": loaded_pipeline}


def _build_nlp_engine() -> NlpEngine:
    """Build Presidio's NLP engine from OUR configured spaCy model.

    Two bugs die here. (1) `AnalyzerEngine()` with no arguments ignores
    SPACY_MODEL entirely and loads Presidio's own default - so the setting
    was decorative. (2) If the model is missing, Presidio tries to download
    it. We check `spacy.util.is_package` first, and on a miss we log a loud
    warning and fall back to a BLANK English pipeline: no download, offline
    safe - and NLP recognisers (PERSON, LOCATION, ORG) silently stop firing.

    That silence is the danger: a scrubber that quietly stops catching names
    still looks like it works. `startup_check()` turns it into a boot crash
    instead of a false sense of safety.
    """
    settings = get_settings()
    model_name = settings.spacy_model

    if not spacy_model_installed(model_name):
        logger.warning(
            "no spaCy model found (%s); NLP recognisers disabled. "
            "Run: python -m spacy download %s",
            model_name,
            model_name,
        )
        return _LoadedSpacyNlpEngine(spacy.blank("en"))

    # The model IS installed, so the provider can safely build the engine -
    # and this is the only construction path that actually hands SPACY_MODEL
    # to Presidio. Announce it: this call is where the ~560MB load happens, so
    # if a boot (or a first request, when startup_check is disabled) pauses,
    # this line names the reason.
    logger.info("loading spaCy model %r into Presidio (this is the slow part)...", model_name)
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
        }
    )
    return provider.create_engine()


# ---- analyzer construction ----

class _ScoredPhoneRecognizer(PhoneRecognizer):
    """Presidio's phone recogniser, with a score that clears the threshold."""

    SCORE = PHONE_SCORE


def _build_analyzer() -> AnalyzerEngine:
    """Default analyzer plus the two domain custom recognisers."""
    analyzer = AnalyzerEngine(nlp_engine=_build_nlp_engine(), supported_languages=["en"])

    # Swap the built-in phone recogniser for the re-scored one. See PHONE_SCORE.
    analyzer.registry.remove_recognizer("PhoneRecognizer")
    analyzer.registry.add_recognizer(_ScoredPhoneRecognizer(supported_language="en"))

    # TICKET_ID - about fifteen lines per entity.
    # Word boundaries, NOT anchors: this is a scanner, not a validator. An
    # anchored pattern (^...$) catches the first ID in a string and misses
    # every later one.
    ticket_id = PatternRecognizer(
        supported_entity="TICKET_ID",
        name="ticket_id_recognizer",
        patterns=[
            Pattern(
                name="tck_id",
                regex=r"\bTCK-\d{4}-\d{6}\b",
                score=0.9,
            )
        ],
        # Context words near a match bump its confidence - the word "ticket"
        # beside a TCK identifier is what separates a real ID from a random
        # string of the same shape in a log line.
        context=["ticket", "case", "issue"],
    )
    analyzer.registry.add_recognizer(ticket_id)

    # CUSTOMER_ID - same fifteen-line shape, different entity.
    customer_id = PatternRecognizer(
        supported_entity="CUSTOMER_ID",
        name="customer_id_recognizer",
        patterns=[
            Pattern(
                name="cus_id",
                regex=r"\bCUS-[A-Z0-9]{12}\b",
                score=0.9,
            )
        ],
        context=["account", "customer"],
    )
    analyzer.registry.add_recognizer(customer_id)
    return analyzer


_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None


def _get_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    global _analyzer, _anonymizer
    if _analyzer is None:
        _analyzer = _build_analyzer()
        _anonymizer = AnonymizerEngine()
    assert _anonymizer is not None
    return _analyzer, _anonymizer


def reset_engines() -> None:
    """Drop the cached engines. Used by tests and after a SPACY_MODEL change."""
    global _analyzer, _anonymizer
    _analyzer = None
    _anonymizer = None


# ---- startup check ----

def startup_check() -> None:
    """Crash the boot if Presidio is not actually operational.

    A guardrail that fails silently gives the team confidence it has not
    earned. The spaCy package check runs BEFORE the analyzer is constructed:
    Presidio would otherwise try to download the model and die with a stack
    trace nobody can read, offline.
    """
    settings = get_settings()

    if not spacy_model_installed(settings.spacy_model):
        raise RuntimeError(
            f"spaCy model {settings.spacy_model!r} not found. "
            f"Run: python -m spacy download {settings.spacy_model}"
        )

    analyzer, _ = _get_engines()

    # Verify both default and custom recognisers fire on a known input.
    probe = "Contact John Doe at john@example.com about ticket TCK-2026-001234."
    results = analyzer.analyze(text=probe, language="en", entities=ENTITIES)
    entities_found = {r.entity_type for r in results}
    required = {"PERSON", "EMAIL_ADDRESS", "TICKET_ID"}
    missing = required - entities_found
    if missing:
        raise RuntimeError(
            f"Presidio startup check failed. Missing recognisers for: {missing}. "
            f"Found: {entities_found}"
        )


# ---- core scrub function ----

def _operators() -> dict[str, OperatorConfig]:
    """`replace` writes <REDACTED> in place of the span; `redact` deletes it.

    Presidio's own default is `replace` with <ENTITY_TYPE>. We replace too, but
    with one fixed token, so every redaction reads the same. `redact` would
    leave a gap the reader cannot tell from a typo. We record which operator
    fired on every audit event so the choice is never invisible.
    """
    settings = get_settings()
    op = settings.presidio_operator
    if op == "replace":
        return {"DEFAULT": OperatorConfig("replace", {"new_value": REDACTION_TOKEN})}
    return {"DEFAULT": OperatorConfig(op)}


def _dedupe(results: list) -> list:
    """Drop any detection fully contained inside a higher-scoring one.

    Presidio happily reports PHONE_NUMBER for the "2026-004417" inside
    "TCK-2026-004417". The anonymised text is unaffected (the wider, stronger
    span wins), but the AUDIT LOG would carry a phantom phone number that never
    existed. The log is the artefact an auditor reads. Keep it clean.
    """
    kept = []
    for r in sorted(results, key=lambda x: (-x.score, x.start - x.end)):
        if any(k.start <= r.start and r.end <= k.end for k in kept):
            continue
        kept.append(r)
    return kept


def _scrub(text: str, surface: str) -> ScrubResult:
    settings = get_settings()
    analyzer, anonymizer = _get_engines()

    results = _dedupe(analyzer.analyze(text=text, language="en", entities=ENTITIES))
    # Split detections into redact-threshold and log-only.
    to_redact = [r for r in results if r.score >= settings.presidio_redact_threshold]
    log_only = [
        r for r in results
        if settings.presidio_log_only_threshold <= r.score < settings.presidio_redact_threshold
    ]

    if to_redact:
        anonymized = anonymizer.anonymize(
            text=text, analyzer_results=to_redact, operators=_operators()
        )
        cleaned = anonymized.text
    else:
        cleaned = text

    detections = [
        ScrubDetection(
            recognizer=r.entity_type,
            confidence=float(r.score),
            span_start=r.start,
            span_end=r.end,
        )
        for r in (to_redact + log_only)
    ]
    return ScrubResult(cleaned_text=cleaned, detections=detections, surface=surface)


def is_redacted(detection: ScrubDetection) -> bool:
    """True iff this detection cleared the redact threshold and was actually
    transformed. Below it (but above log-only) the detection is recorded for
    human review and the text is left alone - that is the whole point of the
    two-threshold design."""
    return detection.confidence >= get_settings().presidio_redact_threshold


def scrub_input(text: str) -> ScrubResult:
    """Call site 1 of 3: the user's query, before anything else sees it."""
    return _scrub(text, surface="input")


def scrub_output(text: str) -> ScrubResult:
    """Call site 2 of 3: one buffered sentence, before it is streamed out."""
    return _scrub(text, surface="output")


def scrub_chunk(text: str) -> ScrubResult:
    """Call site 3 of 3: INGESTION. Runs in `app/ingest.py`, before the chunk
    is embedded and upserted. Unredacted PII never lands in the store."""
    return _scrub(text, surface="retrieval")


# ---- streaming output buffer ----

class SentenceBufferScrubber:
    """Sentence-boundary buffered output scrubber.

    Accumulates tokens until a sentence boundary (`.`, `!`, `?` followed by
    whitespace) OR until the buffer reaches `max_buffer` characters. Then
    `scrub_output` runs over the buffered text and the scrubbed text is
    yielded. PII routinely spans several tokens, so a per-token filter has
    already shipped the first half of a phone number by the time the pattern
    is recognisable - and SSE cannot retract a token.

    280 is calibrated against the longest span we expect: an international
    phone with an extension is ~50 chars, an email with a display name ~80.
    """

    _BOUNDARIES = {".", "!", "?"}

    def __init__(self, max_buffer: int | None = None) -> None:
        settings = get_settings()
        self.max_buffer = max_buffer or settings.output_scrubber_max_buffer
        self._buf: list[str] = []

    def push(self, token: str) -> str | None:
        """Append one token. Return scrubbed text iff a boundary fired."""
        self._buf.append(token)
        joined = "".join(self._buf)
        if self._should_flush(joined):
            self._buf.clear()
            return scrub_output(joined).cleaned_text
        return None

    def flush(self) -> str | None:
        """Drain any remaining buffered text at end-of-stream."""
        if not self._buf:
            return None
        joined = "".join(self._buf)
        self._buf.clear()
        return scrub_output(joined).cleaned_text

    def _should_flush(self, joined: str) -> bool:
        if len(joined) >= self.max_buffer:
            return True
        # Boundary char followed by whitespace - flushing on the bare boundary
        # would split decimals like "3.14" mid-number.
        if len(joined) >= 2 and joined[-1].isspace() and joined[-2] in self._BOUNDARIES:
            return True
        # A newline is a hard break regardless of punctuation.
        if joined.endswith("\n"):
            return True
        return False
