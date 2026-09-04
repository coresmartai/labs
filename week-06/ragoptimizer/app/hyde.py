"""HyDE - Hypothetical Document Embeddings.

The technique: ask the model to write the answer, then embed THAT answer
and use its vector as the retrieval probe. Throw the content away; only
the embedding shape matters. Wins on vocabulary-gap queries.

The prompt is tunable per corpus via Settings.hyde_prompt_voice. Re-run
the golden eval set after each prompt change.

Failure mode this module guards against: NEIGHBOURHOOD DRIFT - the model
writes a plausible hypothetical answer about the wrong subsystem. The
fix is to also retrieve against the original query in parallel (see
app/retriever.py) and union the candidate sets.
"""
from __future__ import annotations

from app.config import get_settings
from app.llm import generate_text


_HYDE_PROMPT_TEMPLATE = (
    "Write a concise, technical four-sentence answer to the following engineering "
    "question, in the voice of a {voice}. Be specific. Use the vocabulary the "
    "actual documentation would use. Do not hedge or refuse - even if you don't "
    "know the real answer, write a plausible one.\n\nQuestion: {query}\n\nAnswer:"
)


def hyde_probe(query: str) -> str:
    """Return a hypothetical answer string to use as the retrieval probe.

    Uses the small/fast hyde_model (gpt-5.4-nano) - content is discarded
    after embedding, so generation quality matters less than speed.
    """
    settings = get_settings()
    prompt = _HYDE_PROMPT_TEMPLATE.format(
        voice=settings.hyde_prompt_voice,
        query=query.strip(),
    )
    answer = generate_text(prompt, max_tokens=200, model=settings.hyde_model)
    return answer.strip()
