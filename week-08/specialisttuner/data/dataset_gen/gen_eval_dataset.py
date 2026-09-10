"""
gen_eval_dataset.py
Generate eval_aiayn.jsonl for V2 - 42 short-answer questions from
"Attention Is All You Need" (Vaswani et al., 2017).
Each answer is a number, single word, or short phrase verifiable by substring match.
Run: python gen_eval_dataset.py
"""
import json
import pathlib

EVAL = [
    # ── architecture ──────────────────────────────────────────────────────────
    {"id":  1, "q": "How many encoder layers N does the base Transformer have?",
     "e": "6", "a": ["6", "six", "n=6", "n = 6"], "c": "architecture"},
    {"id":  2, "q": "How many decoder layers N does the base Transformer have?",
     "e": "6", "a": ["6", "six", "n=6"], "c": "architecture"},
    {"id":  3, "q": "What is d_model in the base Transformer?",
     "e": "512", "a": ["512"], "c": "architecture"},
    {"id":  4, "q": "What is d_ff (inner feed-forward dimension) in the base Transformer?",
     "e": "2048", "a": ["2048"], "c": "architecture"},
    {"id":  5, "q": "How many attention heads h does the base Transformer have?",
     "e": "8", "a": ["8", "eight", "h=8"], "c": "architecture"},
    {"id":  6, "q": "What is d_k per head in the base Transformer?",
     "e": "64", "a": ["64", "dk=64"], "c": "architecture"},
    {"id":  7, "q": "How many attention heads h does the big Transformer have?",
     "e": "16", "a": ["16", "sixteen", "h=16"], "c": "architecture"},
    {"id":  8, "q": "What is d_model in the big Transformer?",
     "e": "1024", "a": ["1024"], "c": "architecture"},
    {"id":  9, "q": "What is d_ff in the big Transformer?",
     "e": "4096", "a": ["4096"], "c": "architecture"},
    {"id": 10, "q": "How many parameters does the base Transformer have (in millions)?",
     "e": "65", "a": ["65", "65m", "65 m", "65 million"], "c": "architecture"},
    {"id": 11, "q": "How many parameters does the big Transformer have (in millions)?",
     "e": "213", "a": ["213", "213m", "213 million"], "c": "architecture"},
    {"id": 12, "q": "How many sub-layers does each encoder layer contain?",
     "e": "2", "a": ["2", "two"], "c": "architecture"},
    {"id": 13, "q": "How many sub-layers does each decoder layer contain?",
     "e": "3", "a": ["3", "three"], "c": "architecture"},

    # ── attention ─────────────────────────────────────────────────────────────
    {"id": 14, "q": "What is the scaling factor in Scaled Dot-Product Attention?",
     "e": "1/sqrt(dk)", "a": ["sqrt", "1/sqrt", "sqrt(dk)", "dk"], "c": "attention"},
    {"id": 15, "q": "What is the maximum path length for self-attention layers?",
     "e": "O(1)", "a": ["o(1)", "1", "constant"], "c": "attention"},
    {"id": 16, "q": "What is the maximum path length for recurrent layers?",
     "e": "O(n)", "a": ["o(n)", "n"], "c": "attention"},
    {"id": 17, "q": "What activation function does the Transformer position-wise FFN use?",
     "e": "ReLU", "a": ["relu"], "c": "attention"},
    {"id": 18, "q": "What is d_v per head in the base Transformer?",
     "e": "64", "a": ["64"], "c": "attention"},
    {"id": 19, "q": "What normalisation is applied to each sub-layer output in the Transformer?",
     "e": "LayerNorm", "a": ["layernorm", "layer norm", "layer normalization"], "c": "attention"},

    # ── training ──────────────────────────────────────────────────────────────
    {"id": 20, "q": "What optimizer was used to train the Transformer?",
     "e": "Adam", "a": ["adam"], "c": "training"},
    {"id": 21, "q": "What is warmup_steps in the Transformer learning-rate schedule?",
     "e": "4000", "a": ["4000", "4,000"], "c": "training"},
    {"id": 22, "q": "What is beta_1 for the Adam optimizer used in the Transformer?",
     "e": "0.9", "a": ["0.9", "beta1=0.9", "beta_1=0.9"], "c": "training"},
    {"id": 23, "q": "What is beta_2 for the Adam optimizer used in the Transformer?",
     "e": "0.98", "a": ["0.98"], "c": "training"},
    {"id": 24, "q": "How many training steps were used for the base Transformer?",
     "e": "100000", "a": ["100,000", "100000", "100k"], "c": "training"},
    {"id": 25, "q": "How many training steps were used for the big Transformer?",
     "e": "300000", "a": ["300,000", "300000", "300k"], "c": "training"},
    {"id": 26, "q": "How many GPUs were used to train the Transformer?",
     "e": "8", "a": ["8", "eight"], "c": "training"},
    {"id": 27, "q": "What GPU model was used to train the Transformer?",
     "e": "P100", "a": ["p100"], "c": "training"},
    {"id": 28, "q": "Approximately how many sentence pairs are in WMT 2014 EN-DE?",
     "e": "4.5 million", "a": ["4.5", "4.5m", "4.5 million"], "c": "training"},
    {"id": 29, "q": "Approximately how many sentence pairs are in WMT 2014 EN-FR?",
     "e": "36 million", "a": ["36", "36m", "36 million"], "c": "training"},

    # ── results ───────────────────────────────────────────────────────────────
    {"id": 30, "q": "What BLEU score did the big Transformer achieve on WMT 2014 English-German?",
     "e": "28.4", "a": ["28.4"], "c": "results"},
    {"id": 31, "q": "What BLEU score did the big Transformer achieve on WMT 2014 English-French?",
     "e": "41.8", "a": ["41.8"], "c": "results"},
    {"id": 32, "q": "What BLEU score did the base Transformer achieve on WMT 2014 English-German?",
     "e": "27.3", "a": ["27.3"], "c": "results"},
    {"id": 33, "q": "What F1 did the 4-layer Transformer achieve on WSJ-only constituency parsing?",
     "e": "91.3", "a": ["91.3"], "c": "results"},
    {"id": 34, "q": "What F1 did the semi-supervised Transformer achieve on WSJ constituency parsing?",
     "e": "92.7", "a": ["92.7"], "c": "results"},
    {"id": 35, "q": "What beam size was used for Transformer inference on translation tasks?",
     "e": "4", "a": ["4", "beam=4", "beam size of 4"], "c": "results"},
    {"id": 36, "q": "What length penalty alpha was used in beam search?",
     "e": "0.6", "a": ["0.6", "alpha=0.6"], "c": "results"},

    # ── regularization ────────────────────────────────────────────────────────
    {"id": 37, "q": "What residual dropout rate P_drop was used in the base Transformer?",
     "e": "0.1", "a": ["0.1", "pdrop=0.1", "p_drop=0.1"], "c": "regularization"},
    {"id": 38, "q": "What is label smoothing epsilon_ls used in the Transformer?",
     "e": "0.1", "a": ["0.1"], "c": "regularization"},
    {"id": 39, "q": "What residual dropout rate was used for the big Transformer trained on English-French?",
     "e": "0.3", "a": ["0.3"], "c": "regularization"},
    {"id": 40, "q": "How many last checkpoints were averaged for the base model evaluation?",
     "e": "5", "a": ["5", "last 5", "five"], "c": "regularization"},
    {"id": 41, "q": "How many last checkpoints were averaged for the big model evaluation?",
     "e": "20", "a": ["20", "last 20", "twenty"], "c": "regularization"},
    {"id": 42, "q": "What technique was used to reduce overfitting via smoothed label distributions?",
     "e": "label smoothing", "a": ["label smoothing", "label-smoothing"], "c": "regularization"},
]


def make_row(d: dict) -> dict:
    return {
        "id": d["id"],
        "question": d["q"],
        "expected": d["e"],
        "accept": d["a"],
        "category": d["c"],
        "messages": [{"role": "user", "content": d["q"]}],
    }


def main() -> None:
    rows = [make_row(d) for d in EVAL]

    targets = [
        ("v2", r"w08v02\w08v02c01\data\eval_aiayn.jsonl"),
    ]
    base = pathlib.Path(__file__).parent

    for label, rel_path in targets:
        out = base / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{label}: {len(rows)} rows -> {out}")

    cats = {}
    for d in EVAL:
        cats[d["c"]] = cats.get(d["c"], 0) + 1
    print("Categories:", cats)


if __name__ == "__main__":
    main()
