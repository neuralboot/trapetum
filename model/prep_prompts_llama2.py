#!/usr/bin/env python3
"""
Write the 12 prompts of the Llama-2 speculative-decoding benchmark (runtime/src/bin/spec_ar.rs) as
JSON lines {"set": ..., "ids": [...]}: 8 windows of 64 tokens of the wikitext-2 test text, tokenized
as for the perplexity (token offsets 2,000 to 210,000), and 4 short prompts, all with BOS, Llama-2
tokenizer. These are the prompts of the 7 November 2026 run (Llama-2-70B AQLM 1x16 with a
Llama-2-7B drafter).

  python model/prep_prompts_llama2.py prompts.jsonl

No em dashes or en dashes anywhere.
"""
import json
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

OFFSETS = [2000, 30000, 60000, 90000, 120000, 150000, 180000, 210000]
SHORT = ['def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
         "The history of the Roman Empire began",
         "Question: What are the main causes of climate change?\nAnswer:",
         "Ingredients:\n- 2 cups of flour\n- 1 cup of sugar\n"]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "prompts.jsonl"
    tok = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(data["text"])).input_ids
    lines = [{"set": "wiki", "ids": [tok.bos_token_id] + enc[off:off + 64]} for off in OFFSETS]
    lines += [{"set": "misc", "ids": tok(t).input_ids} for t in SHORT]
    with open(out, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    print("wrote", out, ":", len(lines), "prompts, lengths", [len(line["ids"]) for line in lines])


if __name__ == "__main__":
    main()
