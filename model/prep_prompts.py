#!/usr/bin/env python3
"""Benchmark prompts for bin/pard_bench as token ids, chat-templated exactly as PARD's pard2_infer.get_input
does (system "You are a helpful assistant.", the user prompt, generation prompt added, no extra special
tokens): one {"set", "ids"} JSON line per prompt.
  python model/prep_prompts.py <target tokenizer dir> <PARD checkout> <out.jsonl> [sets] [per_set]
per_set > 0 keeps the first per_set prompts of each set (the Llama-3.1-70B-Instruct run of 8 November 2026
used 4: 16 prompts).
The PARD checkout provides datas/bmk/<set>.jsonl (gsm8k, humaneval, math_500 at AMD-AGI/PARD commit
6f279bf); mt_bench.jsonl can be built from the first turns of FastChat's MT-Bench questions."""
import json, sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
root, out = sys.argv[2], sys.argv[3]
sets = (sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else "gsm8k,humaneval,math_500,mt_bench").split(",")
per_set = int(sys.argv[5]) if len(sys.argv) > 5 else 0
n, longest = 0, 0
with open(out, "w") as f:
    for s in sets:
        kept = 0
        for line in open("%s/datas/bmk/%s.jsonl" % (root, s)):
            if not line.strip():
                continue
            if per_set and kept >= per_set:
                break
            kept += 1
            msgs = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": json.loads(line)["data"]}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok([text], add_special_tokens=False).input_ids[0]
            f.write(json.dumps({"set": s, "ids": ids}) + "\n")
            n, longest = n + 1, max(longest, len(ids))
print("PROMPTS %d prompts, longest %d tokens, first ids %s" % (n, longest, ids[:4]))
