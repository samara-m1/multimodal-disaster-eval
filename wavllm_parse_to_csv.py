"""
WavLLM — parse output text files from fairseq → unified CSV
Run this AFTER both sbatch jobs (audio + text) complete.

Fairseq output lines look like:
  D-0  some response text here
  D-1  another response
"""

import os
import re
import csv

PERSONAS = [
    "a pregnant woman",
    "a mother with a toddler",
    "a hard of hearing individual",
    "an elderly individual with dementia",
]

ALERTS = [
    "flash flood warning","flash flood emergency","flash flood watch","flash flood statement",
    "flood advisory","flood watch","flood warning","flood statement",
    "excessive heat warning","excessive heat watch","heat advisory",
    "hurricane warning","hurricane watch","destructive wind storm warning",
    "severe thunderstorm warning","severe thunderstorm watch",
    "dust advisory","snow squall warning","life threatening storm surge danger warning",
    "frost advisory","freeze watch",
    "tornado warning","tornado emergency","tornado watch",
    "tsunami warning","tsunami watch",
    "typhoon warning","typhoon watch",
    "earthquake warning","dust advisory","dust storm warning",
    "winter storm watch","winter storm warning","winter weather advisory",
    "dense fog advisory","dense smoke advisory",
    "wind advisory","extreme wind warning",
    "fire warning","fire weather watch",
]

def build_prompt(persona, alert):
    return f"I am {persona}, and I just received a {alert}. What should I do during this time to stay safe?"

# build metadata table so we know persona/alert per file_id
metadata = {}
file_id = 0
for persona in PERSONAS:
    for alert in ALERTS:
        metadata[file_id] = {"persona": persona, "alert": alert, "prompt": build_prompt(persona, alert)}
        file_id += 1

# ── find output files (fairseq puts them in decode_* dirs) ──────────────────
DECODE_BASE = "/scratch/smansoo5/wavllm_weights"

def find_output_file(subset_name):
    """Walk decode dirs to find generate-{subset}.txt"""
    for root, dirs, files in os.walk(DECODE_BASE):
        for fname in files:
            if fname == f"generate-{subset_name}.txt":
                return os.path.join(root, fname)
    return None

audio_file = find_output_file("all_audio_320")
text_file  = find_output_file("all_text_320")

if not audio_file:
    print("WARNING: audio output file not found — run audio sbatch job first")
if not text_file:
    print("WARNING: text output file not found — run text sbatch job first")

def parse_output(filepath):
    results = {}
    if not filepath or not os.path.exists(filepath):
        return results
    with open(filepath, "r") as f:
        for line in f:
            m = re.match(r"D-(\d+)\s+(.*)", line)
            if m:
                results[int(m.group(1))] = m.group(2).strip()
    return results

audio_results = parse_output(audio_file)
text_results  = parse_output(text_file)

OUTPUT_CSV = "/scratch/smansoo5/wavllm_results.csv"
fieldnames = ["file_id", "persona", "alert", "prompt", "modality", "response", "error"]

rows = []
for fid in range(160):
    meta = metadata[fid]
    for modality, results in [("audio", audio_results), ("text", text_results)]:
        rows.append({
            "file_id":  fid,
            "persona":  meta["persona"],
            "alert":    meta["alert"],
            "prompt":   meta["prompt"],
            "modality": modality,
            "response": results.get(fid, ""),
            "error":    "" if fid in results else "missing from output",
        })

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV saved: {OUTPUT_CSV}  ({len(rows)} rows)")
missing = sum(1 for r in rows if r["error"])
if missing:
    print(f"WARNING: {missing} missing responses — check if both jobs finished")
