"""
SALMONN — full 320-run inference script
Text-only:  silence.wav (dummy) + full text prompt
Audio-only: real wav + instruction prompt
Output: /scratch/smansoo5/salmonn_results.csv
"""

import os
import sys
import csv
import torch
import shutil
import traceback
import numpy as np
import soundfile as sf

sys.path.insert(0, '/scratch/smansoo5/salmonn_inference')
from model import SALMONN

AUDIO_DIR   = "/scratch/smansoo5/tts_dataset"
SILENCE_WAV = "/scratch/smansoo5/salmonn_weights/silence.wav"
OUTPUT_CSV  = "/scratch/smansoo5/salmonn_results.csv"

# create silence.wav if it doesn't exist
if not os.path.exists(SILENCE_WAV):
    silence = np.zeros(16000, dtype=np.float32)
    sf.write(SILENCE_WAV, silence, 16000)
    print("Created silence.wav", flush=True)

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

def build_audio_path(file_id, persona, alert):
    p = persona.replace(" ", "_")
    a = alert.replace(" ", "_")
    return os.path.join(AUDIO_DIR, f"{file_id}_{p}_{a}.wav")

entries = []
file_id = 0
for persona in PERSONAS:
    for alert in ALERTS:
        entries.append({
            "file_id":    file_id,
            "persona":    persona,
            "alert":      alert,
            "prompt":     build_prompt(persona, alert),
            "audio_path": build_audio_path(file_id, persona, alert),
        })
        file_id += 1

fieldnames = ["file_id", "persona", "alert", "prompt", "modality", "response", "error"]

# strip failed rows, build completed set
completed = set()
if os.path.exists(OUTPUT_CSV):
    tmp = OUTPUT_CSV + ".tmp"
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as fin, \
         open(tmp, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            if row["response"].strip():
                writer.writerow(row)
                completed.add((int(row["file_id"]), row["modality"]))
    shutil.move(tmp, OUTPUT_CSV)
    print(f"Resuming — {len(completed)} successful rows kept.", flush=True)

print("Loading SALMONN...", flush=True)
model = SALMONN(
    ckpt='/scratch/smansoo5/salmonn_weights/salmonn_7b_v0.pth',
    whisper_path='/scratch/smansoo5/salmonn_weights/whisper-large-v2',
    beats_path='/scratch/smansoo5/salmonn_weights/BEATs_iter3_plus_AS2M.pt',
    vicuna_path='/scratch/smansoo5/salmonn_weights/vicuna-7b-v1.5',
    low_resource=False,
)
model = model.to('cuda:0')
model.eval()
print("Model loaded.", flush=True)

def run_text(prompt):
    with torch.cuda.amp.autocast(dtype=torch.float16):
        result = model.generate(
            wav_path=SILENCE_WAV,
            prompt=prompt,
        )
    return result[0]

def run_audio(audio_path):
    with torch.cuda.amp.autocast(dtype=torch.float16):
        result = model.generate(
            wav_path=audio_path,
            prompt="Please provide safety instructions in response to the audio.",
        )
    return result[0]

# main loop
write_header = not os.path.exists(OUTPUT_CSV)
total = len(entries) * 2
done  = len(completed)

with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for entry in entries:
        fid = entry["file_id"]

        if (fid, "audio") not in completed:
            done += 1
            print(f"[{done}/{total}] AUDIO fid={fid}  {entry['persona']} | {entry['alert']}", flush=True)
            try:
                if not os.path.exists(entry["audio_path"]):
                    raise FileNotFoundError(f"WAV not found: {entry['audio_path']}")
                response = run_audio(entry["audio_path"])
                error = ""
            except Exception as e:
                response = ""; error = traceback.format_exc()
                print(f"  ERROR: {e}", flush=True)
            writer.writerow({"file_id": fid, "persona": entry["persona"], "alert": entry["alert"],
                             "prompt": entry["prompt"], "modality": "audio", "response": response, "error": error})
            csvfile.flush()

        if (fid, "text") not in completed:
            done += 1
            print(f"[{done}/{total}] TEXT  fid={fid}  {entry['persona']} | {entry['alert']}", flush=True)
            try:
                response = run_text(entry["prompt"])
                error = ""
            except Exception as e:
                response = ""; error = traceback.format_exc()
                print(f"  ERROR: {e}", flush=True)
            writer.writerow({"file_id": fid, "persona": entry["persona"], "alert": entry["alert"],
                             "prompt": entry["prompt"], "modality": "text", "response": response, "error": error})
            csvfile.flush()

print("All done!", flush=True)
