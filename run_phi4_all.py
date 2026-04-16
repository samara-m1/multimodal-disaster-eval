"""
Phi-4 Multimodal — full 320-run inference script
Based on working phi_test.py
Text: standard prompt format <|user|>...<|end|><|assistant|>
Audio: speech LoRA adapter + <|audio_1|> token format
Output: /scratch/smansoo5/phi4_results.csv
"""

import os
import csv
import shutil
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

MODEL_ID   = "microsoft/Phi-4-multimodal-instruct"
AUDIO_DIR  = "/scratch/smansoo5/tts_dataset"
OUTPUT_CSV = "/scratch/smansoo5/phi4_results.csv"

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

print("Loading Phi-4 processor...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Loading Phi-4 config...", flush=True)
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config._attn_implementation = "eager"

print("Loading Phi-4 model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
model.eval()

print("Loading speech LoRA adapter...", flush=True)
model.load_adapter(MODEL_ID, adapter_name="speech", device_map="cuda",
                   adapter_kwargs={"subfolder": "speech-lora"})
print("Model ready!", flush=True)

def run_text(prompt):
    model.disable_adapters()
    text = f"<|user|>{prompt}<|end|><|assistant|>"
    inputs = processor(text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    full = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    if "<|assistant|>" in full:
        return full.split("<|assistant|>")[-1].strip()
    return full.replace(prompt, "").strip()

def run_audio(audio_path):
    import soundfile as sf
    model.set_adapter("speech")
    audio, sr = sf.read(audio_path)
    text = "<|user|><|audio_1|>Please provide safety instructions in response to the audio.<|end|><|assistant|>"
    inputs = processor(text=text, audios=[(audio, sr)], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

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
