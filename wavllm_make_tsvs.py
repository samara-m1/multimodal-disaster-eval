"""
WavLLM — generate TSV files for all 320 runs
Run this FIRST before submitting the sbatch job.
"""

import os
import soundfile as sf
import numpy as np

AUDIO_DIR   = "/scratch/smansoo5/tts_dataset"
SILENCE_WAV = "/scratch/smansoo5/salmonn_weights/silence.wav"
TSV_DIR     = "/scratch/smansoo5/fairseq-main/examples/wavllm/test_data"

os.makedirs(TSV_DIR, exist_ok=True)

# make silence.wav if missing
if not os.path.exists(SILENCE_WAV):
    sf.write(SILENCE_WAV, np.zeros(16000, dtype=np.float32), 16000)
    print("Created silence.wav")

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

text_tsv  = os.path.join(TSV_DIR, "all_text_320.tsv")
audio_tsv = os.path.join(TSV_DIR, "all_audio_320.tsv")

header = "id\taudio\tn_frames\tprompt\ttgt_text\twith_speech\n"

with open(text_tsv, "w") as ft, open(audio_tsv, "w") as fa:
    ft.write(header)
    fa.write(header)

    idx = 0
    for persona in PERSONAS:
        for alert in ALERTS:
            prompt     = build_prompt(persona, alert)
            audio_path = build_audio_path(idx, persona, alert)

            # TEXT: silence.wav + full prompt, with_speech=False
            ft.write(f"{idx}\t{SILENCE_WAV}\t16000\t{prompt}\tunknown\tFalse\n")

            # AUDIO: real wav + minimal prompt, with_speech=True
            try:
                data, _ = sf.read(audio_path)
                n_frames = len(data)
            except Exception:
                n_frames = 16000

            fa.write(f"{idx}\t{audio_path}\t{n_frames}\tPlease provide safety instructions in response to the audio.\tunknown\tTrue\n")

            idx += 1

print(f"TSVs written: {text_tsv}")
print(f"             {audio_tsv}")
print(f"Total rows each: {idx}")
