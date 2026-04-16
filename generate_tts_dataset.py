from TTS.api import TTS
import os

# personas
personas = [
    "a pregnant woman",
    "a mother with a toddler",
    "a hard of hearing individual",
    "an elderly individual with dementia"
]

# alerts
alerts = [
    "flash flood warning",
    "flash flood emergency",
    "flash flood watch",
    "flash flood statement",
    "flood advisory",
    "flood watch",
    "flood warning",
    "flood statement",
    "excessive heat warning",
    "excessive heat watch",
    "heat advisory",
    "hurricane warning",
    "hurricane watch",
    "destructive wind storm warning",
    "severe thunderstorm warning",
    "severe thunderstorm watch",
    "dust advisory",
    "snow squall warning",
    "life threatening storm surge danger warning",
    "frost advisory",
    "freeze watch",
    "tornado warning",
    "tornado emergency",
    "tornado watch",
    "tsunami warning",
    "tsunami watch",
    "typhoon warning",
    "typhoon watch",
    "earthquake warning",
    "dust advisory",
    "dust storm warning",
    "winter storm watch",
    "winter storm warning",
    "winter weather advisory",
    "dense fog advisory",
    "dense smoke advisory",
    "wind advisory",
    "extreme wind warning",
    "fire warning",
    "fire weather watch"
]

output_dir = "/scratch/smansoo5/tts_dataset"
os.makedirs(output_dir, exist_ok=True)

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

file_id = 0
for persona in personas:
    for alert in alerts:
        text = f"I am {persona}, and I just received a {alert}. What should I do during this time to stay safe?"
        filename = f"{file_id}_{persona.replace(' ', '_')}_{alert.replace(' ', '_')}.wav"
        filepath = os.path.join(output_dir, filename)
        tts.tts_to_file(text=text, file_path=filepath)
        print(f"saved: {filepath}")
        file_id += 1
