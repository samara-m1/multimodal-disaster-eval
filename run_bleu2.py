import os
os.environ['HF_HOME'] = '/scratch/smansoo5/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/smansoo5/hf_cache'

import pandas as pd
from sacrebleu.metrics import BLEU

bleu_metric = BLEU(effective_order=True)

models = {
    'AudioFlamingo': '/scratch/smansoo5/af3_results.csv',
    'Qwen': '/scratch/smansoo5/qwen_results.csv',
    'SALMONN': '/scratch/smansoo5/salmonn_results.csv',
}

all_scores = []

for model_name, path in models.items():
    print(f"\n{'='*50}\nMODEL: {model_name}\n{'='*50}")
    df = pd.read_csv(path)
    df = df[df['error'].isna()]
    text_df = df[df['modality'] == 'text']
    audio_df = df[df['modality'] == 'audio']
    merged = pd.merge(text_df, audio_df, on=['file_id', 'persona', 'alert'], suffixes=('_text', '_audio'))

    for _, r in merged.iterrows():
        score = bleu_metric.sentence_score(str(r['response_audio']), [str(r['response_text'])]).score / 100
        all_scores.append({'model': model_name, 'persona': r['persona'], 'score': score})

all_df = pd.DataFrame(all_scores)

print("\n--- BLEU By Model and Persona ---")
print(all_df.groupby(['persona', 'model'])['score'].mean().unstack('model').round(4))

print("\n--- BLEU By Persona (across all models) ---")
print(all_df.groupby('persona')['score'].mean().round(4))

print("\n--- BLEU Overall by Model ---")
print(all_df.groupby('model')['score'].mean().round(4))
