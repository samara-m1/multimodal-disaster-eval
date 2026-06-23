import os
os.environ['HF_HOME'] = '/scratch/smansoo5/hf_cache'

import pandas as pd
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli')
nli = pipeline('zero-shot-classification', model='roberta-large-mnli', device=0)

def truncate(text, max_tokens=200):
    tokens = tokenizer(text, truncation=True, max_length=max_tokens, return_tensors='pt')
    return tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)

models = {
    'AudioFlamingo': '/scratch/smansoo5/af3_results.csv',
    'Qwen': '/scratch/smansoo5/qwen_results.csv',
    'SALMONN': '/scratch/smansoo5/salmonn_results.csv',
}

all_results = []

for model_name, path in models.items():
    df = pd.read_csv(path)
    df = df[df['error'].isna() | (df['error'] == '')]
    text_df = df[df['modality'] == 'text']
    audio_df = df[df['modality'] == 'audio']
    merged = pd.merge(text_df, audio_df, on=['file_id', 'persona', 'alert'], suffixes=('_text', '_audio'))

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc=model_name):
        t = truncate(str(row['response_text']))
        a = truncate(str(row['response_audio']))
        result = nli(t, candidate_labels=[a], hypothesis_template='{}')
        score = result['scores'][0]
        all_results.append({'model': model_name, 'persona': row['persona'], 'alert': row['alert'], 'factsumm_score': score})

results_df = pd.DataFrame(all_results)
print("\nBy Model:")
print(results_df.groupby('model')['factsumm_score'].mean().round(4))
print("\nBy Persona:")
print(results_df.groupby('persona')['factsumm_score'].mean().round(4))
print("\nBy Model + Persona:")
print(results_df.groupby(['model', 'persona'])['factsumm_score'].mean().round(4))

results_df.to_csv('/scratch/smansoo5/factsumm_roberta_results.csv', index=False)
print("\nSaved to /scratch/smansoo5/factsumm_roberta_results.csv")
