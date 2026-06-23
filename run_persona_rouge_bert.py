import os
os.environ['HF_HOME'] = '/scratch/smansoo5/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/smansoo5/hf_cache'

import pandas as pd
from bert_score import score as bert_score
from rouge_score import rouge_scorer

models = {
    'AudioFlamingo': '/scratch/smansoo5/af3_results.csv',
    'Qwen': '/scratch/smansoo5/qwen_results.csv',
    'SALMONN': '/scratch/smansoo5/salmonn_results.csv',
}

rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

all_scores = []

for model_name, path in models.items():
    print(f"\n{'='*50}\nMODEL: {model_name}\n{'='*50}")
    df = pd.read_csv(path)
    df = df[df['error'].isna()]
    text_df = df[df['modality'] == 'text']
    audio_df = df[df['modality'] == 'audio']
    merged = pd.merge(text_df, audio_df, on=['file_id', 'persona', 'alert'], suffixes=('_text', '_audio'))

    texts = merged['response_text'].astype(str).tolist()
    audios = merged['response_audio'].astype(str).tolist()
    personas = merged['persona'].tolist()

    print("Running BERTScore...")
    P, R, F1 = bert_score(audios, texts, lang='en', verbose=True)

    for i, (t, a, p) in enumerate(zip(texts, audios, personas)):
        rouge_scores = rouge.score(t, a)
        all_scores.append({
            'model': model_name,
            'persona': p,
            'bertscore': F1[i].item(),
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
        })

all_df = pd.DataFrame(all_scores)

print("\n--- BERTScore By Model and Persona ---")
print(all_df.groupby(['persona', 'model'])['bertscore'].mean().unstack('model').round(4))
print("\n--- BERTScore By Persona (across all models) ---")
print(all_df.groupby('persona')['bertscore'].mean().round(4))

print("\n--- ROUGE-L By Model and Persona ---")
print(all_df.groupby(['persona', 'model'])['rougeL'].mean().unstack('model').round(4))
print("\n--- ROUGE-L By Persona (across all models) ---")
print(all_df.groupby('persona')['rougeL'].mean().round(4))
