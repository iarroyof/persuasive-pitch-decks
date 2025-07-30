# metrics.py (actualizado)

import pandas as pd
import ast
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import nltk

# Descargar recursos de NLTK 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def evaluate_metrics(df, output_path="./output/pitch_decks_metrics.csv"):
    rouge_scores, bleu_scores, meteor_scores, semantic_similarities = [], [], [], []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleu_smooth = SmoothingFunction().method4
    model = SentenceTransformer('all-MiniLM-L6-v2')

    for i, row in df.iterrows():
        original = str(row['original_text']).strip()
        created = str(row['created_text']).strip()

        # ROUGE-L
        rouge = scorer.score(original, created)
        rouge_scores.append({
            'Precision': round(rouge['rougeL'].precision, 3),
            'Recall': round(rouge['rougeL'].recall, 3),
            'F1': round(rouge['rougeL'].fmeasure, 3)
        })

        # BLEU
        bleu = sentence_bleu([original.lower().split()], created.lower().split(), smoothing_function=bleu_smooth)
        bleu_scores.append(round(bleu, 3))

        # Semantic Similarity
        embeddings = model.encode([original, created], convert_to_tensor=True)
        sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        semantic_similarities.append(round(sim, 3))

        # METEOR
        try:
            tokens_ref = word_tokenize(original.lower())
            tokens_hyp = word_tokenize(created.lower())
            meteor = meteor_score([tokens_ref], tokens_hyp)
            meteor_scores.append(round(meteor, 3))
        except Exception:
            meteor_scores.append(None)

    df['Rouge-L-Score'] = rouge_scores
    df['BLEU'] = bleu_scores
    df['Semantic_Similarity'] = semantic_similarities
    df['METEOR_Score'] = meteor_scores

    # Extraer componentes de ROUGE
    df['Rouge_Precision'] = df['Rouge-L-Score'].apply(lambda x: x['Precision'])
    df['Rouge_Recall'] = df['Rouge-L-Score'].apply(lambda x: x['Recall'])
    df['Rouge_F1'] = df['Rouge-L-Score'].apply(lambda x: x['F1'])

    df.to_csv(output_path, index=False)
    print(f"\n📊 Métricas guardadas en {output_path}")

    # Calcular resumen por modelo y tipo de prompt
    resumen = df.groupby(['model_name', 'prompt_type'])[
        ['BLEU', 'Semantic_Similarity', 'METEOR_Score', 'Rouge_Precision', 'Rouge_Recall', 'Rouge_F1']
    ].agg(['mean', 'std']).round(4)

    return df, resumen
