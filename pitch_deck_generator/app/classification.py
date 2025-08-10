import os
import pandas as pd
import re
from inference import inference  # inference(text) -> (complete_labels, child_labels)
import matplotlib.pyplot as plt
from collections import Counter

def clean_created_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    parts = cleaned.split('---', 1)
    return parts[0].strip()

def classify_and_plot(df, output_csv="output_with_classifications.csv", img_dir="./output/img", grouped_dir="./output/grouped"):
    # Preparar directorios
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(grouped_dir, exist_ok=True)

    # Limpiar texto
    for col in ["original_text", "created_text"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\r?\n", " ", regex=True)
    df["created_text"] = df["created_text"].apply(clean_created_text)

    # Clasificación
    def classify_text_with_hierarchy(text: str) -> dict:
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return {"complete_hierarchy": [], "child_only": []}
        return inference(text)

    print("Clasificando original_text y created_text...")
    df_orig = df["original_text"].apply(lambda t: pd.Series(dict(zip(["complete_hierarchy", "child_only"], classify_text_with_hierarchy(t)))))
    df_created = df["created_text"].apply(lambda t: pd.Series(dict(zip(["complete_hierarchy", "child_only"], classify_text_with_hierarchy(t)))))
    print("Clasificación completada.")

    # Renombrar y combinar
    df_orig.columns = ["orig_complete_hierarchy", "orig_child_only"]
    df_created.columns = ["created_complete_hierarchy", "created_child_only"]
    df_final = pd.concat([df, df_orig, df_created], axis=1)

    # Guardar CSV completo
    df_final.to_csv(output_csv, index=False)
    print(f"Clasificaciones guardadas en {output_csv}")

    # Guardar agrupados por modelo/prompt
    groups = df_final.groupby(["prompt_type", "model_name"])
    for (prompt, model), group in groups:
        subset = group[["category_name", "prompt_type", "model_name", "created_text", "created_complete_hierarchy", "created_child_only"]]
        prompt_safe = prompt.replace("/", "_").replace(" ", "_")
        model_safe = model.replace("/", "_").replace(" ", "_")
        filename = f"created_classifications_{prompt_safe}__{model_safe}.csv"
        subset.to_csv(os.path.join(grouped_dir, filename), index=False)

    # Histograma
    def generate_histograms(df, col, label_type):
        exploded = df.explode(col)
        exploded[col] = exploded[col].fillna("")
        for (prompt, model), group in exploded.groupby(["prompt_type", "model_name"]):
            counts = Counter(group[col])
            counts.pop("", None)
            if not counts:
                continue
            labels, values = zip(*counts.items())
            plt.figure()
            plt.bar(labels, values)
            plt.xticks(rotation=45, ha='right')
            plt.title(f"{label_type.capitalize()} Label Distribution\nPrompt: {prompt} | Model: {model}")
            plt.tight_layout()
            fname = f"hist_{label_type}_{prompt.replace('/','_')}_{model.replace('/','_')}.png"
            plt.savefig(os.path.join(img_dir, fname))
            plt.close()
            print(f"Histograma guardado: {fname}")

    for text_type, prefix in [("original", "orig"), ("created", "created")]:
        for part in ["complete_hierarchy", "child_only"]:
            col_name = f"{prefix}_{part}"
            generate_histograms(df_final, col_name, f"{text_type}_{part}")

    print("Clasificación e histogramas completados.")
    return df_final
