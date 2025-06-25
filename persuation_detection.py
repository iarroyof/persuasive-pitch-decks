import os
import pandas as pd
import re
from inference import inference  # inference(text) -> (complete_labels, child_labels)
import matplotlib.pyplot as plt

# Configuration
INPUT_CSV = "pitches_combinados.csv"
OUTPUT_CSV = "output_with_classifications.csv"
OUTPUT_DIR = "grouped_outputs"
PLOTS_DIR = "histogram_plots"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load data (comma-separated CSV)
df = pd.read_csv(INPUT_CSV)

# Clean newline characters in text fields for CSV integrity
for col in ["original_text", "created_text"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(r"\r?\n", " ", regex=True)

# Additional cleaning for created_text:
# 1) Remove content within <think>...</think>
# 2) Remove any trailing report-like sections starting with '---'
def clean_created(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    parts = cleaned.split('---', 1)
    return parts[0].strip()

if 'created_text' in df.columns:
    df['created_text'] = df['created_text'].apply(clean_created)

# Classification function
def classify_text_with_hierarchy(text: str) -> dict:
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return {"complete_hierarchy": [], "child_only": []}
    return inference(text)

# Apply classification
print("Starting classification for original_text and created_text...")
df_orig = df["original_text"].apply(lambda t: pd.Series(dict(zip(["complete_hierarchy","child_only"], classify_text_with_hierarchy(t)))))
df_created = df["created_text"].apply(lambda t: pd.Series(dict(zip(["complete_hierarchy","child_only"], classify_text_with_hierarchy(t)))))
print("Classification completed.")

# Rename columns
df_orig.columns = ["orig_complete_hierarchy", "orig_child_only"]
df_created.columns = ["created_complete_hierarchy", "created_child_only"]

# Combine into final dataframe
df_final = pd.concat([df, df_orig, df_created], axis=1)

# Save full classifications
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"Saved full classifications to {OUTPUT_CSV}")

# Grouped CSV outputs for created_text only
groups = df_final.groupby(["prompt_type", "model_name"])
for (prompt, model), group in groups:
    subset = group[["category_name","prompt_type","model_name","created_text","created_complete_hierarchy","created_child_only"]]
    prompt_safe = prompt.replace("/","_").replace(" ","_")
    model_safe = model.replace("/","_").replace(" ","_")
    filename = f"created_classifications_{prompt_safe}__{model_safe}.csv"
    subset.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

# Function to generate and save histograms for a given column label
from collections import Counter

def generate_histograms(df, col, label_type):
    exploded = df.explode(col)
    exploded[col] = exploded[col].fillna("")
    for (prompt, model), group in exploded.groupby(["prompt_type", "model_name"]):
        counts = Counter(group[col])
        # Remove empty label key if present
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
        plt.savefig(os.path.join(PLOTS_DIR, fname))
        plt.close()
        print(f"Saved {label_type} histogram for prompt='{prompt}', model='{model}' to {fname}")

# Generate histograms for both complete_hierarchy and child_only for original and created
for text_type, prefix in [("original", "orig"), ("created", "created")]:
    for part in ["complete_hierarchy", "child_only"]:
        col_name = f"{prefix}_{part}"
        generate_histograms(df_final, col_name, f"{text_type}_{part}")

print("All tasks completed.")
