import os
import pandas as pd
from transformers import pipeline

# Configuration
MODEL_NAME = "nishan-chatterjee/multilingual-persuasion-detection-from-text"
INPUT_CSV = "data/pitches_combinados.csv"
OUTPUT_CSV = "output_with_classifications.csv"
OUTPUT_DIR = "grouped_outputs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_CSV, sep='\t')  # adjust delimiter if needed

# Initialize the classification pipeline
classifier = pipeline(
    task="text-classification",
    model=MODEL_NAME,
    return_all_scores=False
)

# Classification function
def classify_text(text: str) -> dict:
    """
    Classify a single text and return the label and score.
    """
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return {"label": None, "score": None}
    result = classifier(text)[0]
    return {"label": result.get("label"), "score": result.get("score")}

# Apply classification to original_text and created_text
df_orig = df["original_text"].apply(classify_text).apply(pd.Series)
def created_classifications(text):
    return pd.Series(classify_text(text), index=["created_label", "created_score"])

df_created = df["created_text"].apply(created_classifications)

# Combine results into original dataframe
df = pd.concat([df, df_orig.rename(columns={"label": "original_label", "score": "original_score"}), df_created], axis=1)

# Save full output
df.to_csv(OUTPUT_CSV, index=False)
print(f"Full classified data saved to {OUTPUT_CSV}")

# Separate created_text classifications by prompt_type and model_name
groups = df.groupby(["prompt_type", "model_name"])
for (prompt, model), group in groups:
    # Only include columns relevant to created_text classification and grouping keys
    subset = group[["category_name", "prompt_type", "model_name", "created_text", "created_label", "created_score"]]
    # Generate a safe filename
    prompt_safe = prompt.replace("/", "_").replace(" ", "_")
    model_safe = model.replace("/", "_").replace(" ", "_")
    filename = f"created_classifications_{prompt_safe}__{model_safe}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    subset.to_csv(filepath, index=False)
    print(f"Group created_text classifications for prompt_type='{prompt}', model_name='{model}' saved to {filepath}")
