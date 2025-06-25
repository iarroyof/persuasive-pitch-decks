import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the full classifications
df = pd.read_csv('output_with_classifications.csv')

# Parse the stringified lists in 'created_complete_hierarchy'
# It assumes lists are stored as "['Label1', 'Label2']"
df['created_complete_hierarchy'] = df['created_complete_hierarchy'].apply(eval)

# Explode to have one label per row
df_exploded = df.explode('created_complete_hierarchy')

# Prepare output directory for plots
PLOTS_DIR = 'histogram_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Group by prompt_type and model_name to generate histograms
groups = df_exploded.groupby(['prompt_type', 'model_name'])

for (prompt, model), group in groups:
    # Count label frequencies
    label_counts = group['created_complete_hierarchy'].value_counts()
    
    # Skip empty groups
    if label_counts.empty:
        continue
    
    # Create histogram
    plt.figure()
    plt.bar(label_counts.index, label_counts.values)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Label Distribution\nPrompt: {prompt} | Model: {model}')
    plt.tight_layout()
    
    # Save plot
    filename = f'histogram_{prompt.replace("/", "_")}_{model.replace("/", "_")}.png'
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    
    print(f"Saved histogram for prompt_type='{prompt}', model_name='{model}' to {filepath}")
