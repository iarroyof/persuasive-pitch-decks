import torch
import numpy as np
import networkx as nx
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Function to make logits consistent based on the hierarchy matrix R
def _make_logits_consistent(x, R):
    c_out = x.unsqueeze(1) + 10
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1]).to(x.device)
    final_out, _ = torch.max(R_batch * c_out, dim=2)
    return final_out - 10

# Function to initialize the model, tokenizer, and hierarchy matrix
def initialize_model():
    # Define the hierarchy graph
    G = nx.DiGraph()
    edges = [
        ("ROOT", "Logos"), 
            ("Logos", "Repetition"), ("Logos", "Obfuscation, Intentional vagueness, Confusion"), ("Logos", "Reasoning"), ("Logos", "Justification"), 
	            ("Justification", "Slogans"), ("Justification", "Bandwagon"), ("Justification", "Appeal to authority"), ("Justification", "Flag-waving"), ("Justification", "Appeal to fear/prejudice"),
	            ("Reasoning", "Simplification"), 
	            	("Simplification", "Causal Oversimplification"), ("Simplification", "Black-and-white Fallacy/Dictatorship"), ("Simplification", "Thought-terminating cliché"),  
                    ("Reasoning", "Distraction"),
                    	("Distraction", "Misrepresentation of Someone's Position (Straw Man)"), ("Distraction", "Presenting Irrelevant Data (Red Herring)"), ("Distraction", "Whataboutism"),
        ("ROOT", "Ethos"), 
            ("Ethos", "Appeal to authority"), ("Ethos", "Glittering generalities (Virtue)"), ("Ethos", "Bandwagon"), ("Ethos", "Ad Hominem"), ("Ethos", "Transfer"), 
                ("Ad Hominem", "Doubt"), ("Ad Hominem", "Name calling/Labeling"), ("Ad Hominem", "Smears"), ("Ad Hominem", "Reductio ad hitlerum"), ("Ad Hominem", "Whataboutism"), 
        ("ROOT", "Pathos"), 
            ("Pathos", "Exaggeration/Minimisation"), ("Pathos", "Loaded Language"), ("Pathos", "Appeal to (Strong) Emotions"), ("Pathos", "Appeal to fear/prejudice"), ("Pathos", "Flag-waving"), ("Pathos", "Transfer")
    ]
    G.add_edges_from(edges)
    
    # model and tokenizer is saved in the current directory
    model_dir = "."
    # loading the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create the hierarchy matrix R based on the graph structure
    A = nx.to_numpy_array(G).transpose()
    R = np.zeros(A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(A)
    for i in range(len(A)):
        descendants = list(nx.descendants(g, i))
        if descendants:
            R[i, descendants] = 1
    R = torch.tensor(R).transpose(1, 0).unsqueeze(0)
    
    return tokenizer, model, R, G, device

# Function to predict persuasion labels for a given text
def predict_persuasion_labels(text, tokenizer, model, R, G, device):
    # Tokenize and encode the input text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
        )

    # Make logits consistent based on the hierarchy matrix R
    logits = _make_logits_consistent(outputs.logits, R)
    logits[:, 0] = -1.0
    logits = logits > 0.0

    # Get the complete predicted hierarchy of labels
    complete_predicted_hierarchy = np.array(G.nodes)[logits[0].cpu().nonzero()].flatten().tolist()

    # Get the child-only labels (labels without any successors)
    child_only_labels = []
    for label in complete_predicted_hierarchy:
        if not list(G.successors(label)):
            child_only_labels.append(label)

    return complete_predicted_hierarchy, child_only_labels

tokenizer, model, R, G, device = initialize_model()

# Main inference function
def inference(text):
    return predict_persuasion_labels(text, tokenizer, model, R, G, device)

if __name__ == "__main__":
    # ask the user for input
    text = input("Enter the text: ")
    print(inference(text))