import gradio as gr
import torch
import numpy as np
import networkx as nx
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

def _make_logits_consistent(x, R):
    c_out = x.unsqueeze(1) + 10
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1]).to(x.device)
    final_out, _ = torch.max(R_batch * c_out, dim=2)
    return final_out - 10

def persuasion_labels(text):
    model_dir = "models"
    # Initialize the graph and other necessary components
    G = nx.DiGraph()
    # Add edges to the graph
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
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    A = nx.to_numpy_array(G).transpose()
    R = np.zeros(A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(A)
    for i in range(len(A)):
        descendants = list(nx.descendants(g, i))
        if descendants:
            R[i, descendants] = 1
    R = torch.tensor(R).transpose(1, 0).unsqueeze(0)
    
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

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
        )
    logits = _make_logits_consistent(outputs.logits, R)
    logits[:, 0] = -1.0
    logits = logits > 0.0
    complete_predicted_hierarchy = np.array(G.nodes)[logits[0].cpu().nonzero()].flatten().tolist()

    # if any label doesn't have children, add them to the list
    child_only_labels = []
    for label in complete_predicted_hierarchy:
        if not list(G.successors(label)):
            child_only_labels.append(label)

    return complete_predicted_hierarchy, child_only_labels

def launch_interface():
    iface = gr.Interface(
        fn=persuasion_labels,
        inputs=gr.Textbox(lines=5, placeholder="Enter your text here..."),
        outputs=[
            gr.Textbox(label="Complete Hierarchical Label List"),
            gr.Textbox(label="Child-only Label List")
        ],
        title="Persuasion Labels",
        description="Enter your text and get the persuasion labels.",
    )
    iface.launch(share=True)

if __name__ == "__main__":
    launch_interface()