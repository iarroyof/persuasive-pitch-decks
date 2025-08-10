import torch
import numpy as np
import networkx as nx
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def _make_logits_consistent(x, R):
    c_out = x.unsqueeze(1) + 10
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1]).to(x.device)
    final_out, _ = torch.max(R_batch * c_out, dim=2)
    return final_out - 10

def initialize_model():

    model_name = "nishan-chatterjee/multilingual-persuasion-detection-from-text"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

 # Obtener etiquetas desde la configuración del modelo
    if model.config.id2label:
        id2label = model.config.id2label
        labels = [id2label[i] for i in range(len(id2label))]
    else:
        raise ValueError("No se encontraron etiquetas en model.config.id2label.")

    # Crear grafo G solo con etiquetas
    G = nx.DiGraph()
    G.add_node("ROOT")
    for label in labels:
        G.add_edge("ROOT", label)

    # Matriz de adyacencia y jerarquía
    A = nx.to_numpy_array(G).transpose()
    R = np.zeros(A.shape)
    np.fill_diagonal(R, 1)

    g = nx.DiGraph(A)
    for i in range(len(A)):
        descendants = list(nx.descendants(g, i))
        if descendants:
            R[i, descendants] = 1

    R = torch.tensor(R, dtype=torch.float32).transpose(1, 0).unsqueeze(0)

    return tokenizer, model, R, G, device

def predict_persuasion_labels(text, tokenizer, model, R, G, device):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(input_ids=encoding["input_ids"].to(device),
                        attention_mask=encoding["attention_mask"].to(device))

    logits = _make_logits_consistent(outputs.logits, R)
    logits[:, 0] = -1.0
    selection = logits[0].cpu().numpy() > 0

    nodes = np.array(list(G.nodes))
    complete = nodes[selection].tolist()
    child_only = [lab for lab in complete if not list(G.successors(lab))]

    return complete, child_only

tokenizer, model, R, G, device = initialize_model()

def inference(text):
    return predict_persuasion_labels(text, tokenizer, model, R, G, device)
