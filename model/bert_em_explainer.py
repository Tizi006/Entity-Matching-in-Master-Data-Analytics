# !pip install -q transformers captum pandas
# !mkdir -p models 

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from captum.attr import IntegratedGradients
import json
import pandas as pd
import os
import sys
import io

'''
Sowohl deepmatcher als auch ditto haben einige dependency-probleme verursacht (gerade deepmatcher hat einige probleme wegen älteren Pythonversionen und Linux-exclusives).
Das hier ist ist jetzt ein Modell, welches ein pretrained model (BERT) verwendet. (Quasi wie Ditto)
Vorteil:
- keine nennenswerten dependency probleme
- code läuft
- 

mögliche Nachteile:
- wahrscheinlich etwas ungenauer (habe mich damit nicht besonders auseinander gesetzt)
- mehr custom-code (kann man aber als mehraufwand verkaufen)
- der code war bei mir sehr langsam...


Mit Shap hatte ich auch ein paar dependency probleme (eig. nur python-Version).
Bin deswegen auf Captum umgestiegen. (ist ähnlich zu SHAP (kann unter anderem auch gradient-shap))
Es ist im gegenssatz zu shap nicht modelagnostisch, aber für deep-learning spezialisiert
Hab damit noch nicht so viel gemacht...

Vorteil (und evtl auch Nachteil):
- ich hab noch nicht viel dazu gesehen
'''


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') # evtl lieber nicht buffern, da man sonst erst am ende was sieht...

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/em_bert_model.pt" 
BATCH_SIZE = 16 #bei cpu-training weniger (4 oder 8)
NUM_EPOCHS = 3 #mehr wäre eig nice, aber des dauert bei mir viel zu lange. Man könnte die Dateigröße reduzieren...

print("Device:", DEVICE)
print(f"Modell wird gespeichert unter: {MODEL_PATH}")


def load_data_from_file(
    file_path: str, 
    text_cols_left: list[str], 
    text_cols_right: list[str], 
    label_col: str = 'label'
):
    """
    Lädt Daten aus einer Datei (CSV oder JSONL) und formatiert sie für das EM-Modell.
    
    Args:
        file_path (str): Pfad zur Datenquelle.
        text_cols_left (list[str]): Liste der Spaltennamen für die linke Entität.
        text_cols_right (list[str]): Liste der Spaltennamen für die rechte Entität.
        label_col (str): Name der Spalte, die das Label (0 oder 1) enthält.
    """
    
    print(f"Lade Daten von: {file_path}")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
    else:
        raise ValueError("Unterstützte Dateiformate sind CSV oder JSONL.")

    formatted_data = []
    
    for _, row in df.iterrows():
        try:
            label = int(row[label_col])
        except (KeyError, ValueError) as e:
            print(f"Fehler beim Parsen des Labels in Zeile: {row}. Fehler: {e}")
            continue

        text_a_parts = []
        for col in text_cols_left:
            value = str(row.get(col, '')).strip()
            if value and value.lower() != 'nan':
                text_a_parts.append(f"{col}: {value}")
        
        text_a = " ".join(text_a_parts)
        
        text_b_parts = []
        for col in text_cols_right:
            value = str(row.get(col, '')).strip()
            if value and value.lower() != 'nan':
                text_b_parts.append(f"{col}: {value}")
                
        text_b = " ".join(text_b_parts)

        combined_text = f"{text_a} || {text_b}"
        
        formatted_data.append({
            "text": combined_text, 
            "label": label
        })
        
    return formatted_data



def infer_left_right_columns_from_csv(csv_path, label_col="label"):
    """
    Liest nur den Header einer CSV-Datei und leitet LEFT_COLS und RIGHT_COLS ab.
    """
    df = pd.read_csv(csv_path, nrows=1)

    left_cols = sorted(
        [c for c in df.columns if c.endswith("_1") and c != label_col]
    )
    right_cols = sorted(
        [c for c in df.columns if c.endswith("_2") and c != label_col]
    )

    if not left_cols or not right_cols:
        raise ValueError(
            f"Keine gültigen _1 / _2 Spalten in {csv_path} gefunden"
        )

    return left_cols, right_cols


try:
    LEFT_COLS, RIGHT_COLS = infer_left_right_columns_from_csv("train.csv")

    print("Automatisch erkannte Spalten:")
    print("LEFT_COLS :", LEFT_COLS)
    print("RIGHT_COLS:", RIGHT_COLS)

    train_data = load_data_from_file(
        "train.csv",
        text_cols_left=LEFT_COLS,
        text_cols_right=RIGHT_COLS
    )

    valid_data = load_data_from_file(
        "valid.csv",
        text_cols_left=LEFT_COLS,
        text_cols_right=RIGHT_COLS
    )

    test_data = load_data_from_file(
        "test.csv",
        text_cols_left=LEFT_COLS,
        text_cols_right=RIGHT_COLS
    )

except FileNotFoundError as e:
    print(f"Kritischer Fehler: Datei nicht gefunden: {e}")
    exit()
except ValueError as e:
    print(f"Kritischer Fehler bei der Spaltenanalyse: {e}")
    exit()

print(
    f"\nDatensatzgrößen:"
    f"\n  Train: {len(train_data)}"
    f"\n  Valid: {len(valid_data)}"
    f"\n  Test : {len(test_data)}"
)

# --- Tokenizer und der Rest des Codes bleiben gleich ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


class EMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, 2) 

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls = outputs.last_hidden_state[:, 0, :] 
        return self.classifier(cls)

    def forward_embeds(self, embeddings, attention_mask, token_type_ids=None):
        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls)

model = EMModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
loss_fn = nn.CrossEntropyLoss()


def encode_batch(batch):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch]).to(DEVICE)
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    return enc, labels

print("\n============================================")
print(f"Starte Training ({NUM_EPOCHS} Epochen)...")
print("============================================")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for i in range(0, len(train_data), BATCH_SIZE):
        batch = train_data[i:i+BATCH_SIZE]
        
        if not batch: 
            continue
            
        enc, labels = encode_batch(batch)
        logits = model(**enc)
        loss = loss_fn(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Loss: {total_loss:.4f}")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n--- Modell nach dem Training gespeichert unter: {MODEL_PATH} ---")


# 5. Captum: Integrated Gradients für Erklärbarkeit

def forward_embeds_wrapper(embeddings, attention_mask, token_type_ids=None):
    return model.forward_embeds(embeddings, attention_mask, token_type_ids)

ig = IntegratedGradients(forward_embeds_wrapper)

def attribute_text(text, target_label=1):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", None)

    embeddings = model.bert.embeddings(input_ids)
    baseline = torch.zeros_like(embeddings)

    attributions = ig.attribute(
        embeddings,
        baselines=baseline,
        target=target_label, 
        additional_forward_args=(attention_mask, token_type_ids),
        n_steps=50 
    )

    token_attrib = attributions.sum(dim=2).squeeze().detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return tokens, token_attrib

def print_token_attributions(tokens, scores):
    print("\nToken-Attributions (positiv = stützt Klasse 1 'Match'):")
    for tok, sc in zip(tokens, scores):
        tok_disp = tok.replace("##", "▂")
        print(f"{tok_disp:20s}  {sc:+.6f}") 



model.eval()
print("\n============================================")
print("Analyse der Erklärbarkeit (Integrated Gradients)")
print("============================================")

for i in range(min(10, len(test_data))):
    text = test_data[i]["text"]
    label = test_data[i]["label"]

    print(f"\n=== Beispiel {i+1} / True Label: {label} ===")
    print("Eingabetext:", text)
    
    tokens, scores = attribute_text(text, target_label=1)
    
    print_token_attributions(tokens, scores)

# ich habe hier vergessen, die accuracy mitzunehmen und das Modell braucht bei mir einfach zu lange
# dadurch kann ich nur schlecht sagen, wie gut das Modell gerade ist...

'''
zu erledigen:
- eig. Programm für Tests mit neuem Modell
- ausgereifte Erklärungen (da evtl. mit chatbot-Einbindung)
- Accuracy evaluieren (da model prediction mit label vergleichen und einen correct-Score mitführen und am ende ausgeben)
- evtl. bessere Visualisierung
'''