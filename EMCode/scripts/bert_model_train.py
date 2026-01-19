# !pip install -q transformers captum pandas
# !mkdir -p models 

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import os
import sys
import io

from data_loader import (
    infer_left_right_columns_from_csv,
    load_data_from_file,
    get_dataloader, project_root
)

from EMCode.model.em_bert_model import EMModel

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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                              encoding='utf-8')  # evtl lieber nicht buffern, da man sonst erst am ende was sieht...

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pfade relativ zur Root
ROOT = project_root()
MODEL_PATH = os.path.abspath(os.path.join(ROOT, "models/em_bert_model.pt"))
DATASET_PATH = os.path.abspath(os.path.join(ROOT, "datasets/"))

BATCH_SIZE = 16  # bei cpu-training weniger (4 oder 8)
NUM_EPOCHS = 3  # mehr wäre eig nice, aber des dauert bei mir viel zu lange. Man könnte die Dateigröße reduzieren...

print("Device:", DEVICE)
print(f"Modell wird gespeichert unter: {MODEL_PATH}")

try:
    LEFT_COLS, RIGHT_COLS = infer_left_right_columns_from_csv(os.path.join(DATASET_PATH, "train.csv"))

    print("Automatisch erkannte Spalten:")
    print("LEFT_COLS :", LEFT_COLS)
    print("RIGHT_COLS:", RIGHT_COLS)

    train_data = load_data_from_file(
        os.path.join(DATASET_PATH, "train.csv"),
        text_cols_left=LEFT_COLS,
        text_cols_right=RIGHT_COLS
    )

    valid_data = load_data_from_file(
        os.path.join(DATASET_PATH, "valid.csv"),
        text_cols_left=LEFT_COLS,
        text_cols_right=RIGHT_COLS
    )

    test_data = load_data_from_file(
        os.path.join(DATASET_PATH, "test.csv"),
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

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Erzeuge DataLoader
device = torch.device(DEVICE)
train_loader = get_dataloader(train_data, tokenizer, batch_size=BATCH_SIZE, max_length=128, shuffle=True, device=device)
valid_loader = get_dataloader(valid_data, tokenizer, batch_size=BATCH_SIZE, max_length=128, shuffle=False,
                              device=device)
test_loader = get_dataloader(test_data, tokenizer, batch_size=BATCH_SIZE, max_length=128, shuffle=False, device=device)

model = EMModel().to(DEVICE)
# for param in EMCode.bert.parameters(): param.requires_grad = False
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
loss_fn = nn.CrossEntropyLoss()

print("\n============================================")
print(f"Starte Training ({NUM_EPOCHS} Epochen)...")
print("============================================")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for enc, labels in train_loader:
        logits = model(**enc)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} Loss: {total_loss:.4f}")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n--- Modell nach dem Training gespeichert unter: {MODEL_PATH} ---")
