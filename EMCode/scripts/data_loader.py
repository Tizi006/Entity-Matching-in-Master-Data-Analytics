import pandas as pd
import json
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import DataLoader
from pathlib import Path


def project_root():
    """
    Projekt-Root relativ zu diesem Skript.
    Fallback: aktuelles Arbeitsverzeichnis (z.B. REPL).
    """
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd()


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


def encode_batch(batch: List[Dict], tokenizer, device: Optional[torch.device] = None, max_length: Optional[int] = None):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = labels.to(device)
    return enc, labels


def collate_fn_factory(tokenizer, max_length: Optional[int] = None, device: Optional[torch.device] = None):
    def collate(batch):
        return encode_batch(batch, tokenizer, device=device, max_length=max_length)

    return collate


def get_dataloader(
        formatted_data: List[Dict],
        tokenizer,
        batch_size: int = 16,
        max_length: Optional[int] = None,
        shuffle: bool = False,
        device: Optional[torch.device] = None
) -> DataLoader:
    collate = collate_fn_factory(tokenizer, max_length=max_length, device=device)
    return DataLoader(formatted_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
