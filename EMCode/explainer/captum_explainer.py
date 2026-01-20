import torch
from transformers import AutoTokenizer
from captum.attr import IntegratedGradients
import os
import matplotlib.pyplot as plt
import numpy as np

from EMCode.scripts.data_loader import load_data_from_file, infer_left_right_columns_from_csv, project_root
from EMCode.model.em_bert_model import EMModel

# Pfade relativ zur aktuellen Root
ROOT = project_root()
MODEL_PATH = os.path.abspath(os.path.join(ROOT, "models/em_bert_model.pt"))
DATASET_PATH = os.path.abspath(os.path.join(ROOT, "datasets/"))

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lade Tokenizer und Modell
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = EMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Lade Testdaten
LEFT_COLS, RIGHT_COLS = infer_left_right_columns_from_csv(os.path.join(DATASET_PATH, "test_20.csv"))
test_data = load_data_from_file(
    os.path.join(DATASET_PATH, "test_20.csv"),
    text_cols_left=LEFT_COLS,
    text_cols_right=RIGHT_COLS
)


def plot_diverging_attributes(attr_scores, structural_share, title="Attributions-Analyse"):
    """
    Erstellt ein Balkendiagramm mit Anzeige des Structural Bias.
    """
    sorted_items = sorted(attr_scores.items(), key=lambda x: abs(x[1]), reverse=False)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color=colors, edgecolor='black', alpha=0.8)
    ax.axvline(0, color='black', linewidth=1.5)

    # Styling
    ax.set_xlabel('Attributions-Score (Wichtigkeit)')
    ax.set_title(title, fontsize=14, pad=25)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Structural Bias Textbox oben rechts
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    bias_text = f"Structural Bias: {structural_share:.1%}"
    ax.text(0.95, 1.05, bias_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Werte an die Balken schreiben
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + (0.05 if width > 0 else -0.05)
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2,
                f'{width:+.3f}', va='center',
                ha='left' if width > 0 else 'right', fontsize=9)

    plt.tight_layout()
    plt.show()


def forward_embeds_wrapper(embeddings, attention_mask, token_type_ids=None):
    return model.forward_embeds(embeddings, attention_mask, token_type_ids)


ig = IntegratedGradients(forward_embeds_wrapper)


def attribute_text(text, target_label=1, baseline_type="pad"):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", None)

    with torch.no_grad():
        output = model(input_ids, attention_mask, token_type_ids)
        prediction = torch.argmax(output, dim=1).item()

    input_embeddings = model.bert.embeddings(input_ids)

    # --- Baseline Auswahl ---
    if baseline_type == "zeros":
        baseline = torch.zeros_like(input_embeddings)

    elif baseline_type == "pad":
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        baseline = model.bert.embeddings(baseline_ids)

    elif baseline_type == "mask":
        baseline_ids = torch.full_like(input_ids, tokenizer.mask_token_id)
        baseline = model.bert.embeddings(baseline_ids)

    elif baseline_type == "random":
        baseline = torch.randn_like(input_embeddings) * 0.02

    else:
        baseline = torch.zeros_like(input_embeddings)

    attributions, delta = ig.attribute(
        input_embeddings,
        baselines=baseline,
        target=target_label,
        additional_forward_args=(attention_mask, token_type_ids),
        n_steps=100,  # 50-200 ist sehr teuer für cpu
        return_convergence_delta=True
    )

    token_attrib = attributions.sum(dim=2).squeeze().detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    total_abs_score = np.sum(np.abs(token_attrib))

    # Indices der Sonder-Tokens finden
    special_token_ids = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]

    clean_tokens = []
    clean_scores = []
    special_scores_abs_sum = 0

    for i, (tok, score) in enumerate(zip(tokens, token_attrib)):
        if input_ids[0][i].item() in special_token_ids:
            special_scores_abs_sum += abs(score)
        else:
            clean_tokens.append(tok)
            clean_scores.append(score)

    # "Verlust" durch die SEP-Tokens dokumentieren
    # print("\n====================================================")
    # print(f"Info: {special_scores_abs_sum:.4f} Attribution auf Struktur-Tokens entfallen.")

    structural_share = (special_scores_abs_sum / total_abs_score) if total_abs_score > 0 else 0
    delta_val = delta.detach().cpu().item()

    return clean_tokens, clean_scores, prediction, delta_val, structural_share


def evaluate_baseline_quality(test_data, baseline_types=["zeros", "pad", "mask"]):
    """
    Berechnet statistische Metriken für verschiedene Baselines.
    """
    print("\n" + "=" * 70)
    print(f"{'Baseline Evaluation':^70}")
    print("=" * 70)

    summary = {}

    for b_type in baseline_types:
        deltas, shares = [], []
        print(f"Evaluiere {b_type}...", end="\r")

        for item in test_data:
            _, _, _, d, s = attribute_text(item["text"], baseline_type=b_type)
            deltas.append(abs(d))
            shares.append(s)

        summary[b_type] = {
            "avg_delta": np.mean(deltas),
            "std_delta": np.std(deltas),
            "avg_struct": np.mean(shares)
        }

    # Tabelle drucken
    print(f"{'Baseline':<12} | {'Avg Delta':<12} | {'Std Dev':<10} | {'Struct. Bias %':<15}")
    print("-" * 70)
    for b_type, m in summary.items():
        print(f"{b_type:<12} | {m['avg_delta']:<12.6f} | {m['std_delta']:<10.4f} | {m['avg_struct']:<15.2%}")
    print("-" * 70)

    return summary


def print_token_attributions(tokens, scores):
    print("\nToken-Attributions (sortiert nach Relevanz):")

    pairs = list(zip(tokens, scores))

    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Token':20s} | {'Score':10s}")
    print("-" * 35)

    for tok, sc in pairs_sorted:
        tok_disp = tok.replace("##", "▂")
        print(f"{tok_disp:20s} | {sc:+.6f}")


def aggregate_for_data(tokens, scores, left_cols, right_cols):
    """
    Nutzt die Spaltenlisten aus deinem DataLoader, um die Scores zuzuordnen.
    """
    try:
        split_idx = [i for i, t in enumerate(tokens) if t == "|"][0]
    except IndexError:
        split_idx = len(tokens)  # Fallback

    attr_scores = {}
    current_field = "Präambel"

    clean_left = [c.replace("_1", "") for c in left_cols]
    clean_right = [c.replace("_2", "") for c in right_cols]

    for i, (tok, sc) in enumerate(zip(tokens, scores)):
        t_clean = tok.replace("▂", "").lower()

        side = "_1" if i < split_idx else "_2"
        active_columns = clean_left if i < split_idx else clean_right

        if t_clean in active_columns:
            current_field = f"{t_clean}{side}"
        elif t_clean == "|":
            current_field = "Trenner"

        attr_scores[current_field] = attr_scores.get(current_field, 0) + sc

    return attr_scores


def print_attribute_summary(attr_scores):
    print(f"\n{'Attribut':<25} | {'Gesamt-Attribution':<20}")
    print("-" * 50)
    sorted_attrs = sorted(attr_scores.items(), key=lambda x: abs(x[1]), reverse=True)

    for attr, score in sorted_attrs:
        print(f"{attr:<25} | {score:+.6f}")


model.eval()
print("\n============================================")
print("Analyse der Erklärbarkeit (Integrated Gradients)")
print("============================================")

# --- Statistischer Vergleich der Baselines ---
baseline_stats = evaluate_baseline_quality(test_data)
best_baseline = min(baseline_stats, key=lambda k: baseline_stats[k]["avg_delta"])
print(f"Empfohlene Baseline für Analyse: {best_baseline.upper()}")

correct_predictions = 0
total_examples = min(10, len(test_data))

for i in range(min(10, len(test_data))):
    text = test_data[i]["text"]
    label = test_data[i]["label"]

    # tokens, scores, pred, delta_val = attribute_text(text, target_label=1, baseline_type="mask")
    tokens, scores, pred, delta_val, struct_share = attribute_text(text, target_label=1, baseline_type=best_baseline)

    if pred == label:
        correct_predictions += 1

    print(f"\n=== Beispiel {i + 1} | True: {label} | Pred: {pred} ===")
    print(f"Convergence Delta: {abs(delta_val):.6f} (niedriger ist besser)")

    print_token_attributions(tokens, scores)

    attr_scores = aggregate_for_data(tokens, scores, LEFT_COLS, RIGHT_COLS)
    print_attribute_summary(attr_scores)
    plot_diverging_attributes(
        attr_scores,
        structural_share=struct_share,
        title=f"Beispiel {i + 1}: Influence per Attribute\n(Baseline: {best_baseline})"
    )

