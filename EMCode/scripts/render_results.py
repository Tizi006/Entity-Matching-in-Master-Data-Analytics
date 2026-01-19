import pandas as pd
from textattack.loggers import CSVLogger
from textattack.attack_results import SuccessfulAttackResult
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt


# Html comparison
def attack_results_to_html(attack_results, output_path="attack_results.html"):
    """
    Takes TextAttack attack_results and writes a standalone HTML file
    showing original and perturbed text side-by-side with highlights.

    Args:
        attack_results: list of AttackResult (from attacker.attack_dataset())
        output_path: output HTML file path
    """

    # Collect successful attacks with HTML highlighting
    logger = CSVLogger(color_method="html")

    for result in attack_results:
        if isinstance(result, SuccessfulAttackResult):
            logger.log_attack_result(result)

    if not logger.row_list:
        raise ValueError("No successful attack results to render.")

    df = pd.DataFrame.from_records(logger.row_list)

    # Build custom HTML
    rows_html = []
    for i, row in df.iterrows():
        rows_html.append(f"""
        <tr>
            <td class="cell">
                <b>Original</b><br>
                {row["original_text"]}
            </td>
            <td class="cell">
                <b>Perturbed</b><br>
                {row["perturbed_text"]}
            </td>
        </tr>
        """)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>TextAttack Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th {{
                background-color: #f2f2f2;
                padding: 10px;
                border: 1px solid #ccc;
            }}
            td.cell {{
                vertical-align: top;
                padding: 12px;
                border: 1px solid #ccc;
                width: 50%;
                word-wrap: break-word;
            }}
            tr:nth-child(even) {{
                background-color: #fafafa;
            }}
        </style>
    </head>
    <body>
        <h2>TextAttack Adversarial Examples</h2>
        <table>
            <thead>
                <tr>
                    <th>Original Text</th>
                    <th>Perturbed Text</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </body>
    </html>
    """

    # Write file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML file written to: {output_path}")
    logger.flush()


# Heat map render

def get_changed_tokens_from_html(html_text):
    """
    Returns a set of tokens highlighted in red (TextAttack HTML),
    works with <span> or <font color="red">.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    changed_tokens = []

    # Find all <span> or <font color="red"> elements
    for elem in soup.find_all(['span', 'font']):
        # Only keep if color is red
        color = elem.get('color')
        if color and color.lower() == 'green':
            text_tokens = elem.get_text().split()
            changed_tokens.extend(text_tokens)

    return set(changed_tokens)


def extract_fields(text, fields):
    """
    Extract field contents from structured text.
    Returns dict: {field: content_string}
    """
    result = {}
    for i, field in enumerate(fields):
        start = text.find(f"{field}:")
        if start == -1:
            result[field] = ""
            continue
        if i + 1 < len(fields):
            next_field = text.find(f"{fields[i + 1]}:", start)
            end = next_field if next_field != -1 else len(text)
        else:
            end = len(text)
        result[field] = text[start + len(f"{field}:"):end].strip()
    return result


def field_token_mask_from_html(original_text, perturbed_html, fields):
    """
    Returns:
        field_tokens: dict field -> list of tokens
        field_mask: dict field -> list of 0/1 if token was changed
    """
    changed_tokens = get_changed_tokens_from_html(perturbed_html)
    field_contents = extract_fields(original_text, fields)

    field_tokens = {}
    field_mask = {}

    for field, content in field_contents.items():
        clean_content = content.replace(',', '')
        tokens = clean_content.split()
        field_tokens[field] = tokens
        mask = [1 if t in changed_tokens else 0 for t in tokens]
        field_mask[field] = mask

    return field_tokens, field_mask


def aggregate_field_frequencies(attack_results, fields):
    """
    Returns dict: field -> np.array of token-level change frequencies
    """
    logger = CSVLogger(color_method="html")
    for result in attack_results:
        if isinstance(result, SuccessfulAttackResult):
            logger.log_attack_result(result)
            # Add the true original text from AttackResult
            logger.row_list[-1]['true_original_text'] = result.original_result.attacked_text.text

    df = pd.DataFrame.from_records(logger.row_list)

    field_freq = {f: [] for f in fields}

    for _, row in df.iterrows():
        _, masks = field_token_mask_from_html(row["true_original_text"], row["original_text"], fields)
        for f in fields:
            field_freq[f].append(masks[f])

    # Compute mean frequency per token in each field
    field_freq_mean = {}
    for f in fields:
        max_len = max(len(lst) for lst in field_freq[f])
        padded = [lst + [0] * (max_len - len(lst)) for lst in field_freq[f]]
        field_freq_mean[f] = np.mean(padded, axis=0)

    logger.flush()

    return field_freq_mean


def plot_field_heatmap(field_freq_mean, output_path="field_heatmap.png"):
    fields = list(field_freq_mean.keys())
    max_len = max(len(arr) for arr in field_freq_mean.values())
    matrix = np.zeros((len(fields), max_len))
    for i, f in enumerate(fields):
        arr = field_freq_mean[f]
        matrix[i, :len(arr)] = arr

    plt.figure(figsize=(12, 2 + len(fields)))
    plt.imshow(matrix, aspect="auto", cmap="Reds")
    plt.colorbar(label="Probability of Change")
    plt.yticks(range(len(fields)), fields)
    plt.xlabel("Token Position in Field")
    plt.title("Field-level Attack Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved field-level heatmap: {output_path}")


def aggregate_field_frequencies_1d(attack_results, fields):
    """
    Returns dict: field -> mean token-level change probability
    """
    logger = CSVLogger(color_method="html")
    for result in attack_results:
        if isinstance(result, SuccessfulAttackResult):
            logger.log_attack_result(result)
            # Add the true original text from AttackResult
            logger.row_list[-1]['true_original_text'] = result.original_result.attacked_text.text

    df = pd.DataFrame.from_records(logger.row_list)

    field_masks = {f: [] for f in fields}

    for _, row in df.iterrows():
        original_text = row["true_original_text"]
        perturbed_html = row["original_text"]

        changed_tokens = get_changed_tokens_from_html(perturbed_html)
        field_contents = extract_fields(original_text, fields)

        for f, content in field_contents.items():
            # remove punctuation, split tokens
            tokens = [t for t in content.replace(",", "").split()]
            mask = [1 if t in changed_tokens else 0 for t in tokens]
            field_masks[f].append(mask)

    # Compute mean probability per field
    field_scores = {}
    for f in fields:
        ratios = [np.sum(m) / len(m) for m in field_masks[f] if len(m) > 0]
        field_scores[f] = float(np.mean(ratios)) if ratios else 0.0
    logger.flush()

    return field_scores


def plot_field_heatmap_1d(field_scores, output_path="field_heatmap_1d.png"):
    fields = list(field_scores.keys())
    values = np.array([field_scores[f] for f in fields]).reshape(-1, 1)

    plt.figure(figsize=(4, 2 + len(fields)*0.6))
    plt.imshow(values, cmap="Reds", aspect="auto")
    plt.colorbar(label="Probability of Change")
    plt.yticks(range(len(fields)), fields)
    plt.xticks([])
    plt.title("Field-level Attack Sensitivity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved 1D field heatmap: {output_path}")
