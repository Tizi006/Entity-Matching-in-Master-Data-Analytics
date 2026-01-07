import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients

from model.bert_model_train import test_data

MODEL_NAME = "em_bert_model"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lade Tokenizer und Modell
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
# 5. Captum: Integrated Gradients für Erklärbarkeit

def forward_embeds_wrapper(embeddings, attention_mask, token_type_ids=None):
    return model.forward_embeds(embeddings, attention_mask, token_type_ids)


ig = IntegratedGradients(forward_embeds_wrapper)  # evtl. lieber GradientShap, Saliency etc.


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
        n_steps=50  # bei CPU eher 10. 50 ist da wohl zu viel
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

    print(f"\n=== Beispiel {i + 1} / True Label: {label} ===")
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