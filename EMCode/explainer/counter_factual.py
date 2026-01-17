import os

import textattack
import torch
from transformers import AutoTokenizer

from textattack.attack_recipes import TextFoolerJin2019
from textattack import Attacker
from textattack.datasets import Dataset
from textattack.models.wrappers import ModelWrapper
from textattack.constraints.overlap import MaxWordsPerturbed
#from textattack.constraints.pre_transformation import WordRegexSubstitution

from EMCode.model.em_bert_model import EMModel
from EMCode.scripts.data_loader import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = project_root()
MODEL_PATH = os.path.abspath(os.path.join(ROOT, "models/em_bert_model.pt"))
DATASET_PATH = os.path.abspath(os.path.join(ROOT, "datasets/test.csv"))


def load_model(model_path):
    """
    Loads tokenizer and trained EM EMCode.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = EMModel().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return tokenizer, model


class SpanPerturbationWrapper(ModelWrapper):
    """
    Allows TextAttack to perturb ONLY a specific span of the input text.
    """

    def __init__(self, model, tokenizer, template, device):
        """
        template: string containing exactly one '{}' placeholder
                  Example:
                  "title_2: ... authors_2: {} year_2: 2001"
        """
        assert "{}" in template, "Template must contain exactly one '{}' placeholder"

        self.model = model
        self.tokenizer = tokenizer
        self.template = template
        self.device = device

    def __call__(self, text_list, **kwargs):
        """
        text_list: List[str] of ONLY the mutable span
        Returns: np.ndarray (batch_size, num_classes)
        """
        full_texts = [self.template.format(span) for span in text_list]

        enc = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        return logits.cpu().numpy()


def main():
    device = torch.device(DEVICE)
    tokenizer, model = load_model(MODEL_PATH)

    test_data = load_filtered_data(DATASET_PATH, 1)

    sides = test_data[10]["text"].split(" || ")
    left_fixed = sides[0]
    right_orig = sides[1]
    label = test_data[10]["label"]

    template = f"{left_fixed} || {{}}"

    dataset = Dataset(
        [(right_orig, label)]
    )

    model_wrapper = SpanPerturbationWrapper(
        model=model,
        tokenizer=tokenizer,
        template=template,
        device=device
    )

    attack = TextFoolerJin2019.build(model_wrapper)

    # Optimisations
    # Only 10% of tokens changed to cause a flip
    max_words = MaxWordsPerturbed(max_percent=0.10)
    attack.constraints.append(max_words)

    # Blocking out Years
    #attack.constraints.append(
    #    WordRegexSubstitution(r"\b\d{4}\b")
    #)

    # test only filed aware (only one filed to flip)
    # right_title = extract_field(right_orig, "title_2")
    # right_venue = extract_field(right_orig, "venue_2")

    attack_args = textattack.AttackArgs(num_examples=1, disable_stdout=False, log_to_csv="counter_factual_results.csv")
    attacker = Attacker(attack, dataset, attack_args)

    results = attacker.attack_dataset()
    for res in results:
        print(res)




if __name__ == "__main__":
    main()
