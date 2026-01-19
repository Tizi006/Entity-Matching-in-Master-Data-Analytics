import os

import textattack
from transformers import AutoTokenizer
from textattack.attack_recipes import TextFoolerJin2019
from textattack import Attacker
from textattack.datasets import Dataset
from textattack.models.wrappers import ModelWrapper
from textattack.constraints.overlap import MaxWordsPerturbed
import gc

from EMCode.model.em_bert_model import EMModel
from EMCode.scripts.data_loader import *
from EMCode.scripts.render_results import (attack_results_to_html, aggregate_field_frequencies, plot_field_heatmap,
                                           plot_field_heatmap_1d, aggregate_field_frequencies_1d)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = project_root()
MODEL_PATH = os.path.abspath(os.path.join(ROOT, "models/em_bert_model.pt"))
DATASET_PATH = os.path.abspath(os.path.join(ROOT, "datasets/test.csv"))
FIELDS = ["authors_2", "title_2", "venue_2", "year_2"]


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

        return logits.detach().cpu().numpy()


def run_span_attack(
        model_wrapper,
        dataset
):
    """
    Runs a TextFooler attack on a span-based wrapper.

    Args:
        dataset: data to attack
        model_wrapper: your EM model stack wrapped data
    """

    attack = TextFoolerJin2019.build(model_wrapper)

    # Optimisations
    # Only 10% of tokens changed to cause a flip
    max_words = MaxWordsPerturbed(max_percent=0.15)
    attack.constraints.append(max_words)

    attack_args = textattack.AttackArgs(
        num_examples=len(dataset),
        disable_stdout=False,
    )

    attacker = Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()

    # Memory cleanup
    del attacker
    del attack
    torch.cuda.empty_cache()
    gc.collect()

    for res in results:
        print(res)

    return results


def main():
    device = torch.device(DEVICE)
    tokenizer, model = load_model(MODEL_PATH)

    test_data = load_filtered_data(DATASET_PATH, 1)
    all_attack_results = []

    for i in range(200):
        label = test_data[i]["label"]

        sides = test_data[i]["text"].split(" || ")
        left_fixed = sides[0]
        right_orig = sides[1]

        template = f"{left_fixed} || {{}}"

        dataset = Dataset([(right_orig, label)])

        model_wrapper = SpanPerturbationWrapper(
            model=model,
            tokenizer=tokenizer,
            template=template,
            device=device,
        )

        attack_results = run_span_attack(model_wrapper, dataset)
        all_attack_results.extend(attack_results)

    attack_results_to_html(all_attack_results)

    field_freq_mean = aggregate_field_frequencies(all_attack_results, FIELDS)
    plot_field_heatmap(field_freq_mean)

    field_scores = aggregate_field_frequencies_1d(all_attack_results, FIELDS)
    plot_field_heatmap_1d(field_scores)


if __name__ == "__main__":
    main()
