from torch import nn
from transformers import AutoModel

"""
das ist das große Modell, das für EM hergenommen wird.
"""


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