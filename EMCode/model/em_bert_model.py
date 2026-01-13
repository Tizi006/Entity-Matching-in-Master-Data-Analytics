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

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, embeddings=None):
        """
        Gibt das CLS-Embedding (batch, embed_dim) zurück.
        Akzeptiert entweder bereits berechnete `embeddings` oder `input_ids`.
        """
        if embeddings is None:
            if input_ids is None:
                raise ValueError("Provide `input_ids` or `embeddings` to encode().")
            embeddings = self.embeddings(input_ids)
        return embeddings[:, 0, :]
