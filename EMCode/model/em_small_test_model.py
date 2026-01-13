from torch import nn

"""
`em_small_test_model.py`
Kleines Debug-/Test-Modell — NICHT für Produktion. 
Zweck: schnelle lokale Tests, Unit-Tests und Entwicklung ohne schweren Pretrained-Backbone.
"""


class EMModel(nn.Module):
    def __init__(self, vocab_size: int = 30522, embed_dim: int = 64):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        embeds = self.embeddings(input_ids)  # (batch, seq_len, embed_dim)
        # einfache CLS-Auswahl wie im Original (Token 0)
        cls = embeds[:, 0, :]
        return self.classifier(cls)

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, embeddings=None):
        if embeddings is None:
            if input_ids is None:
                raise ValueError("Provide `input_ids` or `embeddings` to encode().")
            embeddings = self.embeddings(input_ids)
        return embeddings[:, 0, :]
