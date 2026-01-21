
import sys, logging, re
import os, gc, time, tracemalloc, statistics

import numpy as np
import pandas as pd

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from EMCode.scripts.data_loader import project_root

from lemon import explain

import random



# Seed setzen

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



# hier wird ein fehler aus lemon behoben, welche durch veraltete versionen entsteht
import random as _py_random
_orig_random_init = _py_random.Random.__init__
def _patched_random_init(self, x=None):
    try:
        import numpy as _np
        if isinstance(x, _np.generic):
            x = x.item()
    except Exception:
        pass
    return _orig_random_init(self, x)
_py_random.Random.__init__ = _patched_random_init






# Dataloader

def load_data_from_file(
        df,
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

        formatted_data.append(
            combined_text
        )

    return formatted_data

def infer_left_right_columns_from_df(df: pd.DataFrame):

    left_cols = [c for c in df.columns if c.endswith("_1")]
    right_cols = [c for c in df.columns if c.endswith("_2")]
    return left_cols, right_cols

def format(df):
    LEFT_COLS, RIGHT_COLS = infer_left_right_columns_from_df(df)

    # zwei Tabellen +  IDs
    records_a = df[LEFT_COLS].fillna("").astype(str).copy()
    records_b = df[RIGHT_COLS].fillna("").astype(str).copy()
    records_a.index = [f"a{i}" for i in range(len(records_a))]
    records_b.index = [f"b{i}" for i in range(len(records_b))]

    record_id_pairs = pd.DataFrame(
        {"a.rid": records_a.index.to_list(), "b.rid": records_b.index.to_list()},
        index=[f"p{i}" for i in range(len(df))]
    )
    return records_a,records_b, record_id_pairs


# Datasets Qualitativ

df_q = pd.DataFrame([{ 
    "label": 0,
    "authors_1": "raymond t. ng , divesh srivastava , h. v. jagadish , olga kapitskaia",
    "authors_2": "charu c. aggarwal",  
    "title_1": "charu c. aggarwal,one-dimensional and multi-dimensional substring selectivity estimation",
    "title_2": "hierarchical subspace sampling : a unified framework for high dimensional data reduction , selectivity estimation and nearest neighbor search",  # identisch
    "venue_1": "secure transaction processing in firm real-time database systems",
    "venue_2": "international conference on management of data",  
    "year_1": 2000,
    "year_2": 2002,  
}])

# Dataset Quantitativ

dfs = [
    pd.DataFrame([{
        "label": 0,
        "authors_1": "shaoyu zhou , m. seetha lakshmi",
        "authors_2": "wen-syan li , chris clifton",
        "title_1": "selectivity estimation in extensible databases - a neural network approach",
        "title_2": "semantic integration in heterogeneous databases using neural networks",
        "venue_1": "vldb",
        "venue_2": "very large data bases",
        "year_1": 1998,
        "year_2": 1994,
    }]),
    pd.DataFrame([{
        "label": 0,
        "authors_1": "roberta cochrane , hamid pirahesh , nelson mendonça mattos",
        "authors_2": "daniel barbará - millá , hector garcia-molina",
        "title_1": "integrating triggers and declarative constraints in sql database sytems",
        "title_2": "the demarcation protocol : a technique for maintaining constraints in distributed database systems",
        "venue_1": "vldb",
        "venue_2": "the vldb journal -- the international journal on very large data bases",
        "year_1": 1996,
        "year_2": 1994,
    }]),
    pd.DataFrame([{
        "label": 1,
        "authors_1": "michel tourn , man abrol , john wang , grace zhang , jianchang mao , uma mahadevan , rajat mukherjee , prabhakar raghavan , neil latarche",
        "authors_2": "man abrol , neil latarche , uma mahadevan , jianchang mao , rajat mukherjee , prabhakar raghavan , michel tourn , john wang , grace zhang",
        "title_1": "navigating large-scale semi-structured data in business portals",
        "title_2": "navigating large-scale semi-structured data in business portals",
        "venue_1": "vldb",
        "venue_2": "very large data bases",
        "year_1": 2001,
        "year_2": 2001,
    }]),
    pd.DataFrame([{
        "label": 0,
        "authors_1": "v. s. subrahmanian , eric lemar , k. selçuk candan",
        "authors_2": "thomas seidl , hans-peter kriegel",
        "title_1": "view management in multimedia databases",
        "title_2": "efficient user-adaptable similarity search in large multimedia databases",
        "venue_1": "vldb j.",
        "venue_2": "very large data bases",
        "year_1": 2000,
        "year_2": 1997,
    }]),
    pd.DataFrame([{
        "label": 0,
        "authors_1": "kinji ono , jihad boulos",
        "authors_2": "chun zhang , jeffrey naughton , david dewitt , qiong luo , guy lohman",
        "title_1": "cost estimation of user-defined methods in object-relational database systems",
        "title_2": "on supporting containment queries in relational database management systems",
        "venue_1": "sigmod record",
        "venue_2": "international conference on management of data",
        "year_1": 1999,
        "year_2": 2001,
    }]),
    pd.DataFrame([{
        "label": 1,
        "authors_1": "gerome miklau , jayant madhavan , igor tatarinov , zachary g. ives , xin dong , yana kadiyska , alon y. halevy , nilesh n. dalvi , dan suciu , peter mork",
        "authors_2": "igor tatarinov , zachary ives , jayant madhavan , alon halevy , dan suciu , nilesh dalvi , xin ( luna ) dong , yana kadiyska , gerome miklau , peter mork",
        "title_1": "the piazza peer data management project",
        "title_2": "the piazza peer data management project",
        "venue_1": "sigmod record",
        "venue_2": "acm sigmod record",
        "year_1": 2003,
        "year_2": 2003,
    }]),
    pd.DataFrame([{
        "label": 1,
        "authors_1": "surajit chaudhuri , sanjay agrawal , vivek r. narasayya",
        "authors_2": "sanjay agrawal , surajit chaudhuri , vivek r. narasayya",
        "title_1": "automated selection of materialized views and indexes in sql databases",
        "title_2": "automated selection of materialized views and indexes in sql databases",
        "venue_1": "vldb",
        "venue_2": "very large data bases",
        "year_1": 2000,
        "year_2": 2000,
    }]),
    pd.DataFrame([{
        "label": 0,
        "authors_1": "tobias mayr , thorsten von eicken , michael w. godfrey , praveen seshadri",
        "authors_2": "michael ubell",
        "title_1": "secure and portable database extensibility",
        "title_2": "the montage extensible datablade architecture",
        "venue_1": "sigmod conference",
        "venue_2": "international conference on management of data",
        "year_1": 1998,
        "year_2": 1994,
    }]),
    pd.DataFrame([{
        "label": 0,
        "authors_1": "reagan moore , arun jagatheesan , paul watson , norman w. paton",
        "authors_2": "james r. hamilton",
        "title_1": "grid data management systems & services",
        "title_2": "networked data management design points",
        "venue_1": "vldb",
        "venue_2": "very large data bases",
        "year_1": 2003,
        "year_2": 1999,
    }]),
]



# -----------------------------
# Tokenizer + Model laden 
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device(DEVICE)

MODEL_NAME = "bert-base-uncased"
ROOT = project_root()
MODEL_PATH = os.path.abspath(os.path.join(ROOT, "models/em_bert_model.pt"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


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


model = EMModel().to(device)


# model laden
state = torch.load(MODEL_PATH, map_location="cpu")

'''
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
# robust gegen DataParallel "module."
if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
'''


model.load_state_dict(state, strict=True)
model.eval()


# -----------------------------
# predict_proba für LEMON
# -----------------------------


def infer_left_right_columns_from_csv(pd_f):
    """
    Liest nur den Header einer CSV-Datei und leitet LEFT_COLS und RIGHT_COLS ab.
    """
    left_cols = sorted(
        [c for c in pd_f.columns if c.endswith("_1") and c != 'label']
    )
    right_cols = sorted(
        [c for c in pd_f.columns if c.endswith("_2") and c != 'label']
    )

    return left_cols, right_cols


def predict_proba(records_a: pd.DataFrame,
                  records_b: pd.DataFrame,
                  record_id_pairs: pd.DataFrame,
                  **kwargs) -> np.ndarray:
    
# jedes gelieferte pd pair wird wieder in die ursprüngliche Form gebracht
    a_ids = record_id_pairs["a.rid"].tolist()
    b_ids = record_id_pairs["b.rid"].tolist()

    a_part = records_a.loc[a_ids].copy()
    b_part = records_b.loc[b_ids].copy()

    a_part.index = record_id_pairs.index
    b_part.index = record_id_pairs.index

    base_names = [c[:-2] for c in a_part.columns if c.endswith("_1")]

    # Output in gewünschter Reihenfolge zusammenbauen
    df_pairs = pd.DataFrame(index=record_id_pairs.index)
    df_pairs["label"] = 0

    for base in base_names:
        col_l = f"{base}_1"
        col_r = f"{base}_2"
        if col_l in a_part.columns:
            df_pairs[col_l] = a_part[col_l].astype(str)
        if col_r in b_part.columns:
            df_pairs[col_r] = b_part[col_r].astype(str)

# preprocessing + tokenization

    LEFT_COLS, RIGHT_COLS = infer_left_right_columns_from_csv(df_pairs)
    test_data = load_data_from_file(
        df_pairs,
        text_cols_left=LEFT_COLS,
        text_cols_right=RIGHT_COLS
    )

    enc = tokenizer(test_data, return_tensors="pt", truncation=True, padding=True)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
# Vorhersage berechnen
    with torch.no_grad():
        logits = model(**enc)          # wenn HF-Model; sonst siehe Fall D
        logits = logits.logits if hasattr(logits, "logits") else logits

        probs = torch.softmax(logits, dim=1)[:,1]
    probs = probs.detach().cpu().numpy().astype(float)
    return probs

# -----------------------------
# 9 verschidene Beispielpaare
# -----------------------------

for df in dfs:
    records_a,records_b,record_id_pairs= format(df)
    res=explain(
        records_a=records_a,
        records_b=records_b,
        record_id_pairs=record_id_pairs,
        predict_proba=predict_proba,
        num_features=4,
        num_samples=100,
        return_dict=True,
        estimate_potential=True,
        granularity="attributes",
        show_progress=False,
        )
    res = res["p0"]
    print("\n=== LEMON Erklärung ===") 
    print("prediction_score:", res.prediction_score) 
    print("\nrecord_pair:\n", res.record_pair)
    print("\nAttributions:") 
    for a in res.attributions: 
        print( f" weight={a.weight:+.6f} | potential={a.potential} | positions={a.positions} | name={a.name}" )

# -----------------------------
# Eine Aufwendigere Erklärng
# -----------------------------

records_a,records_b,record_id_pairs= format(df_q)
res=explain(
    records_a=records_a,
    records_b=records_b,
    record_id_pairs=record_id_pairs,
    predict_proba=predict_proba,
    num_features=4,
    return_dict=True,
    num_samples=200,
    estimate_potential=True,
    granularity="counterfactual",
    show_progress=False,
    )
res = res["p0"]
print("\n=== LEMON Erklärung ===") 
print("prediction_score:", res.prediction_score) 
print("\nrecord_pair:\n", res.record_pair)
print("\nAttributions:") 
for a in res.attributions: 
    print( f" weight={a.weight:+.6f} | potential={a.potential} | positions={a.positions} | name={a.name}" )


# Ausgeben der Feature welche gefunden wurden

def _iter_token_spans(text: str, token_patterns="[^ ]+"):
    patterns = [token_patterns] if isinstance(token_patterns, str) else list(token_patterns)
    pattern = "|".join(f"(?:{p})" for p in patterns)
    for m in re.finditer(pattern, text):
        yield (m.start(), m.end(), text[m.start():m.end()])

def print_feature_token_breakdown(exp, token_patterns="[^ ]+"):
    for i, a in enumerate(exp.attributions, start=1):
        print(f"\nFeature #{i}")
        for (source, attr, attr_or_val, j) in a.positions:
            rep = exp.string_representation.get((source, attr, attr_or_val))

            if rep is None or j is None or not hasattr(rep, "spans"):
                txt = "" if rep is None else str(rep)
                print(f"  - {source}.{attr}.{attr_or_val}: {txt!r}")
                continue
            token = rep[j]  # Token-String
            start = int(rep.spans[2 * j + 1])
            end = int(rep.spans[2 * j + 2])

            # Optional: “Untertokens” innerhalb dieses (evtl. zusammengefassten) Tokens anzeigen
            raw_text = rep.string
            subtoks = [t for (s0, e0, t) in _iter_token_spans(raw_text, token_patterns) if e0 > start and s0 < end]

            print(
                f"  - {source}.{attr}.{attr_or_val}[{j}] -> {token!r} "
                f"(chars {start}:{end}) | subtokens={subtoks}"
            )

print_feature_token_breakdown(res, token_patterns="[^ ]+")




