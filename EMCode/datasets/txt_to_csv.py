import re
import csv
from pathlib import Path


def parse_line(line):
    label_match = re.search(r'\s(\d)\s*$', line)
    if not label_match:
        return None

    label = int(label_match.group(1))
    line = line[:label_match.start()].strip()

    pattern = re.compile(r'COL\s+(.*?)\s+VAL\s+(.*?)(?=\s+COL|\s*$)')
    matches = pattern.findall(line)

    half = len(matches) // 2
    ent1 = matches[:half]
    ent2 = matches[half:]

    record = {}

    for key, val in ent1:
        record[f"{key.strip()}_1"] = val.strip()

    for key, val in ent2:
        record[f"{key.strip()}_2"] = val.strip()

    record["label"] = label
    return record


def convert_txt_to_csv(input_txt, output_csv):
    records = []

    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = parse_line(line)
            if record:
                records.append(record)

    if not records:
        raise ValueError(f"Keine gültigen Daten in {input_txt}")

    fieldnames = sorted(
        {k for r in records for k in r.keys()},
        key=lambda x: (x != "label", x)
    )

    with open(output_csv, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def batch_convert(base_dir="."):
    base_dir = Path(base_dir)

    for split in ["train", "valid", "test"]:
        input_txt = base_dir / f"{split}.txt"
        output_csv = base_dir / f"{split}.csv"

        if not input_txt.exists():
            print(f"[WARN] Übersprungen (nicht gefunden): {input_txt}")
            continue

        print(f"[INFO] Konvertiere {input_txt} -> {output_csv}")
        convert_txt_to_csv(input_txt, output_csv)

    print("\n[FERTIG] Batch-Konvertierung abgeschlossen.")


if __name__ == "__main__":
    batch_convert(".")
