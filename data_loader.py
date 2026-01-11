import json
import random

def load_sarcasm_dataset(file_path, batch_size=10, interactive_size=20):
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                records.append({
                    "sentence": obj["headline"].strip(),
                    "label": "Sarcastic" if obj["is_sarcastic"] == 1 else "Not Sarcastic"
                })
            except Exception:
                continue

    if not records:
        raise ValueError("Dataset is empty or could not be parsed.")

    random.shuffle(records)

    interactive_records = records[:interactive_size]
    batch_records = records[:batch_size]

    return batch_records, interactive_records

