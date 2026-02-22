import os
import json
import random

def create_metadata_jsonl(dataset_dir="dataset"):
    pos_dir = os.path.join(dataset_dir, "positive")
    neg_dir = os.path.join(dataset_dir, "negative")
    output_file = os.path.join(dataset_dir, "metadata.jsonl")

    records = []

    # Process positive (standard clean fish)
    if os.path.exists(pos_dir):
        for filename in os.listdir(pos_dir):
            if filename.lower().endswith('.jpg'):
                # We store the relative path for HuggingFace (e.g., positive/img.jpg)
                rel_path = f"positive/{filename}"
                records.append({
                    "file_name": rel_path,
                    "text": "This is a standard clean fish fillet"
                })

    # Process negative (defects)
    if os.path.exists(neg_dir):
        for filename in os.listdir(neg_dir):
            if filename.lower().endswith('.jpg'):
                rel_path = f"negative/{filename}"
                records.append({
                    "file_name": rel_path,
                    "text": "This fish fillet has a red blood spot defect"
                })

    # Shuffle the records randomly to ensure even distribution during training
    random.shuffle(records)

    # Write to metadata.jsonl
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Successfully generated {output_file} with {len(records)} shuffled records.")

if __name__ == "__main__":
    create_metadata_jsonl()
