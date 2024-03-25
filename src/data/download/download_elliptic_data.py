import pathlib

import gdown

output_path = pathlib.Path("data/raw/elliptic")
output_path.mkdir(parents=True, exist_ok=True)

files_ids = {
    "elliptic_txs_classes_original": "1C3r52ACXLYUGMjRde3dR1emN33HDDfyR",
    "elliptic_txs_edgelist_original": "1Byp-lWrR4ShL9UBLFhCdL9sQUu3O5GDb",
    "elliptic_txs_features_original": "1mOVwwsjwqkJnnyW6VMQFD9C0BM9B80yP",
}

for name, dataset_id in files_ids.items():
    output_file = str(output_path) + "/" + name
    gdown.download(id=dataset_id, output=output_file)
