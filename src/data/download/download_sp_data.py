import pathlib

import gdown

output_csv_path = pathlib.Path("data/raw/sp")
output_csv_path.mkdir(parents=True, exist_ok=True)

dataset_ids = {
    "sp_road_net": "1tljSNcBGoLhf5upPZDP44dUcDU_1rMBs",
}

for name, dataset_id in dataset_ids.items():
    output_file = str(output_csv_path) + "/" + name
    gdown.download_folder(id=dataset_id, output=output_file)
