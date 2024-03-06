import pathlib

import gdown

raw_output_data_folder = pathlib.Path("data/raw/sf")
raw_output_data_folder.mkdir(parents=True, exist_ok=True)

dataset_ids = {"full_sf.csv": "1OwJ8tU9WTLIS6POPFOt9j028zBTXOxvW", "small_sf.npz": "1jX9AQz5orzy5ShV8uo2pZwOh85VnJQIe"}
for name, dataset_id in dataset_ids.items():
    output_file = str(raw_output_data_folder) + "/" + name
    gdown.download(id=dataset_id, output=output_file)
