import io
import pathlib
import zipfile

import gdown
import requests

output_shapefile_path = pathlib.Path("data/raw/sf/shapefile")
output_shapefile_path.mkdir(parents=True, exist_ok=True)

shapefile_url = {"tl_2017_06075_roads": "https://www2.census.gov/geo/tiger/TIGER2017//ROADS/tl_2017_06075_roads.zip"}

for name, url in shapefile_url.items():
    r = requests.get(url, stream=True)

    is_zip = zipfile.is_zipfile(io.BytesIO(r.content))

    if is_zip:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(output_shapefile_path / name)

# TODO: Deal with non zip files


output_csv_path = pathlib.Path("data/raw/sf/csv")
output_csv_path.mkdir(parents=True, exist_ok=True)

dataset_ids = {
    "full_sf.csv": "1OwJ8tU9WTLIS6POPFOt9j028zBTXOxvW",
    "small_sf.npz": "1jX9AQz5orzy5ShV8uo2pZwOh85VnJQIe",
}

for name, dataset_id in dataset_ids.items():
    output_file = str(output_csv_path) + "/" + name
    gdown.download(id=dataset_id, output=output_file)
