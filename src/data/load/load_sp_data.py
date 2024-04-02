import pathlib

import h5py

raw_data_path = pathlib.Path("data/raw/sp/processed_data")

# Read h5 crimes data
with h5py.File(raw_data_path / "crimes.h5", "r") as f:
    crime_data = {k: None for k in f["key"].keys()}

    for k in crime_data.keys():
        crime_data[k] = f["key"][k][...]

# Read h5 edges information data
with h5py.File(raw_data_path / "edges_data.h5", "r") as f:
    edges_data = {k: None for k in f["edges_data_sp"].keys()}

    for k in edges_data.keys():
        edges_data[k] = f["edges_data_sp"][k][...]
