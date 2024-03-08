# gnn-crimes

## Basic Usage

All python packages necessary to run our code are in the `pyproject.toml` file. We suggest the use of Poetry for package dependency management.

### How to Download San Francisco Dataset

To download SF dataset, go to the root project directory and run the following script

```
python src/data/download/download_sf_data.py
```

This will create a folder `data/raw/sf` in the root directory with the raw dataset.

### How to Process San Francisco Dataset

To process SF dataset, go to the root project directory and run the following script

```
python src/data/process/process_full_sf_data.py
```

This will create a folder `data/processed/sf` in the root directory with the processed dataset.
