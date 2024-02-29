import pathlib

import numpy as np
import pandas as pd

raw_data_path = "data/raw/full_sf.csv"

processed_data_output_path = pathlib.Path("data/processed")
processed_data_output_path.mkdir(parents=True, exist_ok=True)

raw_data = pd.read_csv(raw_data_path)

# TODO: verify if this is the only problemati case
processed_data = raw_data[raw_data["Y"] != 90].reset_index(drop=True)

# TODO: Do not drop columns by index, instead drop by name
processed_data.drop(columns=processed_data.columns[14:], inplace=True)
processed_data.drop(columns=processed_data.columns[:3], inplace=True)

processed_data.drop(columns=["X", "Y"], inplace=True)
processed_data.drop(columns=["PdDistrict", "Address"], inplace=True)

# Creating time columns
processed_data[["month", "day", "year"]] = processed_data["Date"].str.split("/", expand=True)
processed_data["datetime"] = (
    processed_data["year"] + "-" + processed_data["month"] + "-" + processed_data["day"] + " " + processed_data["Time"]
)

processed_data["month"] = processed_data["month"].astype(int)
processed_data["day"] = processed_data["day"].astype(int)
processed_data["year"] = processed_data["year"].astype(int)

processed_data["datetime"] = pd.to_datetime(processed_data["datetime"])
processed_data["timestamp"] = processed_data["datetime"].astype("int64")
processed_data["timestamp"] = processed_data["timestamp"].divide(10**9)
processed_data["timestamp"] = processed_data["timestamp"] - processed_data["timestamp"].min()

processed_data.drop(columns=["Date", "Time", "datetime"], inplace=True)


# Creating lat/lon columns
processed_data["location"] = processed_data["location"].str.replace("POINT (", "", regex=False)
processed_data["location"] = processed_data["location"].str.replace(")", "", regex=False)
processed_data[["longitude", "latitude"]] = processed_data["location"].str.split(" ", expand=True)
processed_data["longitude"] = processed_data["longitude"].astype(float)
processed_data["latitude"] = processed_data["latitude"].astype(float)
processed_data.drop(columns=["location"], inplace=True)


# Processing string columns
processed_data["Category"] = processed_data["Category"].str.lower()
processed_data["Category"] = processed_data["Category"].str.strip()
processed_data["Category"] = processed_data["Category"].str.replace(" ", "_")
processed_data["Category"] = processed_data["Category"].str.replace(",", " ")

processed_data["Descript"] = processed_data["Descript"].str.lower()
processed_data["Descript"] = processed_data["Descript"].str.strip()
processed_data["Descript"] = processed_data["Descript"].str.replace(",", " ")

processed_data["DayOfWeek"] = processed_data["DayOfWeek"].str.lower()
processed_data["DayOfWeek"] = processed_data["DayOfWeek"].str.strip()
processed_data["DayOfWeek"] = processed_data["DayOfWeek"].str.replace(",", " ")

processed_data["Resolution"] = processed_data["Resolution"].str.lower()
processed_data["Resolution"] = processed_data["Resolution"].str.strip()
processed_data["Resolution"] = processed_data["Resolution"].str.replace(",", " ")

# Change string None to NaN
processed_data = processed_data.replace("none", np.nan)


# Make all columns names lower case
processed_data.columns = [x.lower() for x in processed_data.columns]


processed_data.to_csv(processed_data_output_path / "processed_full_sf.csv")
