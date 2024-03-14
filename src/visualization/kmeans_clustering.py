import pathlib

import folium
import pandas as pd
from sklearn.cluster import KMeans

df_path = pathlib.Path("data/processed/processed_full_sf.csv")

df = pd.read_csv(df_path)


output_path = pathlib.Path("personal")

# TODO: Use argparse

initial_date = "2017-01-01"
final_date = "2017-05-01"

restricted_df = df.loc[(df["datetime"] >= initial_date) & (df["datetime"] <= final_date)]

# restricted_df["has_resolution"] = restricted_df["resolution"].notna() * 0.1

category_dummies = pd.get_dummies(restricted_df["reduced_categories"]) * 0.1
restricted_df = restricted_df.join(category_dummies)

restricted_df = restricted_df.drop(
    columns=[
        "descript",
        "dayofweek",
        "datetime",
        "timestamp",
        "resolution",
        "category",
        "reduced_categories",
        "day",
        "month",
        "year",
        "day_number",
    ]
)


colors = [
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "darkred",
    "lightred",
    "beige",
    "darkblue",
    "darkgreen",
    "cadetblue",
    "darkpurple",
    "white",
    "pink",
    "lightblue",
    "lightgreen",
    "gray",
    "black",
    "lightgray",
]

color_map = {i: colors[i] for i in range(len(colors))}


kmeans = KMeans(
    n_clusters=5,
    random_state=0,
).fit(restricted_df)


restricted_df["kmeans_cluster"] = kmeans.predict(restricted_df)
restricted_df["kmeans_cluster"] = restricted_df["kmeans_cluster"].map(color_map)


folium_map = folium.Map(prefer_canvas=True)

n_rows, n_columns = restricted_df.shape

for i in range(n_rows):
    lat = restricted_df.iloc[i]["latitude"]
    lon = restricted_df.iloc[i]["longitude"]

    color = restricted_df.iloc[i]["kmeans_cluster"]

    folium.CircleMarker(
        location=[lat, lon],
        radius=2,
        weight=3,
        color=color,
    ).add_to(folium_map)

folium_map.fit_bounds(folium_map.get_bounds())

folium_map.save(output_path / "map.html")
