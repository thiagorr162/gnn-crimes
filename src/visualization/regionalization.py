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

restricted_df["has_resolution"] = restricted_df["resolution"].notna() * 0.5

# category_dummies = pd.get_dummies(restricted_df["category"]) * 0.01
# restricted_df = restricted_df.join(category_dummies)

restricted_df = restricted_df.drop(
    columns=[
        "descript",
        "dayofweek",
        "datetime",
        "timestamp",
        "resolution",
        "category",
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
    n_clusters=3,
    random_state=0,
).fit(restricted_df)
restricted_df["kmeans_cluster"] = kmeans.predict(restricted_df)
restricted_df["kmeans_cluster"] = restricted_df["kmeans_cluster"].map(color_map)


folium_map = folium.Map(prefer_canvas=True)


def plotDot(point):
    """input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map"""
    folium.CircleMarker(
        location=[point["latitude"], point["longitude"]], radius=2, weight=5, color=point["kmeans_cluster"]
    ).add_to(folium_map)


restricted_df.apply(plotDot, axis=1)

folium_map.fit_bounds(folium_map.get_bounds())

folium_map.save(output_path / "map.html")
