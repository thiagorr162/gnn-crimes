import pathlib

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

sf_shapefile_path = pathlib.Path("data/raw/sf/shapefile/tl_2017_06075_roads/tl_2017_06075_roads.shp")
df_path = pathlib.Path("data/processed/sf/processed_full_sf.csv")

output_path = pathlib.Path("visualization/geopandas")
output_path.mkdir(exist_ok=True, parents=True)

sf_shapefile = gpd.read_file(sf_shapefile_path)

df = pd.read_csv(df_path)


initial_date = "2017-01-01"
final_date = "2017-01-02"

restricted_df = df.loc[(df["datetime"] >= initial_date) & (df["datetime"] <= final_date)]

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

categories_colors = {}

for i, cat in enumerate(restricted_df["reduced_categories"].unique()):
    categories_colors[cat] = colors[i]


gdf = gpd.GeoDataFrame(
    restricted_df,
    geometry=gpd.points_from_xy(restricted_df["longitude"], restricted_df["latitude"]),
    crs="EPSG:4326",
)

fig, ax = plt.subplots(figsize=(10, 10))

gdf.plot(
    ax=ax,
    column="reduced_categories",
    legend=True,
)

sf_shapefile.plot(ax=ax, alpha=0.2)

plt.savefig(output_path / "map.png")
