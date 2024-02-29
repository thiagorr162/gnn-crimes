import pathlib

import folium
import pandas as pd

from src.utils.data_utils import create_grid, restrict_dataset

# TODO: Refactor code using argparse

processed_dataset_path = pathlib.Path("data/processed/processed_full_sf.csv")

output_viz_path = pathlib.Path("visualization/folium/")
output_viz_path.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(processed_dataset_path)

spatio_temporal_grid = create_grid(20, df)


restricted_df = restrict_dataset(
    df,
    spatio_temporal_grid[0][10],
    spatio_temporal_grid[0][12],
    spatio_temporal_grid[1][8],
    spatio_temporal_grid[1][10],
    df["timestamp"].max() // 2,
    df["timestamp"].max(),
)

folium_map = folium.Map(prefer_canvas=True)


def plotDot(point):
    """input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map"""
    folium.CircleMarker(location=[point["latitude"], point["longitude"]], radius=2, weight=5).add_to(folium_map)


restricted_df.apply(plotDot, axis=1)


# Set the zoom to the maximum possible
folium_map.fit_bounds(folium_map.get_bounds())

folium_map.save(output_viz_path / "map.html")
