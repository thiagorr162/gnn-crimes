import argparse
import pathlib

import folium
import pandas as pd

from src.utils.data_utils import create_grid, restrict_dataset

parser = argparse.ArgumentParser()

parser.add_argument(
    "-N",
    "--n_grid",
    help="Grid granularity. E.g. NxN.",
    type=int,
    default=20,
)
parser.add_argument(
    "-c",
    "--coordiantes",
    help="Spatial grid coordiantes. E.g. [0 1 2 3] will user x from 0 to 1 and y from 2 to 3.",
    nargs="+",
    type=int,
    default=[0, 1, 2, 3],
)
parser.add_argument(
    "-t",
    "--time",
    help="Temporal coordiantes in timestamp units normalized to start at 0. E.g. [0 1000].",
    nargs="+",
    type=int,
    default=[0, 1000000],
)

args = parser.parse_args()

processed_dataset_path = pathlib.Path("data/processed/processed_full_sf.csv")

output_viz_path = pathlib.Path("visualization/folium/")
output_viz_path.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(processed_dataset_path)

spatio_temporal_grid = create_grid(args.n_grid, df)

x_initial = spatio_temporal_grid[0][args.coordiantes[0]]
x_final = spatio_temporal_grid[0][args.coordiantes[1]]

y_initial = spatio_temporal_grid[1][args.coordiantes[2]]
y_final = spatio_temporal_grid[1][args.coordiantes[3]]

t_initial = args.time[0]
t_final = args.time[1]

restricted_df = restrict_dataset(
    df,
    x_initial,
    x_final,
    y_initial,
    y_final,
    t_initial,
    t_final,
)


folium_map = folium.Map(prefer_canvas=True)

n_rows, n_columns = restricted_df.shape

for i in range(n_rows):
    lat = restricted_df.iloc[i]["latitude"]
    lon = restricted_df.iloc[i]["longitude"]

    folium.CircleMarker(
        location=[lat, lon],
        radius=2,
        weight=5,
    ).add_to(folium_map)

# Set the zoom to the maximum possible
folium_map.fit_bounds(folium_map.get_bounds())

folium_map.save(output_viz_path / "map.html")
