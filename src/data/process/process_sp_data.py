import pathlib
import pickle

import networkx as nx
import pandas as pd

raw_data_path = "data/raw/sp/sp_road_net/SaoPaolo_Edgelist.csv"

processed_data_output_path = pathlib.Path("data/processed/sp/")
processed_data_output_path.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(raw_data_path)

edges_as_nodes = data.groupby("EDGE").agg(
    {
        "XCoord": "mean",
        "YCoord": "mean",
        "START_NODE": "nunique",  # conta elementos únicos
        "END_NODE": "nunique",
        "LENGTH": "mean",
    }
)

edges_as_nodes["NUM_NODE"] = edges_as_nodes["START_NODE"]
edges_as_nodes.drop(["START_NODE", "END_NODE"], axis=1, inplace=True)

g = nx.DiGraph()
g.add_nodes_from(edges_as_nodes.to_dict("index").items())

adjacency_as_edges = set()
edges = data.drop(["XCoord", "YCoord", "LENGTH"], axis=1)

adjacency = pd.merge(edges, edges, left_on="START_NODE", right_on="END_NODE")[["EDGE_x", "EDGE_y"]]
adjacency = adjacency[adjacency["EDGE_x"] != adjacency["EDGE_y"]]

adjacency_as_edges = adjacency_as_edges.union(
    set(map(lambda record: (record["EDGE_x"], record["EDGE_y"]), adjacency.to_dict("records")))
)


adjacency = pd.merge(edges, edges, left_on="END_NODE", right_on="START_NODE")[["EDGE_x", "EDGE_y"]]
adjacency = adjacency[adjacency["EDGE_x"] != adjacency["EDGE_y"]]
adjacency_as_edges = adjacency_as_edges.union(
    set(map(lambda record: (record["EDGE_x"], record["EDGE_y"]), adjacency.to_dict("records")))
)

g.add_edges_from(adjacency_as_edges)

g = nx.convert_node_labels_to_integers(g)

#  read_gpickle and write_gpickle were removed in 3.0.

with open(processed_data_output_path / "saopaulo_roadmap.gpickle", "wb") as f:
    pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
