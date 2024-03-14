import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_DATA_PATH = "data/raw/SF-incidents-4h.npz"

viz_output_path = pathlib.Path("data/viz/incidents_proportions")
viz_output_path.mkdir(parents=True, exist_ok=True)

raw_data = np.load(RAW_DATA_PATH)

incidents = raw_data["incident"]
incident_reshaped = incidents.reshape(5124, 100, 5)
sum_of_incidents = np.sum(incident_reshaped, axis=0)

df = pd.DataFrame(sum_of_incidents)

df = (df / incident_reshaped.shape[0]).copy()

plt.figure(figsize=(5, 8))

for col in df.columns:
    plt.scatter(x=np.arange(100), y=df[col], s=8)

plt.savefig(viz_output_path / "SF.png")
