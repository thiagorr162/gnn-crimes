import pandas as pd

# First Step

df1 = pd.read_csv("elliptic_txs_features_original.csv", header=None)
df1 = df1.reset_index()


new_df1 = df1[[0, "index"]].copy()
new_df1.rename(columns={0: "originalId", "index": "contiguosId"}, inplace=True)


df1.drop([0], axis=1, inplace=True)
df1.rename(columns={"index": 0}, inplace=True)


df1[0] = df1[0].astype(float)
df1[1] = df1[1].astype(float)


new_df1.to_csv("elliptic_txs_orig2contiguos.csv", index=False)
df1.to_csv("elliptic_txs_features.csv", index=False)


# Second Step

df2 = pd.read_csv("elliptic_txs_classes_original.csv")


df2["txId"] = df1[0]


dicio = {"unknown": -1.0, "1": 1.0, "2": 0.0}
df2["class"] = df2["class"].map(dicio)


df2.to_csv("elliptic_txs_classes.csv", index=False)


# Third Step

dicio = {"txId": [], "timestep": []}
df3 = pd.DataFrame(data=dicio)


df3["txId"] = df2["txId"]
df3["timestep"] = df1[1] - 1.0


df3.to_csv("elliptic_txs_nodetime.csv", index=False)


# Fourth Step


df4 = pd.read_csv("elliptic_txs_edgelist_original.csv")


dc_mapeamento1 = new_df1.set_index("originalId").to_dict()["contiguosId"]


df4["txId1"] = df4["txId1"].map(dc_mapeamento1)
df4["txId2"] = df4["txId2"].map(dc_mapeamento1)


dc_mapeamento2 = df3.set_index("txId").to_dict()["timestep"]


df4["timestep1"] = df4["txId1"].map(dc_mapeamento2)
df4["timestep2"] = df4["txId2"].map(dc_mapeamento2)


df4.drop(columns=["timestep2"], inplace=True)
df4.rename(columns={"timestep1": "timestep"}, inplace=True)


df4.to_csv("elliptic_txs_edgelist_timed.csv", index=False)
