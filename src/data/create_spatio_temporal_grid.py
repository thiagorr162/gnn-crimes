import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------- TRATAMENTO DO DATASET ------------------------------------------------------------

data = pd.read_csv("data/SF-Full_2003-2018.csv")

## retirando posições que não fazem sentido

data = data[data["Y"] != 90].reset_index(drop=True)

## Retirando features inexpressivos para nós

data.drop(columns=data.columns[14:], inplace=True)
data.drop(columns=data.columns[:3], inplace=True)
data.drop(columns=["X", "Y"], inplace=True)  # Retirando por falta de precisão em relação ao 'location'
data.drop(columns=["PdDistrict", "Address"], inplace=True)  # Retirada opcional


data_hora = []
for i in range(len(data["Date"])):
    dia = data["Date"][i]
    data_hora += [dia[-4:] + "-" + dia[:2] + "-" + dia[3:5] + " " + data["Time"][i] + ":00"]


## Transformando as informações de data e hora em timestamp, para facilitar o tratamento

df_datetime = pd.DataFrame({"datetime": data_hora})

df_datetime["datetime"] = pd.to_datetime(df_datetime["datetime"])

df_datetime["timestamp"] = df_datetime["datetime"].astype("int64")

df_datetime["timestamp"] = df_datetime["timestamp"].divide(
    10**9
)  # Isso aqui é pra escalar pq aparentemente vinha em nanosegundos
# então dividindo por 10**9 fica em segundos, timestamp padrão


data.drop(columns=["Date", "Time"], inplace=True)
data.insert(3, "Timestamp", df_datetime["timestamp"])


## Shiftando os timestamps, subtraindo o menor de todos os outros e tranformando coordenas lon/lat em km

new_X, new_Y, new_T = [], [], []
min_T = data["Timestamp"].min()

for i in range(len(data["location"])):
    loc = data["location"][i][7:-1].split(" ")

    lon, lat = float(loc[0]), float(loc[1])

    new_X += [lon]
    new_Y += [lat]

    new_T += [data["Timestamp"][i] - min_T]


data.insert(3, "timestamp_shift", new_T)
data.insert(5, "lon", new_X)
data.insert(6, "lat", new_Y)
data.drop(columns=["location", "Timestamp"], inplace=True)


# ------------------------------------------------------------------------------ FUNÇÕES ---------------------------------------------------------------------------


## Primeira função: Divide o mapa de São Francisco em NxN grids e nos dá os pares de coordenadas de cada vértice
## assim, se quisermos em alguma outra função selecionar um único grid para análise, precisaremos de 4 pares de
## coordenadas dessa lista, teremos os limitadores tanto em X quanto em Y a partir deles e poderemos observar todos
## as ocorrências neste limite

# funções auxiliares


# closest_x = lambda x: data.iloc[(data["X_reescalado (km)"] - x).abs().argsort()[:1]]["X_reescalado (km)"].tolist()[0]
# closest_y = lambda y: data.iloc[(data["Y_reescalado (km)"] - y).abs().argsort()[:1]]["Y_reescalado (km)"].tolist()[0]


def vert_list(N, df):
    passo_X = (df["lon"].max() - df["lon"].min()) / N
    passo_Y = (df["lat"].max() - df["lat"].min()) / N

    vertices = [[], []]

    for i in range(N + 1):
        closest_X = data.iloc[(data["lon"] - (passo_X * i)).abs().argsort()[:1]]["lon"].tolist()[0]
        closest_Y = data.iloc[(data["lat"] - (passo_Y * i)).abs().argsort()[:1]]["lat"].tolist()[0]

        vertices[0] += [closest_X]
        vertices[1] += [closest_Y]

    return vertices


teste = vert_list(10, data)
print(teste)

## Segunda função: Recebe o nosso dataframe, 4 coordenadas, um x inicial e um final, um y inicial e um final
## e um intervalo de tempo em segundos. Nos devolve um dataframe reduzido contendo os dados da região
## dentro do intervalo espacial no intervalo de tempo pedido.


def df_reduzido(df, x_initial, x_final, y_initial, y_final, t_initial, t_final):
    new_df = df.loc[
        (df["Timestamp_shift"] >= t_initial)
        & (df["Timestamp_shift"] <= t_final)
        & (df["lon"] >= x_initial)
        & (df["lat"] >= y_initial)
        & (df["lon"] <= x_final)
        & (df["lat"] <= y_final)
    ]

    return new_df


new_df = df_reduzido(data, teste[0][0], teste[0][1], teste[1][8], teste[1][9], 0, data["Timestamp_shift"].max())
breakpoint()
