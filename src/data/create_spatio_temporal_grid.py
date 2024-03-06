import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------- T
print("Reading dataset")
data = pd.read_csv("data/SF-Full_2003-2018.csv")
print("Done!")
# retirando posições que não fazem sentido

print("Processing dataset")
data = data[data["Y"] != 90].reset_index(drop=True)

# Retirando features inexpressivos para nós

data.drop(columns=data.columns[14:], inplace=True)
data.drop(columns=data.columns[:3], inplace=True)

data.drop(columns=["X", "Y"], inplace=True)  # Retirando por falta de precisão em relação ao 'location'
data.drop(columns=["PdDistrict", "Address"], inplace=True)  # Retirada opcional

data[["month", "day", "year"]] = data["Date"].str.split("/", expand=True)
data["datetime"] = data["year"] + "-" + data["month"] + "-" + data["day"] + " " + data["Time"] + ":00"


# Transformando as informações de data e hora em timestamp, para facilitar o tratamento

data["datetime"] = pd.to_datetime(data["datetime"])

data["timestamp"] = data["datetime"].astype("int64")

# Para escalar pq aparentemente vem em nanosegundos
data["timestamp"] = data["timestamp"].divide(10**9)
# então dividindo por 10**9 fica em segundos, timestamp padrão

data.drop(columns=["Date", "Time", "datetime"], inplace=True)


# Shiftando os timestamps, subtraindo o menor de todos os outros e tranformando coordenas lon/lat em km
data["location"] = data["location"].str.replace("POINT (", "", regex=False)
data["location"] = data["location"].str.replace(")", "", regex=False)
data[["lon", "lat"]] = data["location"].str.split(" ", expand=True)
data["lon"] = data["lon"].astype(float)
data["lat"] = data["lat"].astype(float)
data.drop(columns=["location"], inplace=True)

data["timestamp"] = data["timestamp"] - data["timestamp"].min()


print("Finished")
# ------------------------------------------------------------------------


# Primeira função: Divide o mapa de São Francisco em NxN grids e nos dá os pares de coordenadas de cada vértice
# assim, se quisermos em alguma outra função selecionar um único grid para análise, precisaremos de 4 pares de
# coordenadas dessa lista, teremos os limitadores tanto em X quanto em Y a partir deles e poderemos observar todos
# as ocorrências neste limite

# funções auxiliares


def vert_list(N, df):
    passo_X = (df["lon"].max() - df["lon"].min()) / N
    passo_Y = (df["lat"].max() - df["lat"].min()) / N

    vertices = [[df["lon"].min()], [df["lat"].min()]]

    for i in range(1, N + 1):
        vertices[0] += [vertices[0][0] + i * passo_X]
        vertices[1] += [vertices[1][0] + i * passo_Y]
    return vertices


teste = vert_list(10, data)

# Segunda função: Recebe o nosso dataframe, 4 coordenadas, um x inicial e um final, um y inicial e um final
# e um intervalo de tempo em segundos. Nos devolve um dataframe reduzido contendo os dados da região
# dentro do intervalo espacial no intervalo de tempo pedido.


def df_reduzido(df, x_initial, x_final, y_initial, y_final, t_initial, t_final):
    new_df = df.loc[
        (df["timestamp"] >= t_initial)
        & (df["timestamp"] <= t_final)
        & (df["lon"] >= x_initial)
        & (df["lat"] >= y_initial)
        & (df["lon"] <= x_final)
        & (df["lat"] <= y_final)
    ]
    breakpoint()
    return new_df


new_df = df_reduzido(data, teste[0][0], teste[0][1], teste[1][8], teste[1][9], 0, data["timestamp"].max())


# ----------------------------------------------------------------- MAPA -


# create a map
this_map = folium.Map(prefer_canvas=True)


def plotDot(point):
    """input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map"""
    folium.CircleMarker(location=[point["lat"], point["lon"]], radius=2, weight=5).add_to(this_map)


# use df.apply(,axis=1) to "iterate" through every row in your dataframe
new_df.apply(plotDot, axis=1)


# Set the zoom to the maximum possible
this_map.fit_bounds(this_map.get_bounds())

# Save the map to an HTML file
# this_map.save('html_map_output/simple_dot_plot.html')

this_map
