def create_grid_given_lat_long(N, df):
    passo_X = (df["longitude"].max() - df["longitude"].min()) / N
    passo_Y = (df["latitude"].max() - df["latitude"].min()) / N

    vertices = [[df["longitude"].min()], [df["latitude"].min()]]

    for i in range(1, N + 1):
        vertices[0] += [vertices[0][0] + i * passo_X]
        vertices[1] += [vertices[1][0] + i * passo_Y]

    return vertices


def restrict_dataset(df, x_initial, x_final, y_initial, y_final, t_initial, t_final):
    new_df = df.loc[
        (df["timestamp"] >= t_initial)
        & (df["timestamp"] <= t_final)
        & (df["longitude"] >= x_initial)
        & (df["latitude"] >= y_initial)
        & (df["longitude"] <= x_final)
        & (df["latitude"] <= y_final)
    ]
    return new_df
