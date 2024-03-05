import pathlib

import pandas as pd


def create_grid(N, df):
    passo_X = (df["longitude"].max() - df["longitude"].min()) / N
    passo_Y = (df["latitude"].max() - df["latitude"].min()) / N

    vertices = [[df["longitude"].min()], [df["latitude"].min()]]

    for i in range(1, N + 1):
        vertices[0] += [vertices[0][0] + i * passo_X]
        vertices[1] += [vertices[1][0] + i * passo_Y]

    return vertices


def restrict_dataset(
    df,
    x_initial,
    x_final,
    y_initial,
    y_final,
    t_initial,
    t_final,
    time_column_name="timestamp",
):
    assert time_column_name in df.columns

    new_df = df.loc[
        (df[time_column_name] >= t_initial)
        & (df[time_column_name] <= t_final)
        & (df["longitude"] >= x_initial)
        & (df["latitude"] >= y_initial)
        & (df["longitude"] <= x_final)
        & (df["latitude"] <= y_final)
    ]
    return new_df


if __name__ == "__main__":
    processed_dataset_path = pathlib.Path("data/processed/processed_full_sf.csv")
    df = pd.read_csv(processed_dataset_path)

    n_grid = 20

    spatio_temporal_grid = create_grid(n_grid, df)

    restricted_df = restrict_dataset(
        df,
        spatio_temporal_grid[0][10],
        spatio_temporal_grid[0][12],
        spatio_temporal_grid[1][8],
        spatio_temporal_grid[1][10],
        df["timestamp"].max() // 2,
        df["timestamp"].max(),
    )
    breakpoint()
