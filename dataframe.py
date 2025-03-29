import polars as pl


def xyz_name(index: int) -> str:
    return {
        0: "x",
        1: "y",
        2: "z",
    }[index]


def convert_to_dataframe(data: pl.Series | pl.DataFrame) -> pl.DataFrame:
    if isinstance(data, pl.Series):
        return data.to_frame()
    return data


def unnest_column(data: pl.Series) -> pl.DataFrame | pl.Series:
    if data.dtype == pl.List:
        data = data.list.to_struct(fields=xyz_name)

    if data.dtype != pl.Struct:
        return data

    return pl.concat(
        (
            convert_to_dataframe(unnest_column(series))
            for series in data.struct.unnest().rename(
                lambda name: f"{data.name}.{name}"
            )
        ),
        how="horizontal",
    )


def load(path: str):
    df = pl.read_parquet(path).with_columns(
        (pl.col("time") - pl.col("time").min())
        .over("robot_identifier", "match_identifier")
        .dt.total_seconds()
        .alias("time_in_game"),
    )
    struct_columns = [col for col, schema in df.schema.items() if schema == pl.Struct]
    for column in struct_columns:
        df.hstack(unnest_column(df[column]), in_place=True)
        df.drop_in_place(column)
    return df.rechunk()
