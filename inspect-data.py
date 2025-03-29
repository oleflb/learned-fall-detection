import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as ps
import plotly.express as px


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


def main():
    data = pl.read_parquet("data.parquet")
    struct_columns = [col for col, schema in data.schema.items() if schema == pl.Struct]
    for column in struct_columns:
        data.hstack(unnest_column(data[column]), in_place=True)
        data.drop_in_place(column)

    data = data.drop(
        ps.contains("sensor_data.touch_sensors")
        | ps.contains("sensor_data.currents")
        | ps.contains("sensor_data.temperature_sensors")
    )

    px.scatter(
        data, x="time", y="Control.main_outputs.fall_state", color="robot_identifier"
    ).show()
    px.scatter(
        data.filter(pl.col("robot_identifier") == "10.1.24.32"),
        x="time",
        y="Control.main_outputs.robot_orientation.pitch",
        color="Control.main_outputs.fall_state",
    ).show()


if __name__ == "__main__":
    main()
