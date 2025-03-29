import polars as pl
import plotly.express as px

from learned_fall_detection.data_loading import load


def main():
    data = load("data.parquet")

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
