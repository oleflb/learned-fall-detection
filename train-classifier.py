from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import classification_report
import polars as pl
import polars.selectors as ps

from dataframe import load


def split_dataframe(dataframe: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    split_expression = pl.col("time_in_game") < pl.col("time_in_game").quantile(0.8)
    return dataframe.with_columns(train=split_expression).partition_by("train")


def main(dataframe: pl.DataFrame):
    FEATURES = ps.contains("robot_orientation") | ps.contains("center_of_mass")
    TARGET = "Control.main_outputs.fall_state"
    dataframe = dataframe.with_columns(
        pl.col(TARGET).rank("dense").alias(TARGET),
    )

    nfeatures = dataframe.select(FEATURES).shape[1]
    print("Selected features:", nfeatures)

    train_df, val_df = split_dataframe(dataframe)

    train_X, train_y = train_df.select(FEATURES), train_df[TARGET]
    val_X, val_y = val_df.select(FEATURES), val_df[TARGET]

    pipeline = TransformedTargetRegressor(
        make_pipeline(
            LinearSVC(class_weight="balanced"),
        ),
        transformer=OrdinalEncoder(),
    )

    pipeline.fit(train_X, train_y)
    print("Train score:", pipeline.score(train_X, train_y))
    print("Validation score:", pipeline.score(val_X, val_y))
    print(classification_report(val_y, pipeline.predict(val_X)))


if __name__ == "__main__":
    df = load("data.parquet")
    main(df)
