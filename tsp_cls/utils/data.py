import pandas as pd


def add_species(row) -> pd.Series:
    """Adds a species column in the series"""
    _, species = row.binomial.split()
    return species


def get_image_path(row, data_path=None) -> str:
    """Returns the image path of row of DataFrame"""
    p1, p2, nm = row["image_path"].split("/")[-3:]
    image_path = f"training_data/{p1}/{p2}/{nm}"

    if data_path:
        return f"{data_path}/{image_path}"

    return image_path


# TODO: Do we need to handle for lists?
def field_getter(row, field:str):
    "Returns only the field Series in the row"
    return row[f'{field}']


def read_dataframe(path, dataset, *args, **kwargs) -> pd.DataFrame:
    """Reads the dataset and does some common modifications needed."""
    df = pd.read_csv(path / f"{dataset}", *args, **kwargs)
    df["species"] = df.apply(add_species, axis=1)

    return df


def sample_dataframe(df: pd.DataFrame, field: str, size: int) -> pd.DataFrame:
    "Returns a sampled DataFrame. Useful when training only on a subset of the data"
    to_consider = df[f"{field}"].value_counts()[:size]
    classes = to_consider.keys().tolist()
    df = df.loc[df[f"{field}"].isin(classes)]

    return df


if __name__ == "__main__":
    from tsp_cls.utils.root import get_data_root

    path = get_data_root()
    df = read_dataframe(path, "SnakeCLEF2021_min-train_metadata_PROD.csv", nrows=5)

    print("columns:", df.columns.to_list())

    # TODO: Showcase how sample works after fixing
    big_df = read_dataframe(
        path, "SnakeCLEF2021_min-train_metadata_PROD.csv", nrows=1000
    )
    sampled_df = sample_dataframe(big_df, "genus", 2)
    # print(sampled_df.head())
    print("Before Sampling")
    print(big_df["genus"].value_counts())
    print("After Sampling")
    print(sampled_df["genus"].value_counts())

    # getters on sample
    sample = df.iloc[0]
    print(f"image path of sample: {get_image_path(sample)}")
    print(f"Genus of sample: {(sample)}")
    print(f"Species of sample: {field_getter(sample, 'species')}")
    print(f"Binomial of sample: {field_getter(sample, 'binomial')}")
    print(f"Family of sample: {field_getter(sample, 'family')}")
    print(f"Country of sample: {field_getter(sample, 'country')}")

    print("Testing passing in data_path")
    print(f"image path of sample: {get_image_path(sample, data_path=path)}")
