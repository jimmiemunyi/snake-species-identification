import pandas as pd


def add_species(row) -> pd.Series:
    """Adds a species column in the series"""
    _, species = row.binomial.split()
    return species


# TODO: fix path not working because of datapath
def get_image_path(row, data_path=None) -> str:
    """Returns the image path of row of DataFrame"""
    p1, p2, nm = row["image_path"].split("/")[-3:]
    image_path = f"training_data/{p1}/{p2}/{nm}"

    if data_path:
        return f"{data_path}/{image_path}"

    return image_path


def get_genus(row):
    """Returns the genus of row of Dataframe"""
    return row.genus


def get_species(row):
    """Returns the species of row of Dataframe"""
    return row.species


def get_binomial(row):
    """Returns the binomial of row of Dataframe"""
    return row.binomial


def get_family(row):
    """Returns the family of row of Dataframe"""
    return row.family


def get_country(row):
    """Returns the country of row of Dataframe"""
    return row.country


if __name__ == "__main__":
    from tsp_cls.utils.root import get_data_root

    path = get_data_root()
    df = pd.read_csv(path / "SnakeCLEF2021_train_metadata_PROD.csv", nrows=5)

    # original dataframe columns
    print("original columns:", df.columns.to_list())

    # adding our species column
    df["species"] = df.apply(add_species, axis=1)
    print("edited columns:", df.columns.to_list())

    # getters on sample
    sample = df.iloc[0]
    print(f"image path of sample: {get_image_path(sample)}")
    print(f"Genus of sample: {get_genus(sample)}")
    print(f"Species of sample: {get_species(sample)}")
    print(f"Binomial of sample: {get_binomial(sample)}")
    print(f"Family of sample: {get_family(sample)}")
    print(f"Country of sample: {get_country(sample)}")

    print("Testing passing in data_path")
    print(f"image path of sample: {get_image_path(sample, data_path=path)}")
