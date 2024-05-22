from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import json

from datasets import load_dataset


class DataLoader(ABC):
    @abstractmethod
    def get_train_data(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_test_data(self) -> None:
        raise NotImplementedError


class MNISTData(DataLoader):
    def get_train_data(self) -> None:
        dataset = load_dataset("mnist", split="train")

        vectors = np.asarray([np.asarray(image) for image in dataset["image"]])
        labels = dataset["label"]

        vectors = vectors / 255.0
        vectors = vectors.reshape(vectors.shape[0], -1)

        return vectors, labels

    def get_test_data(self) -> None:
        pass


def data_logger(df: pd.DataFrame, filename: str, metadata: dict) -> None:
    # Save DataFrame to Parquet file
    df.to_parquet("../data/interim/" + filename + ".parquet", index=False)

    # Update JSON file with information from the dictionary
    # json file path
    json_file_path = "../data/data.json"

    # Load existing JSON file or create a new one
    try:
        with open(json_file_path, "r") as json_file:
            json_data = json.load(json_file)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        json_data = {}

    # Create or update the 'interim' section in the JSON data
    interim_section = json_data.get("interim", {})

    # Create an item named after the filename with elements from info_dict
    interim_section[filename] = metadata

    # Update the 'interim' section in the JSON data
    json_data["interim"] = interim_section

    # Write the updated JSON data back to the file
    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=2)


def data_loader(filename):
    # Read JSON file
    json_file_path = "../data/data.json"
    try:
        with open(json_file_path, "r") as json_file:
            json_data = json.load(json_file)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        raise ValueError(f"JSON file '{json_file_path}' not found or invalid.")

    # Extract metadata from the 'interim' section
    interim_section = json_data.get("interim", {})
    metadata = interim_section.get(filename, {})

    # Load Parquet file into DataFrame
    parquet_file_path = "../data/interim/" + filename
    try:
        df = pd.read_parquet(parquet_file_path)
    except FileNotFoundError:
        raise ValueError(f"Parquet file '{parquet_file_path}' not found.")

    return df, metadata


if __name__ == "__main__":
    pass
