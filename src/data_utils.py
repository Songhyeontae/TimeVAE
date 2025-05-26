import os
import pickle
import pandas as pd

import yaml
import numpy as np

SCALER_FNAME = "scaler.pkl"


def load_yaml_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    return loaded


def load_data(data_dir: str, dataset: str) -> np.ndarray:
    """
    Load data from a dataset located in a directory.

    Args:
        data_dir (str): The directory where the dataset is located.
        dataset (str): The name of the dataset file (without the .npz extension).

    Returns:
        np.ndarray: The loaded dataset.
    """

    if dataset == "delhi":
        df = pd.read_csv(os.path.join(data_dir, "DailyDelhiClimateTest.csv"))
        df = df[["meantemp", "humidity", "wind_speed", "meanpressure"]]
        df = df.dropna().reset_index(drop=True)
        return df.values.astype(np.float32)
    else:
        return get_npz_data(os.path.join(data_dir, f"{dataset}.npz"))


def save_data(data: np.ndarray, output_file: str) -> None:
    """
    Save data to a .npz file.

    Args:
        data (np.ndarray): The data to save.
        output_file (str): The path to the .npz file to save the data to.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(output_file, data=data)


def get_npz_data(input_file: str) -> np.ndarray:
    """
    .npz 파일에서 데이터 추출
    Args:
        input_file (str): npz 파일 경로
    Returns:
        np.ndarray: .npz 파일에서 추출한 데이터
    """
    loaded = np.load(input_file)
    return loaded["data"]


def split_data(
    data: np.ndarray, valid_perc: float, shuffle: bool = True, seed: int = 123
) -> tuple[np.ndarray, np.ndarray]:
    """
    데이터를 training set, validation set 으로 스플릿

    Args:
        data (np.ndarray): 스플릿할 데이터
        valid_perc (float): validation set의 비중
        shuffle (bool, optional): 스플릿 전 데이터 셔플 여부. Default=True.
        seed (int, optional): 랜덤 셔플용 시드. Default=123.

    Returns:
        tuple[np.ndarray, np.ndarray]: training set, validation set을 포함하는 tuple.

    """
    N = data.shape[0]
    N_train = int(N * (1 - valid_perc))

    if shuffle:
        np.random.seed(seed)
        data = data.copy()
        np.random.shuffle(data)

    train_data = data[:N_train]
    valid_data = data[N_train:]
    return train_data, valid_data


class MinMaxScaler:
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """

    def fit_transform(self, data):
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data

    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data


def inverse_transform_data(data, scaler):
    return scaler.inverse_transform(data.copy())


def scale_data(train_data, valid_data):
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_valid_data = scaler.transform(valid_data)
    return scaled_train_data, scaled_valid_data, scaler


def save_scaler(scaler: MinMaxScaler, dir_path: str) -> None:
    """
    Save a MinMaxScaler to a file.

    Args:
        scaler (MinMaxScaler): The scaler to save.
        dir_path (str): The path to the directory where the scaler will be saved.

    Returns:
        None
    """
    os.makedirs(dir_path, exist_ok=True)
    scaler_fpath = os.path.join(dir_path, SCALER_FNAME)
    with open(scaler_fpath, "wb") as file:
        pickle.dump(scaler, file)


def load_scaler(dir_path: str) -> MinMaxScaler:
    """
    Load a MinMaxScaler from a file.

    Args:
        dir_path (str): The path to the file from which the scaler will be loaded.

    Returns:
        MinMaxScaler: The loaded scaler.
    """
    scaler_fpath = os.path.join(dir_path, SCALER_FNAME)
    with open(scaler_fpath, "rb") as file:
        scaler = pickle.load(file)
    return scaler
