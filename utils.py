import pandas as pd
import torch
import numpy as np
import random
import time

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(client_id, num_clients=2):

    # Random seed (different every run)
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv("diabetes.csv")

    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Random split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Split among clients
    X_splits = np.array_split(X_train, num_clients)
    y_splits = np.array_split(y_train, num_clients)

    X_client = X_splits[client_id]
    y_client = y_splits[client_id]

    trainset = TensorDataset(
        torch.tensor(X_client, dtype=torch.float32),
        torch.tensor(y_client, dtype=torch.float32)
    )

    testset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    return DataLoader(trainset, batch_size=16, shuffle=True), DataLoader(testset, batch_size=16)