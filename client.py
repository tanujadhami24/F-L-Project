import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from model import DiabetesModel
from utils import load_data

import numpy as np

# 🔐 Differential Privacy Noise
USE_DP = True  # 🔁 change to False to disable privacy
def add_noise(parameters, noise_scale=0.01):
    noisy_params = []
    for param in parameters:
        noise = np.random.normal(0, noise_scale, param.shape)
        noisy_params.append(param + noise)
    return noisy_params


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.model = DiabetesModel()
        self.trainloader, self.testloader = load_data(cid)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    # ✅ FIXED (INSIDE CLASS)
    def get_parameters(self, config):
        params = [val.cpu().numpy() for val in self.model.state_dict().values()]

        if USE_DP:
            params = add_noise(params, noise_scale=0.01)

        return params


    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.train()
        for epoch in range(1):
            for X, y in self.trainloader:
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self.model.eval()

        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in self.testloader:
                outputs = self.model(X)
                preds = (outputs > 0.5).float()

                correct += (preds == y).sum().item()
                total += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # 🔥 Convert to simple lists (important)
        all_preds = [int(p[0]) for p in all_preds]
        all_labels = [int(l[0]) for l in all_labels]

        # 🔥 Accuracy
        accuracy = correct / total

        # 🔥 Precision, Recall, F1 (manual calculation)
        tp = sum((p == 1 and l == 1) for p, l in zip(all_preds, all_labels))
        fp = sum((p == 1 and l == 0) for p, l in zip(all_preds, all_labels))
        fn = sum((p == 0 and l == 1) for p, l in zip(all_preds, all_labels))

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) != 0 else 0
        )

        return 0.0, len(self.testloader.dataset), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }


def main():
    client_id = int(input("Enter client ID (0 or 1): "))
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=FlowerClient(client_id),
    )


if __name__ == "__main__":
    main()