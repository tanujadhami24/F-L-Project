import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from model import DiabetesModel
from utils import load_data

# 🔐 Differential Privacy Toggle
USE_DP = True  # change to False if needed


# 🔐 Add noise for privacy
def add_noise(parameters, noise_scale=0.005):
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

    # 📦 Get model weights
    def get_parameters(self, config):
        params = [val.cpu().numpy() for val in self.model.state_dict().values()]

        if USE_DP:
            params = add_noise(params)

        return params

    # 🔁 Set model weights
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    # 🧠 Train model
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        print(f"Client {self.cid} training...")

        self.model.train()

        for epoch in range(5):  # ✅ balanced training
            for X, y in self.trainloader:
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

        # ✅ Evaluate UPDATED model (FIXED)
        loss, size, metrics = self.evaluate(None, config)

        return self.get_parameters(config), len(self.trainloader.dataset), metrics

    # 📊 Evaluate model (NO overwriting)
    def evaluate(self, parameters, config):

        self.model.eval()

        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in self.testloader:
                outputs = self.model(X)

                # ✅ balanced threshold
                preds = (outputs > 0.5).float()

                correct += (preds == y).sum().item()
                total += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # Convert to simple lists
        all_preds = [int(p[0]) for p in all_preds]
        all_labels = [int(l[0]) for l in all_labels]

        # Metrics
        accuracy = correct / total if total != 0 else 0

        tp = sum((p == 1 and l == 1) for p, l in zip(all_preds, all_labels))
        fp = sum((p == 1 and l == 0) for p, l in zip(all_preds, all_labels))
        fn = sum((p == 0 and l == 1) for p, l in zip(all_preds, all_labels))

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) != 0 else 0

        print(f"Client {self.cid} → Acc: {accuracy:.3f}, Prec: {precision:.3f}, Recall: {recall:.3f}")

        return 0.0, len(self.testloader.dataset), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }


# 🚀 Start client
def main():
    client_id = random.randint(0, 1)

    print(f"Client {client_id} started")

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=FlowerClient(client_id),
    )


if __name__ == "__main__":
    main()