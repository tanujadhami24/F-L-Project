import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from model import DiabetesModel
from utils import load_data


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.model = DiabetesModel()
        self.trainloader, self.testloader = load_data(cid)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.train()
        for epoch in range(1):  # can increase later
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

        with torch.no_grad():
            for X, y in self.testloader:
                outputs = self.model(X)
                preds = (outputs > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)

        accuracy = correct / total
        return 0.0, len(self.testloader.dataset), {"accuracy": accuracy}


def main():
    client_id = int(input("Enter client ID (0 or 1): "))
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=FlowerClient(client_id),
    )


if __name__ == "__main__":
    main()