import flwr as fl
import matplotlib.pyplot as plt

round_accuracies = []


def weighted_average(metrics):
    accuracies = []

    for num_examples, m in metrics:
        if "accuracy" in m:
            accuracies.append(num_examples * m["accuracy"])

    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {"accuracy": 0}

    round_accuracy = sum(accuracies) / total_examples
    round_accuracies.append(round_accuracy)

    print(f"Round {len(round_accuracies)} Accuracy: {round_accuracy:.4f}")

    return {"accuracy": round_accuracy}


strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average
)


fl.server.start_server(
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)


# Plot graph
plt.figure(figsize=(10, 6))

plt.plot(range(1, len(round_accuracies) + 1), round_accuracies, marker='o')

plt.title("Federated Learning Accuracy over Rounds")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.grid(True)

for i, acc in enumerate(round_accuracies):
    plt.text(i + 1, acc, f"{acc:.2f}", ha='center')

plt.savefig("federated_accuracy_plot.png")
plt.show()