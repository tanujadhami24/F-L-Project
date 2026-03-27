import flwr as fl
import matplotlib.pyplot as plt
import os

# 🔥 Reset accuracy list every run
round_accuracies = []


def weighted_average(metrics):
    global round_accuracies

    accuracies = []

    for num_examples, m in metrics:
        if "accuracy" in m:
            accuracies.append(num_examples * m["accuracy"])

    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {"accuracy": 0}

    round_accuracy = sum(accuracies) / total_examples

    round_accuracies.append(round_accuracy)

    # 🔥 Debug print (VERY IMPORTANT)
    print("Round accuracies so far:", round_accuracies)

    return {"accuracy": round_accuracy}


# Strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average
)


# Start server
fl.server.start_server(
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)


# 🔥 BEFORE SAVING GRAPH → DELETE OLD ONE (IMPORTANT)
graph_path = "static/federated_accuracy_plot.png"

if os.path.exists(graph_path):
    os.remove(graph_path)


# 🔥 Plot graph
plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(round_accuracies) + 1),
    round_accuracies,
    marker='o',
    linestyle='-'
)

plt.title("Federated Learning Accuracy over Rounds", fontsize=14)
plt.xlabel("Rounds", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(True)

# 🔥 Annotate points (professional look)
for i, acc in enumerate(round_accuracies):
    plt.text(i + 1, acc, f"{acc:.2f}", ha='center', fontsize=8)

plt.tight_layout()

# 🔥 Save inside static folder (VERY IMPORTANT for UI)
plt.savefig(graph_path)

plt.show()