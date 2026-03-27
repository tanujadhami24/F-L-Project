async function startTraining() {
    const status = document.getElementById("status");
    const loader = document.getElementById("loader");

    status.innerText = "Training Started...";
    loader.style.display = "block";

    try {
        await fetch('/start');

        // 🔥 WAIT UNTIL FILE IS READY
        let data = null;

        let attempts = 0;

        while (attempts < 20) {
    const res = await fetch('/results');
    const data = await res.json();

    if (data.status === "done") {
        break;
    }

    await new Promise(r => setTimeout(r, 2000));
    attempts++;
}

        while (true) {
            const res = await fetch('/results');
            data = await res.json();

            if (data.federated && data.federated.accuracy > 0) {
                break; // ✅ training completed
            }

            await new Promise(r => setTimeout(r, 2000)); // wait 2 sec
        }

        // =========================
        // UPDATE UI
        // =========================

        // 🔐 Privacy status (NEW LINE)
        document.getElementById("privacy-status").innerText = data.privacy;

        // Federated metrics
        document.getElementById("f-accuracy").innerText = data.federated.accuracy.toFixed(2);
        document.getElementById("f-precision").innerText = data.federated.precision.toFixed(2);
        document.getElementById("f-recall").innerText = data.federated.recall.toFixed(2);
        document.getElementById("f-f1").innerText = data.federated.f1_score.toFixed(2);
        document.getElementById("f-time").innerText = data.federated.training_time.toFixed(2);

        // Centralized metrics
        document.getElementById("c-acc").innerText = data.centralized.accuracy.toFixed(2);
        document.getElementById("c-pre").innerText = data.centralized.precision.toFixed(2);
        document.getElementById("c-rec").innerText = data.centralized.recall.toFixed(2);
        document.getElementById("c-f1").innerText = data.centralized.f1_score.toFixed(2);
        document.getElementById("c-time").innerText = data.centralized.training_time.toFixed(2);

        // Table values
        document.getElementById("f-acc").innerText = data.federated.accuracy.toFixed(2);
        document.getElementById("f-pre").innerText = data.federated.precision.toFixed(2);
        document.getElementById("f-rec").innerText = data.federated.recall.toFixed(2);
        document.getElementById("f-f1-table").innerText = data.federated.f1_score.toFixed(2);
        document.getElementById("f-time-table").innerText = data.federated.training_time.toFixed(2);

        // Final status
        status.innerText = "Training Completed ✅";
        loader.style.display = "none";

        // Refresh graph
        document.getElementById("graph").src =
            "/static/federated_accuracy_plot.png?t=" + new Date().getTime();

    } catch (error) {
        console.error(error);
        status.innerText = "Error occurred ❌";
        loader.style.display = "none";
    }
}