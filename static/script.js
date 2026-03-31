let chart;
let accuracyData = [];
let labels = [];

// ✅ INIT CHART
function initChart() {
    const ctx = document.getElementById("chart").getContext("2d");

    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Federated Accuracy",
                data: accuracyData,
                borderWidth: 2
            }]
        }
    });
}

// 🚀 START TRAINING
async function startTraining() {
    const status = document.getElementById("status");
    const loader = document.getElementById("loader");

    status.innerText = "Training Started...";
    loader.style.display = "block";

    try {
        const res = await fetch('/start');
        if (!res.ok) throw new Error("Start failed");

        fetchUpdates();

    } catch (error) {
        console.error(error);
        status.innerText = "Error occurred ❌";
        loader.style.display = "none";
    }
}

// 🔄 FETCH LIVE UPDATES
function fetchUpdates() {
    let interval = setInterval(async () => {
        try {
            const res = await fetch('/results');
            const data = await res.json();

            if (!data) return;

            // 🔐 Privacy
            document.getElementById("privacy-status").innerText = data.privacy;

            // ✅ Federated metrics
            document.getElementById("accuracy").innerText = data.accuracy.toFixed(3);
            document.getElementById("precision").innerText = data.precision.toFixed(3);
            document.getElementById("recall").innerText = data.recall.toFixed(3);
            document.getElementById("f1").innerText = data.f1.toFixed(3);
            document.getElementById("time").innerText = data.time.toFixed(2);

            // ✅ Centralized metrics
            document.getElementById("c-acc").innerText = data.c_accuracy.toFixed(3);
            document.getElementById("c-pre").innerText = data.c_precision.toFixed(3);
            document.getElementById("c-rec").innerText = data.c_recall.toFixed(3);
            document.getElementById("c-f1").innerText = data.c_f1.toFixed(3);
            document.getElementById("c-time").innerText = data.c_time.toFixed(2);

            // ✅ Table update
            document.getElementById("f-acc").innerText = data.accuracy.toFixed(3);
            document.getElementById("f-pre").innerText = data.precision.toFixed(3);
            document.getElementById("f-rec").innerText = data.recall.toFixed(3);
            document.getElementById("f-f1-table").innerText = data.f1.toFixed(3);
            document.getElementById("f-time-table").innerText = data.time.toFixed(2);

            // 📈 Graph update
            labels.push(labels.length + 1);
            accuracyData.push(data.accuracy);
            chart.update();

            // ✅ Stop when done
            if (data.status === "done") {
                clearInterval(interval);
                document.getElementById("status").innerText = "Training Completed ✅";
                document.getElementById("loader").style.display = "none";
            }

        } catch (error) {
            console.error(error);
            document.getElementById("status").innerText = "Error occurred ❌";
            document.getElementById("loader").style.display = "none";
        }

    }, 2000);
}

// INIT
initChart();