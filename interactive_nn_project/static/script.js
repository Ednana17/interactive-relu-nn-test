function trainNetwork() {
    let layers = document.getElementById("num-layers").value;
    let nodes = document.getElementById("nodes-per-layer").value;
    let trainSplit = document.getElementById("train-split").value;

    fetch("/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            layers: layers,
            nodes: nodes,
            train_split: trainSplit
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("plot-container").innerHTML = `<iframe src="${data.plot_url}" width="800" height="600"></iframe>`;
    })
    .catch(error => console.error("Error:", error));
}
