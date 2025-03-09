async function trainNetwork() {
    let pyodide = await loadPyodide();
    await pyodide.loadPackage(["numpy", "pandas", "scikit-learn"]);

    let layers = document.getElementById("num-layers").value;
    let nodes = document.getElementById("nodes-per-layer").value;
    let trainSplit = document.getElementById("train-split").value;

    // 显示 Loading 动画
    document.getElementById("loading").style.display = "block";

    // GitHub 上的 star.json 文件 URL
    let dataURL = "https://github.com/Ednana17/interactive-relu-nn-test/blob/main/star.json";

    let response = await fetch(dataURL);
    let jsonData = await response.json();

    let pythonScript = `
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import json

data = pd.DataFrame(${JSON.stringify(jsonData)})

X = data[['B-V', 'Amag']].values
y = data['TargetClass'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

train_size = int(len(X) * ${trainSplit} / 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)

mlp_relu = MLPClassifier(hidden_layer_sizes=(${nodes},) * ${layers}, activation='relu', max_iter=500, random_state=42)
mlp_relu.fit(X_train, y_train)

hidden_layer_output = np.maximum(0, X_train @ mlp_relu.coefs_[0] + mlp_relu.intercepts_[0])
output_data = [{"x": X_train[i][0], "y": X_train[i][1], "z": hidden_layer_output[i][0]} for i in range(len(X_train))]
json.dumps(output_data)
`;

    try {
        let result = await pyodide.runPythonAsync(pythonScript);
        let outputData = JSON.parse(result);

        let trace = {
            x: outputData.map(d => d.x),
            y: outputData.map(d => d.y),
            z: outputData.map(d => d.z),
            mode: "markers",
            type: "scatter3d",
            marker: { size: 5, color: outputData.map(d => d.z), colorscale: "Viridis" }
        };
        Plotly.newPlot("plot-container", [trace]);
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while processing the model.");
    }

    // 隐藏 Loading 动画
    document.getElementById("loading").style.display = "none";
}
