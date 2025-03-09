async function trainNetwork() {
    let pyodide = await loadPyodide();
    await pyodide.loadPackage(["numpy", "pandas", "scikit-learn"]);

    let layers = document.getElementById("num-layers").value;
    let nodes = document.getElementById("nodes-per-layer").value;
    let trainSplit = document.getElementById("train-split").value;
    let maxIter = document.getElementById("max-iter").value;

    // 显示 Loading 动画
    document.getElementById("loading").style.display = "block";

    // GitHub 上的 star.json 文件 URL
    let dataURL = "https://raw.githubusercontent.com/Ednana17/interactive-relu-nn-test/main/star.json";

    try {
        let response = await fetch(dataURL);
        if (!response.ok) throw new Error("Failed to fetch JSON");

        let jsonData = await response.json();

        let pythonScript = `
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import json

# 解析 JSON 数据
data = pd.DataFrame(${JSON.stringify(jsonData)})

X = data[['B-V', 'Amag']].values
y = data['TargetClass'].values

# 归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 训练集划分
train_size = int(len(X) * ${trainSplit} / 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)

# 训练 MLP
mlp_relu = MLPClassifier(hidden_layer_sizes=(${nodes},) * ${layers}, activation='relu', max_iter=${maxIter}, random_state=42)
mlp_relu.fit(X_train, y_train)

# 存储所有隐藏层的 3D 输出
hidden_layer_outputs = []
X_transformed = X_train

for i, (W, b) in enumerate(zip(mlp_relu.coefs_, mlp_relu.intercepts_)):
    X_transformed = np.maximum(0, X_transformed @ W + b)
    layer_output = [{"x": X_transformed[j][0], "y": X_transformed[j][1], "z": X_transformed[j][2]} for j in range(len(X_transformed))]
    hidden_layer_outputs.append(layer_output)

json.dumps(hidden_layer_outputs)
`;

        let result = await pyodide.runPythonAsync(pythonScript);
        let hiddenLayerOutputs = JSON.parse(result);

        // 清空现有可视化
        document.getElementById("plot-container").innerHTML = "";

        // 为每个隐藏层生成 3D 图
        hiddenLayerOutputs.forEach((outputData, index) => {
            let trace = {
                x: outputData.map(d => d.x),
                y: outputData.map(d => d.y),
                z: outputData.map(d => d.z),
                mode: "markers",
                type: "scatter3d",
                marker: { size: 5, color: outputData.map(d => d.z), colorscale: "Viridis" }
            };

            let plotDiv = document.createElement("div");
            plotDiv.id = `plot-layer-${index + 1}`;
            document.getElementById("plot-container").appendChild(plotDiv);

            Plotly.newPlot(plotDiv, [trace], { title: `3D Visualization After Layer ${index + 1}` });
        });

    } catch (error) {
        console.error("Error:", error);
        alert("Failed to load data or process model. Please check JSON file.");
    }

    // 隐藏 Loading 动画
    document.getElementById("loading").style.display = "none";
}
