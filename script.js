async function loadPyodideAndTrain() {
    let pyodide = await loadPyodide();
    await pyodide.loadPackage(["numpy", "pandas", "scikit-learn"]);
    return pyodide;
}

let pyodideReady = loadPyodideAndTrain();

async function trainNetwork() {
    let pyodide = await pyodideReady;

    // 显示加载动画
    document.getElementById("loading").style.display = "block";
    document.getElementById("plot-container").innerHTML = "";

    let dataURL = "https://raw.githubusercontent.com/Ednana17/interactive-relu-nn-test/main/star.json";

    try {
        let response = await fetch(dataURL);
        if (!response.ok) throw new Error("Failed to fetch JSON");

        let jsonData = await response.json();

        // 获取用户输入
        let numLayers = parseInt(document.getElementById("num-layers").value);
        let nodesPerLayer = parseInt(document.getElementById("nodes-per-layer").value);
        let trainSplit = parseInt(document.getElementById("train-split").value) / 100;
        let maxIter = parseInt(document.getElementById("max-iter").value);

        // 生成 Python 代码
        let pythonScript = `
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# 加载 JSON 数据
data = pd.DataFrame(${JSON.stringify(jsonData)})

# 选择输入特征和目标变量
X = data[['B-V', 'Amag']].values  # 选择两个数值特征
y = data['TargetClass'].values    # 目标变量 (二分类)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练/测试拆分
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1 - ${trainSplit}, random_state=42, stratify=y)

# 训练 MLP 神经网络
mlp = MLPClassifier(hidden_layer_sizes=(${Array(numLayers).fill(nodesPerLayer).join(",")}), 
                    activation='relu', max_iter=${maxIter}, random_state=42)
mlp.fit(X_train, y_train)

# 计算每层的输出
layer_outputs = []
layer_input = X_scaled
for i in range(${numLayers}):
    layer_output = np.maximum(0, layer_input @ mlp.coefs_[i] + mlp.intercepts_[i])
    layer_outputs.append(layer_output.tolist())  # 转换为 JSON 可读格式
    layer_input = layer_output

# 返回 JSON 数据
{"layers": layer_outputs, "target": y.tolist()}
        `;

        // 运行 Python 代码
        let resultJson = await pyodide.runPythonAsync(pythonScript);
        let result = JSON.parse(resultJson);

        // 隐藏加载动画
        document.getElementById("loading").style.display = "none";

        let targetClasses = result.target;

        // 渲染每一层的 3D 图像
        result.layers.forEach((layerData, index) => {
            let dfLayer = layerData.map((nodeValues, i) => ({
                x: nodeValues[0], 
                y: nodeValues[1], 
                z: nodeValues[2], 
                target: targetClasses[i]  // 目标分类
            }));

            let trace = {
                x: dfLayer.map(d => d.x),
                y: dfLayer.map(d => d.y),
                z: dfLayer.map(d => d.z),
                mode: 'markers',
                marker: {
                    size: 5,
                    color: dfLayer.map(d => d.target),  // 根据类别着色
                    colorscale: 'Viridis'
                },
                type: 'scatter3d'
            };

            let layout = {
                title: `3D Visualization of Layer ${index + 1}`,
                scene: { xaxis: { title: 'Node 1' }, yaxis: { title: 'Node 2' }, zaxis: { title: 'Node 3' } }
            };

            let div = document.createElement("div");
            div.innerHTML = `<h3>Hidden Layer ${index + 1}</h3><div id="plot-${index}"></div>`;
            document.getElementById("plot-container").appendChild(div);

            Plotly.newPlot(`plot-${index}`, [trace], layout);
        });

    } catch (error) {
        console.error("Error loading or processing data:", error);
        document.getElementById("loading").style.display = "none";
    }
}


