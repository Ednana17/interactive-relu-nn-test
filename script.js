async function loadPyodideAndTrain() {
    let pyodide = await loadPyodide();
    await pyodide.loadPackage(["numpy", "pandas", "scikit-learn", "plotly"]);
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
import plotly.express as px
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
    layer_outputs.append(layer_output)
    layer_input = layer_output

# 生成可视化数据
plots = []
for i, layer in enumerate(layer_outputs):
    df_layer = pd.DataFrame(layer, columns=[f"Node {j+1}" for j in range(${nodesPerLayer})])
    df_layer["TargetClass"] = y
    
    fig = px.scatter_3d(df_layer, x="Node 1", y="Node 2", z="Node 3", 
                        color=df_layer["TargetClass"].astype(str),
                        title=f"3D Visualization of Layer {i+1}",
                        labels={"TargetClass": "Class"})
    
    plots.append(fig.to_json())

plots
        `;

        // 运行 Python 代码
        let plotsJson = await pyodide.runPythonAsync(pythonScript);

        // 隐藏加载动画
        document.getElementById("loading").style.display = "none";

        // 渲染每一层的3D图像
        let plots = JSON.parse(plotsJson);
        plots.forEach((plot, index) => {
            let div = document.createElement("div");
            div.innerHTML = `<h3>Hidden Layer ${index + 1}</h3><div id="plot-${index}"></div>`;
            document.getElementById("plot-container").appendChild(div);
            Plotly.newPlot(`plot-${index}`, JSON.parse(plot).data, JSON.parse(plot).layout);
        });

    } catch (error) {
        console.error("Error loading or processing data:", error);
        document.getElementById("loading").style.display = "none";
    }
}


