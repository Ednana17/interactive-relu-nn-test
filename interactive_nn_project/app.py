from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import plotly.express as px

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def train_network(layers, nodes, train_split):
    df = pd.read_excel("uploads/star.xlsx")  # 读取数据
    X = df[['B-V', 'Amag']].values
    y = df['TargetClass'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 训练集比例
    train_size = int(len(X) * (int(train_split) / 100))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)

    # 训练 MLP
    mlp_relu = MLPClassifier(hidden_layer_sizes=(int(nodes),) * int(layers), activation='relu', max_iter=1000, random_state=42)
    mlp_relu.fit(X_train, y_train)

    # 计算隐藏层输出
    hidden_layer_output = np.maximum(0, X_train @ mlp_relu.coefs_[0] + mlp_relu.intercepts_[0])
    hidden_df = pd.DataFrame(hidden_layer_output, columns=[f"Node {i+1}" for i in range(int(nodes))])
    hidden_df["TargetClass"] = y_train

    fig = px.scatter_3d(hidden_df, x="Node 1", y="Node 2", z="Node 3", color=hidden_df["TargetClass"].astype(str))
    plot_path = os.path.join("static", "hidden_layer_3d.html")
    fig.write_html(plot_path)

    return plot_path

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    data = request.json
    layers = data["layers"]
    nodes = data["nodes"]
    train_split = data["train_split"]

    plot_url = train_network(layers, nodes, train_split)
    return jsonify({"plot_url": plot_url})

if __name__ == "__main__":
    app.run(debug=True)

