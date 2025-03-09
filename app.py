from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import plotly.express as px

app = Flask(__name__, template_folder="templates", static_folder="static")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def train_network(data_path, x_features, y_feature, layers, nodes):
    df = pd.read_csv(data_path)
    X = df[x_features].values
    y = df[y_feature].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    layer_config = tuple([nodes] * layers)
    nn_relu = MLPClassifier(hidden_layer_sizes=layer_config, activation='relu', solver='adam', max_iter=500, random_state=42)
    nn_relu.fit(X_scaled, y)
    
    hidden_layer_output = np.maximum(0, X_scaled @ nn_relu.coefs_[0] + nn_relu.intercepts_[0])
    hidden_df = pd.DataFrame(hidden_layer_output, columns=[f"Node {i+1}" for i in range(nodes)])
    hidden_df[y_feature] = y
    
    fig = px.scatter_3d(hidden_df, x="Node 1", y="Node 2", z="Node 3", 
                        color=hidden_df[y_feature].astype(str), 
                        title="3D Visualization of First Hidden Layer Output",
                        labels={y_feature: "Class"})
    fig_path = os.path.join("static", "hidden_layer_3d.html")
    fig.write_html(fig_path)
    
    return fig_path

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    df = pd.read_csv(file_path)
    return jsonify({"columns": list(df.columns), "file_path": file_path})

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    file_path = data['file_path']
    x_features = data['x_features']
    y_feature = data['y_feature']
    layers = int(data['layers'])
    nodes = int(data['nodes'])
    
    fig_path = train_network(file_path, x_features, y_feature, layers, nodes)
    return jsonify({"plot_url": fig_path})

if __name__ == '__main__':
    app.run(debug=True)

