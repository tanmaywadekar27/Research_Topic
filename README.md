# Research Topic Classification using Graph Convolutional Network (GCN)

A Graph Neural Network based research paper classification system that uses the Cora citation network dataset to classify research papers into one of seven research topics.

The project includes a trained Graph Convolutional Network (GCN), ONNX model deployment, a FastAPI backend, and an interactive Streamlit frontend.

## 📌 Project Overview

Traditional machine learning models generally classify a document using only its individual features.

In this project, research papers are represented as nodes in a graph, while citations between papers are represented as edges.

The Graph Convolutional Network uses both:

- Paper features
- Citation relationships between papers

to classify each research paper into its corresponding research topic.

### Basic Idea

Research Papers  
↓  
Cora Citation Network  
↓  
Nodes → Research Papers  
Edges → Citations  
Features → 1433-dimensional paper features  
↓  
Graph Convolutional Network  
↓  
7 Research Topics

## 🎯 Objectives

- Represent research papers as a graph.
- Use citation relationships for node classification.
- Train a Graph Convolutional Network on the Cora dataset.
- Compare the GCN with a Random Forest baseline.
- Export the trained GCN model to ONNX.
- Build a FastAPI inference API.
- Build an interactive Streamlit interface.
- Deploy the API for remote inference.

## 📊 Dataset

The project uses the Cora citation network dataset through PyTorch Geometric's `Planetoid` dataset wrapper.

The Cora dataset represents research papers as graph nodes and citations between papers as graph edges.

### Dataset Characteristics

| Property | Value |
|---|---:|
| Research papers / Nodes | 2708 |
| Features per paper | 1433 |
| Number of classes | 7 |
| Nodes | Research papers |
| Edges | Citation relationships |

Each paper contains 1433 features representing binary word/vocabulary information.

## 🏷️ Research Topics

The Cora dataset contains seven research topics.

| Class ID | Research Topic |
|---:|---|
| 0 | Case-Based |
| 1 | Genetic Algorithms |
| 2 | Neural Networks |
| 3 | Probabilistic Methods |
| 4 | Reinforcement Learning |
| 5 | Rule Learning |
| 6 | Theory |

## 🧠 Graph Neural Network

### What is a GCN?

A Graph Convolutional Network (GCN) is a type of Graph Neural Network that learns representations of nodes by combining information from the node itself and its neighboring nodes.

The basic process is:

Node Features  
↓  
Collect Information from Neighbors  
↓  
Aggregate Neighbor Information  
↓  
Apply Learnable Transformation  
↓  
Activation Function  
↓  
Updated Node Representation

Unlike a traditional machine learning model that treats every research paper independently, the GCN can use the citation structure of the research-paper network.

## 🏗️ GCN Architecture

The project uses a simple two-layer GCN.

Input  
2708 × 1433  
↓  
GCNConv  
1433 → 32  
↓  
ReLU  
↓  
Dropout  
↓  
GCNConv  
32 → 7  
↓  
Output Logits  
2708 × 7  
↓  
Predicted Research Topic

### Model Configuration

- Input Dimension: 1433
- Hidden Dimension: 32
- Output Dimension: 7
- Dropout: 0.5
- Optimizer: Adam
- Learning Rate: 0.01

The model is implemented using:

- PyTorch
- PyTorch Geometric
- GCNConv

## 🔬 GCN Message Passing

For every paper, the GCN considers information from connected papers.

Example:

Paper B  
│  
│ Citation  
▼  
Paper A ───────── Paper C  
│  
▼  
Paper D

Paper A can use information from its neighboring papers while learning its representation.

This process is repeated through the GCN layers.

## 🧪 Baseline Model

A Random Forest classifier is used as a baseline model.

The Cora binary features are transformed using a TF-IDF transformation before training the Random Forest.

Cora Features  
↓  
TF-IDF Transformation  
↓  
Random Forest  
↓  
Research Topic

The purpose of the baseline is to compare classification using individual paper features with the graph-based GCN approach.

### Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro-F1 Score
- Confusion Matrix

## 🔄 Training Process

The Cora dataset provides training, validation, and test masks.

The GCN uses the training mask to calculate the training loss.

Cora Dataset  
↓  
Node Features  
Edge Index  
Labels  
Train / Validation / Test Masks  
↓  
GCN Model  
↓  
Cross Entropy Loss  
↓  
Backpropagation  
↓  
Adam Optimizer  
↓  
Updated Parameters

The training process runs for a maximum of 200 epochs and uses early stopping based on validation accuracy.

The best validation model state is restored before final test evaluation.

## 📈 Model Evaluation

The project evaluates the trained GCN on the Cora test set.

The following metrics are calculated:

### Accuracy

Accuracy represents the fraction of correctly classified research papers.

Accuracy = Correct Predictions / Total Predictions

### Macro-F1 Score

Macro-F1 calculates the F1 score independently for each class and then takes their average.

This gives equal importance to each research topic.

### Confusion Matrix

A confusion matrix is used to visualize correct and incorrect predictions across the seven research topics.

## 📦 ONNX Model

After training, the GCN is exported to ONNX (Open Neural Network Exchange) format.

The exported model is:

`simple_gcn_cora.onnx`

The project also contains:

`simple_gcn_cora.onnx.data`

The ONNX model accepts:

- `node_features`
- `edge_indices`

and produces:

- `logits`

Dynamic axes are used to support different numbers of nodes and edges during inference.

## ⚡ ONNX Runtime

The trained model is served using ONNX Runtime.

The model is loaded when the FastAPI application starts:

```python
model_session = ort.InferenceSession(
    MODEL_PATH,
    providers=['CPUExecutionProvider']
)
