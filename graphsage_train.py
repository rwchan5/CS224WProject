import csv
import os
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv

class GraphSetup:
    def __init__(self, data_path):
        # Load supplementary data for geotags
        sup_df = pd.read_csv("data/airports.txt", header=None, names=["CustomID", "Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude", "Altitude", "Timezone", "DST", "Tz DB time zone", "IsAirport", "Src"])
        geotags = {row['IATA']: (row['Latitude'], row['Longitude']) for _, row in sup_df.iterrows() if pd.notna(row['IATA'])}
        
        # Load and preprocess main dataset
        df = pd.read_csv(data_path, low_memory=False)
        df_filtered = df.drop(df.columns[[0, 3, 4, 20, 21]], axis=1)
        
        # Prepare node attributes
        airport_ids = set(df_filtered['airportid_1']).union(set(df_filtered['airportid_2']))
        self.node_id_mapping = {airport_id: idx for idx, airport_id in enumerate(airport_ids)}
        node_features = []

        for airport_id in airport_ids:
            row = df_filtered[df_filtered['airportid_1'] == airport_id].iloc[0] if airport_id in df_filtered['airportid_1'].values else None
            latitude, longitude = geotags.get(row['airport_1'], (0.0, 0.0)) if row is not None else (0.0, 0.0)
            node_features.append([latitude, longitude])
        
        self.node_features = np.array(node_features, dtype=np.float32)

        # Prepare edges and labels
        edge_index = []
        edge_labels = []

        for _, row in df_filtered.iterrows():
            idx1 = self.node_id_mapping[row['airportid_1']]
            idx2 = self.node_id_mapping[row['airportid_2']]
            edge_index.append([idx1, idx2])
            edge_labels.append(row['fare'])

        self.edge_index = np.array(edge_index).T
        self.edge_labels = np.array(edge_labels, dtype=np.float32)

        # Convert to PyTorch Geometric data
        self.data = Data(
            x=torch.tensor(self.node_features, dtype=torch.float),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long)
        )

        # Split data for training and testing
        num_edges = len(edge_labels)
        indices = np.arange(num_edges)
        self.train_indices, self.test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    def train(self, num_epochs=100):
        hidden_dim = 64
        input_dim = self.data.x.shape[1]
        
        model = GraphSAGEModel(input_dim, hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        train_edge_index = self.edge_index[:, self.train_indices]
        train_edge_labels = torch.tensor(self.edge_labels[self.train_indices], dtype=torch.float)

        print("Training GraphSAGE model")
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            node_embeddings = model(self.data.x, self.data.edge_index)
            src = node_embeddings[train_edge_index[0]]
            dst = node_embeddings[train_edge_index[1]]
            train_preds = (src * dst).sum(dim=1)

            loss = criterion(train_preds, train_edge_labels)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        self.evaluate(model)

    def evaluate(self, model):
        model.eval()
        test_edge_index = self.edge_index[:, self.test_indices]
        test_edge_labels = torch.tensor(self.edge_labels[self.test_indices], dtype=torch.float)

        with torch.no_grad():
            node_embeddings = model(self.data.x, self.data.edge_index)
            src = node_embeddings[test_edge_index[0]]
            dst = node_embeddings[test_edge_index[1]]
            test_preds = (src * dst).sum(dim=1)

            loss = F.mse_loss(test_preds, test_edge_labels)
            print(f"Test Loss: {loss.item()}")

    def visualize_graph(self, num_nodes=50):
        subgraph = self.data.edge_index[:, :num_nodes]
        print("Visualization not implemented for PyTorch Geometric format")

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Run GraphSAGE
if __name__ == "__main__":
    g = GraphSetup("data/reduced.csv")
    g.train(num_epochs=100)
