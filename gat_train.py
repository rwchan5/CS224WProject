import csv
import os
import sklearn
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import TransformerConv

class GraphSetup:
    def __init__(self, data_path):
        # load the data and convert to lattitude/longitude
        sup_df = pd.read_csv("data/airports.txt", header=None, names=["CustomID", "Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude", "Altitude", "Timezone", "DST", "Tz DB time zone", "IsAirport", "Src"])
        geotags = {}
        for _, airport in sup_df.iterrows():
            if pd.notna(airport['IATA']) and pd.notna(airport['Latitude']) and pd.notna(airport['Longitude']):
                geotags[airport['IATA']] = (airport['Latitude'], airport['Longitude'])

        df = pd.read_csv(data_path, low_memory=False)

        df_filtered = df.drop(df.columns[[0, 3, 4, 20, 21]], axis=1)

        airport1_ids = df_filtered['airportid_1'].unique()
        airport2_ids = df_filtered['airportid_2'].unique()
        all_airport_ids = list(set(airport1_ids) | set(airport2_ids))
        self.node_id_mapping = {airport_id: idx for idx, airport_id in enumerate(all_airport_ids)}
        self.node_idx_mapping = {idx: airport_id for airport_id, idx in self.node_id_mapping.items()}

        # Build airport attributes
        airport_attributes = {}
        node_cities = []
        node_airports = []
        node_latitudes = []
        node_longitudes = []
        node_indices = []

        for airport_id in all_airport_ids:
            rows = df_filtered[(df_filtered['airportid_1'] == airport_id) | (df_filtered['airportid_2'] == airport_id)]
            if not rows.empty:
                row = rows.iloc[0]
                airport_code = row['airport_1'] if row['airportid_1'] == airport_id else row['airport_2']
                city = row['city1'] if row['airportid_1'] == airport_id else row['city2']
                latitude, longitude = geotags.get(airport_code, (0.0, 0.0))
                airport_attributes[airport_id] = {'airport': airport_code, 'city': city, 'latitude': latitude, 'longitude': longitude}

                node_cities.append(city)
                node_airports.append(airport_code)
                node_latitudes.append(latitude)
                node_longitudes.append(longitude)
                node_indices.append(self.node_id_mapping[airport_id])

        # Encode categorical node features
        city_encoder = LabelEncoder()
        airport_encoder = LabelEncoder()
        city_labels = city_encoder.fit_transform(node_cities)
        airport_labels = airport_encoder.fit_transform(node_airports)
        node_latitudes = np.array(node_latitudes, dtype=np.float32)
        node_longitudes = np.array(node_longitudes, dtype=np.float32)

        # Create node feature matrix
        self.node_features = np.vstack((city_labels, airport_labels, node_latitudes, node_longitudes)).T  # Shape: (num_nodes, 4)

        # Build the graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(airport_attributes.keys())
        nx.set_node_attributes(self.graph, airport_attributes)

        # Edge features and labels
        edge_index = []
        edge_attrs = []
        edge_labels = []

        # Encode categorical edge features
        carrier_lg_encoder = LabelEncoder()
        carrier_low_encoder = LabelEncoder()
        carrier_lg_encoder.fit(df_filtered['carrier_lg'].astype(str))
        carrier_low_encoder.fit(df_filtered['carrier_low'].astype(str))

        # Process edges
        print("Processing edges")
        for _, row in df_filtered.iterrows():
            airport1id = row['airportid_1']
            airport2id = row['airportid_2']
            idx1 = self.node_id_mapping[airport1id]
            idx2 = self.node_id_mapping[airport2id]

            # Edge features
            miles = row['nsmiles']
            passengers = row['passengers']
            carrier_lg = int(carrier_lg_encoder.transform([str(row['carrier_lg'])])[0])
            carrier_low = int(carrier_low_encoder.transform([str(row['carrier_low'])])[0])
            label = row['fare']  # Fare to predict

            # print(miles, passengers, carrier_lg, large_ms, carrier_low, lf_ms, label)

            # Exclude 'Fare' from edge features
            edge_feature = [miles, passengers, carrier_lg, carrier_low]

            # check for nan values in edge_feature
            if np.isnan(edge_feature).any():
                print(edge_feature)
                print("nan values in edge_feature")
                break

            edge_attrs.append(edge_feature)
            edge_labels.append(label)

            # check for nan values in edge_attrs
            if np.isnan(edge_attrs).any():
                print("nan values in edge_attrs")
                break

            edge_index.append([idx1, idx2])

            self.graph.add_edge(airport1id, airport2id, miles=miles, passengers=passengers,
                                carrier_lg=carrier_lg, 
                                carrier_low=carrier_low, label=label)

        # Convert to numpy arrays
        print("Converting to numpy arrays")
        self.edge_index = np.array(edge_index).T  # Shape: (2, num_edges)
        self.edge_attrs = np.array(edge_attrs, dtype=np.float32)  # Shape: (num_edges, num_edge_features)
        self.edge_labels = np.array(edge_labels, dtype=np.float32)  # Shape: (num_edges,)

        # Create PyG data object
        self.data = Data(
            x=torch.tensor(self.node_features, dtype=torch.float),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(self.edge_attrs, dtype=torch.float)
        )

        # Split edge indices into train and test sets
        num_edges = self.edge_index.shape[1]
        edge_indices = np.arange(num_edges)
        self.train_indices, self.test_indices = train_test_split(edge_indices, test_size=0.2, random_state=42)

    def train(self, num_epochs=100):
        print("Starting training")
        hidden_dim = 64
        num_node_features = self.data.num_node_features  # dim 4
        num_edge_features = self.data.edge_attr.shape[1]  # dim of 6

        self.gnn_model = GNNModel(num_node_features, num_edge_features, hidden_dim)
        self.edge_mlp = EdgeMLP(hidden_dim, num_edge_features, hidden_dim)

        optimizer = torch.optim.Adam(list(self.gnn_model.parameters()) + list(self.edge_mlp.parameters()), lr=0.01)
        criterion = torch.nn.MSELoss()

        train_edge_index = torch.tensor(self.edge_index[:, self.train_indices], dtype=torch.long)
        train_edge_attr = torch.tensor(self.edge_attrs[self.train_indices], dtype=torch.float)
        train_edge_labels = torch.tensor(self.edge_labels[self.train_indices], dtype=torch.float)

        print("beginning training loop")
        for epoch in range(num_epochs):
            self.gnn_model.train()
            self.edge_mlp.train()
            optimizer.zero_grad()

            node_embeddings = self.gnn_model(self.data.x, self.data.edge_index, self.data.edge_attr)

            train_preds = self.edge_mlp(node_embeddings, train_edge_index, train_edge_attr)

            loss = criterion(train_preds, train_edge_labels)

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        self.evaluate()

    def evaluate(self):
        self.gnn_model.eval()
        self.edge_mlp.eval()

        test_edge_index = torch.tensor(self.edge_index[:, self.test_indices], dtype=torch.long)
        test_edge_attr = torch.tensor(self.edge_attrs[self.test_indices], dtype=torch.float)
        test_edge_labels = torch.tensor(self.edge_labels[self.test_indices], dtype=torch.float)

        with torch.no_grad():
            node_embeddings = self.gnn_model(self.data.x, self.data.edge_index, self.data.edge_attr)
            test_preds = self.edge_mlp(node_embeddings, test_edge_index, test_edge_attr)
            test_loss = F.mse_loss(test_preds, test_edge_labels)
            print(f'Test Loss: {test_loss.item()}')

    def visualize_graph(self, num_nodes=50):
        plt.figure(figsize=(12, 8))
        subgraph = self.graph.subgraph(list(self.graph.nodes)[:num_nodes])
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=8)
        plt.title("Visualization of Airport Graph")
        plt.show()

class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim):
        super(GNNModel, self).__init__()
        self.conv1 = TransformerConv(num_node_features, hidden_dim, edge_dim=num_edge_features)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim, edge_dim=num_edge_features)
        self.conv3 = TransformerConv(hidden_dim, hidden_dim, edge_dim=num_edge_features)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x

class EdgeMLP(torch.nn.Module):
    def __init__(self, node_embedding_dim, edge_feature_dim, hidden_dim):
        super(EdgeMLP, self).__init__()
        input_dim = node_embedding_dim * 2 + edge_feature_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, node_embeddings, edge_index, edge_attr):
        src_embeddings = node_embeddings[edge_index[0]]
        dst_embeddings = node_embeddings[edge_index[1]]
        edge_input = torch.cat([src_embeddings, dst_embeddings, edge_attr], dim=1)
        x = F.relu(self.fc1(edge_input))
        x = self.fc2(x)
        return x.squeeze()

# Instantiate and run
print("training the model")
g = GraphSetup("data/reduced.csv")
g.train(num_epochs=100)
g.visualize_graph(num_nodes=50)
