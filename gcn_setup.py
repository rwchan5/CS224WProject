import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim):
        super(FlightFareGNN, self).__init__()
        self.node_conv1 = GCNConv(input_dim, hidden_dim)
        self.edge_conv1 = EdgeConv(lambda x, edge_attr: x + edge_attr, aggr='mean')  # edge aggregation
        self.fc = torch.nn.Linear(hidden_dim + edge_dim, output_dim)  

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # node level feature propagation
        x = self.node_conv1(x, edge_index)
        x = F.relu(x)

        # edge level feature propagation
        edge_attr = self.edge_conv1((x, x), edge_index)

        # concatenate node and edge features (need to play with this in future)
        x_combined = torch.cat([x, edge_attr], dim=1)
        out = self.fc(x_combined)
        
        return out

def prepare_data(graph):

    node_features = {}

    # calculate aggregated features for each node
    for node in graph.nodes:
        connected_edges = list(graph.edges(node, data=True))

        # aggregate features for each node based on connected edges
        total_passengers = sum(edge_data['passengers'] for _, _, edge_data in connected_edges)
        total_miles = sum(edge_data['miles'] for _, _, edge_data in connected_edges)
        num_flights = len(connected_edges)

        avg_passengers = total_passengers / num_flights if num_flights > 0 else 0
        avg_miles = total_miles / num_flights if num_flights > 0 else 0

        node_features[node] = [avg_passengers, avg_miles]

    x = torch.tensor([node_features[n] for n in graph.nodes], dtype=torch.float)

    # edge index and edge attributes
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    edge_attrs = torch.tensor([[graph.edges[u, v]['miles'], graph.edges[u, v]['passengers']] for u, v in graph.edges], dtype=torch.float)

    # labels for edge level prediction (fares)
    edge_labels = torch.tensor([graph.edges[u, v]['fare_lg'] for u, v in graph.edges], dtype=torch.float).view(-1, 1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attrs, y=edge_labels)

    return data

def train_model(data, input_dim, hidden_dim=16, output_dim=1, epochs=100, learning_rate=0.01):
    model = FlightFareGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    return model

def main():
    from graph_setup import GraphSetup
    graph_setup = GraphSetup("data.csv")
    graph_setup.train() 

    data = prepare_data(graph_setup.graph)

    input_dim = data.x.shape[1]

    model = train_model(data, input_dim)

    torch.save(model.state_dict(), "flight_fare_gnn.pth")
    print("Model training complete and saved to flight_fare_gnn.pth")

if __name__ == "__main__":
    main()
