import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

class FlightFareGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FlightFareGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def prepare_data(graph):
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # this isn't right yet need to change it more
    x = torch.tensor([list(graph.nodes[n].get('features', [0, 0])) for n in graph.nodes], dtype=torch.float)
    y = torch.tensor([graph.nodes[n].get('label', 0) for n in graph.nodes], dtype=torch.float).view(-1, 1)

    data = Data(x=x, edge_index=edge_index, y=y)
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
