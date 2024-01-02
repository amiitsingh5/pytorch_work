import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

# Generate a toy graph dataset (Zachary's Karate Club network)
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 18, 20, 22, 32]], dtype=torch.long)

x = torch.eye(edge_index.max().item() + 1, dtype=torch.float)

y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float)

data = Data(x=x, edge_index=edge_index, y=y)

# Define a simple GNN model
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, 2)  # 2 classes in the output

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the model, loss function, and optimizer
model = GNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training the GNN
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data.y.long())
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Evaluate the trained model
model.eval()
with torch.no_grad():
    output = model(data)
    predicted_class = output.argmax(dim=1)
    accuracy = (predicted_class == data.y).sum().item() / data.num_nodes
    print(f'Accuracy: {accuracy}')

# Visualize the graph with predicted classes
G = to_networkx(data)
pos = nx.spring_layout(G)
colors = predicted_class.numpy()
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.Set1, font_color='white')
plt.show()
