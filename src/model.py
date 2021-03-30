import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
    Set2Set,
)
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add

NUM_ATOM_TYPES = 120  # including the extra mask tokens
NUM_CHIRALITY_TAGS = 3

NUM_BOND_TYPES = 6  # including aromatic and self-loop edge, and extra masked tokens
NUM_BOND_DIRECTIONS = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPES, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTIONS, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        edge_index = edge_index.to(torch.long)

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class NodeEncoder(nn.Module):
    def __init__(self, num_layers, emb_dim):
        super(NodeEncoder, self).__init__()
        self.num_layers = num_layers

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPES, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAGS, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.layers = nn.ModuleList([GINConv(emb_dim, aggr="add") for _ in range(num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers - 1)])

    def forward(self, x, edge_index, edge_attr):
        out = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer_idx in range(self.num_layers):
            out = self.layers[layer_idx](out, edge_index, edge_attr)
            if layer_idx < self.num_layers - 1:
                out = self.batch_norms[layer_idx](out)
                out = F.relu(out)

        return out


class GraphEncoder(NodeEncoder):
    def __init__(self, num_layers=5, emb_dim=256):
        super(GraphEncoder, self).__init__(num_layers, emb_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        out = super(GraphEncoder, self).forward(x, edge_index, edge_attr)
        out = global_mean_pool(out, batch)

        return out


class SubGraphEncoder(NodeEncoder):
    def __init__(self, num_layers=5, emb_dim=256):
        super(SubGraphEncoder, self).__init__(num_layers-1, emb_dim)
        self.last_mlp = nn.Linear(emb_dim, 2 * emb_dim)
        self.last_batch_norm = nn.BatchNorm1d(emb_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        out = super(SubGraphEncoder, self).forward(x, edge_index, edge_attr)
        out = self.last_batch_norm(out)
        out = F.relu(out)
        out = self.last_mlp(out)
        out = global_mean_pool(out, batch)
        mu, logvar = torch.chunk(out, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class NewSubGraphDecoder(nn.Module):
    def __init__(self, num_layers=5, emb_dim=256):
        super(NewSubGraphDecoder, self).__init__()
        self.num_layers = num_layers

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPES, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAGS, emb_dim)
        self.z_embedding = torch.nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            )
            
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.layers = nn.ModuleList([GINConv(emb_dim, aggr="add") for _ in range(num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers - 1)])
        self.classifier = nn.Linear(emb_dim, 1)
        
    def forward(self, z, x, edge_index, edge_attr, batch_num_nodes):
        z_emb = self.z_embedding(z)
        z_emb = torch.repeat_interleave(z_emb, batch_num_nodes, dim=0)
        
        out = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1]) + z_emb
        
        for layer_idx in range(self.num_layers):
            out = self.layers[layer_idx](out, edge_index, edge_attr)
            if layer_idx < self.num_layers - 1:
                out = self.batch_norms[layer_idx](out)
                out = F.relu(out)
                
        out = self.classifier(out)

        return out

class SubGraphDecoder(nn.Module):
    def __init__(self, num_layers=5, emb_dim=256):
        super(SubGraphDecoder, self).__init__()
        self.node_encoder = NodeEncoder(num_layers=num_layers, emb_dim=emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2*emb_dim, 2*emb_dim),
            nn.BatchNorm1d(2*emb_dim),
            nn.ReLU(),
            nn.Linear(2*emb_dim, 1),
        )

    def forward(self, z, x, edge_index, edge_attr, batch_num_nodes):
        out0 = self.node_encoder(x, edge_index, edge_attr)
        #out1 = self.projector(z)
        #out1 = torch.repeat_interleave(out1, batch_num_nodes, dim=0)
        #out = F.cosine_similarity(out0, out1, dim=1)
        out1 = torch.repeat_interleave(z, batch_num_nodes, dim=0)
        out = torch.cat([out0, out1], dim=1)
        out = self.classifier(out)

        return out

if __name__ == "__main__":
    pass
