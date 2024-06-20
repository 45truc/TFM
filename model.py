from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch_geometric.nn.conv as conv
import torch_geometric
import torch.nn as nn
import torch
from tqdm import tqdm

'''
    This first model only performs time agregations throguh 1D conv.
    and max poolings, with the posibility of adding internal repre-
    sentations.
'''
class TimeAggNet(nn.Module):
    def __init__(self, device, num_nodes, T, internal_rep=None):
        super(TimeAggNet, self).__init__()
        self.time_convs = nn.ModuleList()
        self.T = T
        self.n_nodes = num_nodes
        self.irep = internal_rep
        
        if self.irep==None:
             self.irep=[1,1,1]
            
        self.time_convs.append(nn.Conv1d(in_channels=1, out_channels=self.irep[0],
                                         kernel_size=10, stride=8, padding=1))
        self.time_convs.append(nn.MaxPool1d(kernel_size=5, stride=None, 
                                            padding=0, dilation=1))
        self.time_convs.append(nn.Conv1d(in_channels=self.irep[0], out_channels=self.irep[1],
                                         kernel_size=3, stride=3, padding=0))
        self.time_convs.append(nn.MaxPool1d(kernel_size=3, stride=None, 
                                            padding=1, dilation=1))
        self.time_convs.append(nn.Conv1d(in_channels=self.irep[1], out_channels=self.irep[2],
                                         kernel_size=4, padding=0))
        
        self.class_layer = torch.nn.Linear(in_features= self.n_nodes*self.irep[2], out_features=3)            

    def forward(self, input):
        # input.shape [bs, n_nodes, 1200 (time_steps)]
        x = input
        bs = input.size(0)
        
        #1D array of n_nodes channels
        #Time agregation convolution
        t_layer_out = [x]
        for i in range(len(self.time_convs)):
            nodes=[]
            for j in range(self.n_nodes):
                # Internally we are reshaping to the number of canels (automatically infered with -1)
                nodes.append(self.time_convs[i](t_layer_out[i][:,j].view(bs,-1,t_layer_out[i].size(-1))))   
            t_layer_out.append(torch.relu(torch.stack(nodes, axis=1)))
        
        feat = t_layer_out[-1].view(-1,self.n_nodes*self.irep[2])
            
        #Dense layer for out 3 dim (classifier)
        out = self.class_layer(feat)
        out =  nn.Softmax(dim=1)(out) #  torch.sigmoid(out) #For now probabilities
        return out

'''
    This model performs two graph convolutions with the option to
    have adaptable adjacency matrices.
'''
class TimeGraphNet(nn.Module):
    def __init__(self, device, num_nodes, T, adadj=False):
        super(TimeGraphNet, self).__init__()
        self.time_convs = nn.ModuleList()
        self.T = T
        self.n_nodes = num_nodes
        self.adadj = adadj
            
        self.time_convs.append(nn.Conv1d(in_channels=1, out_channels=1,
                                         kernel_size=10, stride=8, padding=1))
        self.time_convs.append(nn.MaxPool1d(kernel_size=5, stride=None, 
                                            padding=0, dilation=1))
        self.time_convs.append(nn.Conv1d(in_channels=1, out_channels=1,
                                         kernel_size=3, stride=3, padding=0))
        self.time_convs.append(nn.MaxPool1d(kernel_size=3, stride=None, 
                                            padding=1, dilation=1))
        self.time_convs.append(nn.Conv1d(in_channels=1, out_channels=1,
                                         kernel_size=4, padding=0))
        #Graph Conv.
        if self.adadj:
            self.gcn1 = AdaptiveGCNLayer(30,30)
            self.gcn2 = AdaptiveGCNLayer(4,4)
        else:
            A = torch.ones(self.n_nodes,self.n_nodes)
            self.A = A.to_sparse()
            self.gcn1 = conv.GCNConv(30,30)
            self.gcn2 = conv.GCNConv(4,4)
        
        self.class_layer = torch.nn.Linear(in_features= self.n_nodes, out_features=3)            

    def forward(self, input):
        # input.shape [bs, n_nodes, 1200 (time_steps)]
        x = input
        bs = input.size(0)
        
        #1D array of n_nodes channels
        #Time agregation convolution
        t_layer_out = [x]
        for i in range(len(self.time_convs)):
            nodes=[]
            for j in range(self.n_nodes):
                # Internally we are reshaping to the number of canels (automatically infered with -1)
                nodes.append(self.time_convs[i](t_layer_out[i][:,j].view(bs,-1,t_layer_out[i].size(-1))))   
            t_layer_out.append(torch.relu(torch.stack(nodes, axis=1)))

            if i==2:
                batches=[]
                for b in range(bs):
                    batches.append(t_layer_out[i].view(bs,self.n_nodes,-1)[b])
                    if self.adadj:
                        batches[b] = self.gcn1(batches[b])
                    else:
                        batches[b] = self.gcn1(batches[b],self.A)
                t_layer_out[i] = torch.stack(batches, axis=1).view(bs,self.n_nodes,1,-1)
                batches=[]
                
            if i==4:
                batches=[]
                for b in range(bs):
                    batches.append(t_layer_out[i].view(bs,self.n_nodes,-1)[b])
                    if self.adadj:
                        batches[b] = self.gcn2(batches[b])
                    else:
                        batches[b] = self.gcn2(batches[b],self.A)
                t_layer_out[i] = torch.stack(batches, axis=1).view(bs,self.n_nodes,1,-1)
                batches=[]
            
        feat = t_layer_out[-1].view(-1,self.n_nodes)
            
        #Dense layer for out 3 dim (classifier)
        out = self.class_layer(feat)
        out =  nn.Softmax(dim=1)(out) #  torch.sigmoid(out) #For now probabilities
        return out


'''
    This model performs two graph convolution with the option to
    have adaptable adjacency matrices. But is deeper with smaller
    kernels.
'''
class DeepTimeGraphNet(nn.Module):
    def __init__(self, device, num_nodes, T, adadj=False):
        super(DeepTimeGraphNet, self).__init__()
        self.time_convs = nn.ModuleList()
        self.T = T
        self.n_nodes = num_nodes
        self.adadj = adadj

            
        self.time_convs.append(nn.Conv1d(in_channels=1, out_channels=1,
                                         kernel_size=2, stride=2, padding=0))
        self.time_convs.append(nn.MaxPool1d(kernel_size=3, stride=None, 
                                            padding=0, dilation=1))
        self.time_convs.append(nn.Conv1d(in_channels=1, out_channels=1,
                                         kernel_size=4, stride=2, padding=1))
        self.time_convs.append(nn.MaxPool1d(kernel_size=2, stride=None, 
                                            padding=0, dilation=1))
        self.time_convs.append(nn.Conv1d(in_channels=1, out_channels=1,
                                         kernel_size=4, stride=2, padding=1))
        self.time_convs.append(nn.MaxPool1d(kernel_size=2, stride=None,
                                            padding=0, dilation=1))
        self.time_convs.append(nn.Conv1d(in_channels=1, out_channels=1, 
                                         kernel_size=4, stride=2, padding=1))
        self.time_convs.append(nn.MaxPool1d(kernel_size=2, stride=None, 
                                            padding=0, dilation=1))
        self.time_convs.append(nn.Conv1d(in_channels=1, out_channels=1,
                                         kernel_size=3))
        #Graph Conv.
        if self.adadj:
            self.gcn1 = AdaptiveGCNLayer(50,50)
            self.gcn2 = AdaptiveGCNLayer(12,12)
        else:
            A = torch.ones(self.n_nodes,self.n_nodes)
            self.A = A.to_sparse()
            self.gcn1 = conv.GCNConv(50,50)
            self.gcn2 = conv.GCNConv(12,12)
        
        self.class_layer = torch.nn.Linear(in_features= self.n_nodes, out_features=3)            

    def forward(self, input):
        # input.shape [bs, n_nodes, 1200 (time_steps)]
        x = input
        bs = input.size(0)
        
        #1D array of n_nodes channels
        #Time agregation convolution
        t_layer_out = [x]
        for i in range(len(self.time_convs)):
            nodes=[]
            for j in range(self.n_nodes):
                # Internally we are reshaping to the number of canels (automatically infered with -1)
                nodes.append(self.time_convs[i](t_layer_out[i][:,j].view(bs,-1,t_layer_out[i].size(-1))))
                
            if i%2==1:
                t_layer_out.append(torch.relu(torch.stack(nodes, axis=1)))
            else:
                t_layer_out.append(torch.stack(nodes, axis=1))

            if i==4:
                batches=[]
                for b in range(bs):
                    batches.append(t_layer_out[i].view(bs,self.n_nodes,-1)[b])
                    if self.adadj:
                        batches[b] = self.gcn1(batches[b])
                    else:
                        batches[b] = self.gcn1(batches[b],self.A)
                t_layer_out[i] = torch.stack(batches, axis=1).view(bs,self.n_nodes,1,-1)
                batches=[]
                
            if i==6:
                batches=[]
                for b in range(bs):
                    batches.append(t_layer_out[i].view(bs,self.n_nodes,-1)[b])
                    if self.adadj:
                        batches[b] = self.gcn2(batches[b])
                    else:
                        batches[b] = self.gcn2(batches[b],self.A)
                t_layer_out[i] = torch.stack(batches, axis=1).view(bs,self.n_nodes,1,-1)
                batches=[]
            
        feat = t_layer_out[-1].view(-1,self.n_nodes)
            
        #Dense layer for out 3 dim (classifier)
        out = self.class_layer(feat)
        out =  nn.Softmax(dim=1)(out) #  torch.sigmoid(out) #For now probabilities
        return out


'''
    This model performs one graph convolution with the option to
    have adaptable adjacency matrices.
'''
class SimpleTimeGraphNet(nn.Module):
    def __init__(self, device, num_nodes, T, adadj=False):
        super(SimpleTimeGraphNet, self).__init__()
        self.time_convs = nn.ModuleList()
        self.T = T
        self.n_nodes = num_nodes
        self.adadj = adadj

            
        self.time_convs.append(nn.Conv1d(in_channels=1, out_channels=1,
                                         kernel_size=10, stride=8, padding=1))
        self.time_convs.append(nn.MaxPool1d(kernel_size=5, stride=None, 
                                            padding=0, dilation=1))
        self.time_convs.append(nn.Conv1d(in_channels=1, out_channels=1,
                                         kernel_size=3, stride=3, padding=0))
        self.time_convs.append(nn.MaxPool1d(kernel_size=3, stride=None, 
                                            padding=1, dilation=1))
        self.time_convs.append(nn.Conv1d(in_channels=1, out_channels=1,
                                         kernel_size=4, padding=0))
        #Graph Conv.
        if self.adadj:
            self.gcn1 = AdaptiveGCNLayer(4,4)
        else:
            A = torch.ones(self.n_nodes,self.n_nodes)
            self.A = A.to_sparse()
            self.gcn1 = conv.GCNConv(4,4)
        
        self.class_layer = torch.nn.Linear(in_features= self.n_nodes, out_features=3)            

    def forward(self, input):
        # input.shape [bs, n_nodes, 1200 (time_steps)]
        x = input
        bs = input.size(0)
        
        #1D array of n_nodes channels
        #Time agregation convolution
        t_layer_out = [x]
        for i in range(len(self.time_convs)):
            nodes=[]
            for j in range(self.n_nodes):
                # Internally we are reshaping to the number of canels (automatically infered with -1)
                nodes.append(self.time_convs[i](t_layer_out[i][:,j].view(bs,-1,t_layer_out[i].size(-1))))   
            t_layer_out.append(torch.relu(torch.stack(nodes, axis=1)))

            if i==4:
                batches=[]
                for b in range(bs):
                    batches.append(t_layer_out[i].view(bs,self.n_nodes,-1)[b])
                    if self.adadj:
                        batches[b] = self.gcn1(batches[b])
                    else:
                        batches[b] = self.gcn1(batches[b],self.A)
                t_layer_out[i] = torch.stack(batches, axis=1).view(bs,self.n_nodes,1,-1)
                batches=[]
            
        feat = t_layer_out[-1].view(-1,self.n_nodes)
            
        #Dense layer for out 3 dim (classifier)
        out = self.class_layer(feat)
        out =  nn.Softmax(dim=1)(out) #  torch.sigmoid(out) #For now probabilities
        return out


'''
    I found this layer but could not
    trace back the authorship of it
'''
class AdaptiveGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveGCNLayer, self).__init__()
        self.gcn =  conv.GCNConv(in_channels, out_channels)
        self.adj_weight = nn.Parameter(torch.FloatTensor(in_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.adj_weight)

    def forward(self, x):
        # Compute the adaptive adjacency matrix
        adj = torch.matmul(x, self.adj_weight)
        adj = torch.matmul(adj, x.transpose(1, 0))

        # Add self-loops to the adjacency matrix
        adj = adj + torch.eye(adj.size(0), device=adj.device)

        # Normalize the adjacency matrix
        row_sum = adj.sum(1)
        d_inv_sqrt = row_sum.pow(-0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj = torch.matmul(d_mat_inv_sqrt, adj)
        adj = torch.matmul(adj, d_mat_inv_sqrt)

        # Use the normalized adjacency matrix in the graph convolution
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        return self.gcn(x, edge_index)