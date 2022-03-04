import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        #Initialize our layers and their inputs/outputs
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        #ReLu Activation function

    def forward(self, x):
        #First Layer takes in x
        out = self.l1(x)
        out = self.relu(out)
        #Second layer, takes out (above from first layer) as input, and outputs a new update of out 
        out = self.l2(out)
        out = self.relu(out)
        #Third layer
        out = self.l3(out)
        #No activation function here
        return out

class AdvancedNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AdvancedNeuralNet, self).__init__()
        # making layers
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # must have the batch as the first dimension
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # x -> (batch_size, seq_len, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, num_layers):
        self.num_layers = num_layers        
        # initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)   
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)   
              
        out, _= self.lstm(x, (h0, c0))
        # batch size, seq_len, hidden_size
        out = out[:, -1, :]
        out = self.fc(out)
        return out