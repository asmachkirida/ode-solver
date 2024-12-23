import torch
import torch.nn as nn

# Define the neural network model 
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(1, 10)
        self.output_layer = nn.Linear(10, 1)

    def forward(self, x):
        layer_out = torch.sigmoid(self.hidden_layer(x))
        output = self.output_layer(layer_out)
        return output

def load_model(model_path='model.pth'):
    model = Network()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to compute loss 1st ODE
def loss_first_order(x, model, f):
    x.requires_grad = True
    y = model(x)
    dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return torch.mean((dy_dx - f(x)) ** 2)

# for 2nd ODE
def loss_second_order(x, model, f):
    x.requires_grad = True
    y = model(x)
    dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    y_double_prime = torch.autograd.grad(dy_dx.sum(), x, create_graph=True)[0]
    return torch.mean((y_double_prime - f(x)) ** 2)
