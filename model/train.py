# model/train.py
import torch
import torch.optim as optim
from model import Network, loss_first_order, loss_second_order

def train_first_order(model, optimizer, x):
    print("Training first-order ODE...")
    def closure():
        optimizer.zero_grad()
        l = loss_first_order(x, model, torch.exp)  
        l.backward()
        return l

    for i in range(10):  
        optimizer.step(closure)
        if i % 2 == 0:  
            print(f"Epoch {i}, Loss: {loss_first_order(x, model, torch.exp).item():.4f}")
    torch.save(model.state_dict(), 'model_first_order.pth')  # Save first-order model
    print("First-order model trained and saved as model_first_order.pth")

def train_second_order(model, optimizer, x):
    print("Training second-order ODE...")
    def closure():
        optimizer.zero_grad()
        l = loss_second_order(x, model, lambda x: -torch.ones(x.shape[0], x.shape[1]))  
        l.backward()
        return l

    for i in range(50):  
        optimizer.step(closure)
        if i % 2 == 0:  
            print(f"Epoch {i}, Loss: {loss_second_order(x, model, lambda x: -torch.ones(x.shape[0], x.shape[1])).item():.4f}")
    torch.save(model.state_dict(), 'model_second_order.pth')  # Save second-order model
    print("Second-order model trained and saved as model_second_order.pth")

if __name__ == '__main__':
    model = Network()
    optimizer = optim.LBFGS(model.parameters())
    x = torch.linspace(0, 1, 100)[:, None] 

    print("Starting training...")
    train_first_order(model, optimizer, x)  # First order training
    train_second_order(model, optimizer, x)  # Uncomment for second order training
