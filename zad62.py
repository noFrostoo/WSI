#!/usr/bin/env python3

from io import BytesIO
import math
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, MSELoss, Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(1)

X = torch.unsqueeze(torch.linspace(-math.pi, math.pi, 10000), dim=1)
Y = torch.sin(X * math.sqrt(6)) + torch.cos(X * math.sqrt(8))

X, Y = Variable(X), Variable(Y)

# class Network(Module):
#     def __init__(self, input=1, hidden=200, output=1):
#         super().__init__()
#         self.predictor = Sequential(
#             Linear(input, hidden),
#             ReLU(),
#             Linear(hidden, output)
#         )

#     def forward(self, x):
#         return self.predictor(x)


class Network(Module):
    def __init__(self, input=1, hidden=200, output=1):
        super().__init__()
        self.firstLayer = Linear(input, hidden)
        self.secondLayer = Linear(hidden, output)
        self.loss = ReLU()

    def forward(self, x):
        x = self.loss(self.firstLayer(x))
        x = self.secondLayer(x)
        return x



net = Network(hidden=300)
optimizer = SGD(net.parameters(), lr=0.01)
loss_func = MSELoss()

learning_rate = 0.01

loader = DataLoader(
    TensorDataset(X, Y),
    batch_size=64,
    shuffle=True,
    num_workers=4
)

images = []
fig, ax = plt.subplots()

for i in range(50):
    for x, y in loader:
        x, y = Variable(x), Variable(y)

        prediction = net(x)
        loss = loss_func(prediction, y)


        # net.zero_grad()
        
        # loss.backward()
        
        # with torch.no_grad():
        #     for param in net.parameters():
        #         param -= learning_rate * param.grad

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.cla()
    plt.plot(X.data.numpy(), Y.data.numpy(), color='orange')
    plt.plot(X.data.numpy(), net(X).data.numpy())
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    images.append(im)

plt.plot(X.data.numpy(), Y.data.numpy(), color='orange')
plt.plot(X.data.numpy(), net(X).data.numpy())
plt.show()

images[0].save('zad6.gif', save_all=True, append_images=images[1:], optimize=False, duration=7)