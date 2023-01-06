import torch
import torchvision

from olla import apis

model = torchvision.models.alexnet()
input = torch.randn((32, 3, 224, 224), requires_grad = True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

model_opt = apis.optimize(model, input, optimizer=optimizer, loss_fn=loss_fn)

y = model_opt(input)
y_orig = model(input)
# FIXME: Assertion failing
# assert(torch.allclose(y, y_orig))
print(y)

assert(not torch.allclose(y, y2))
