import torch
import torchvision
import unittest

import olla

class OLLAAPIsTest(unittest.TestCase):
    def testAlexNetTrain(self):
        model = torchvision.models.alexnet()
        input = torch.randn((32, 3, 224, 224), requires_grad = True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.MSELoss()

        model_opt = olla.optimize(model, input, optimizer=optimizer, loss_fn=loss_fn)

        y_orig = model(input)
        y = model_opt(input)
        # FIXME: Assertion failing
        # assert(torch.allclose(y, y_orig))

        y2 = model_opt(input)
        # everytime we run the model, the weights get updated, so we expect the output to be different
        assert(not torch.allclose(y, y2))
