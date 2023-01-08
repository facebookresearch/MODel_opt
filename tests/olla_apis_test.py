
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
import unittest

import olla

class OLLAAPIsTest(unittest.TestCase):
    def testTrivialEval(self):
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn((1,)), requires_grad=True)

            def forward(self, x):
                return x + self.param + 5

        model = SimpleModule()
        input = torch.randn((1), requires_grad=False)

        model.eval()
        model_opt = olla.optimize(model, input)

        y_orig = model(input)
        y = model_opt(input)
        assert(torch.allclose(y, y_orig))

        y2 = model_opt(input)
        # since this is inference mode, everytime we run the model with same input, we should get same output
        assert(torch.allclose(y, y2))

    def testTrivialTrain(self):
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn((1,)), requires_grad=True)

            def forward(self, x):
                return x + self.param + 5

        model = SimpleModule()
        input = torch.randn((1), requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.MSELoss()

        model.train()
        model_opt = olla.optimize(model, input, optimizer=optimizer, loss_fn=loss_fn)

        y_orig = model(input)
        y = model_opt(input)
        assert(torch.allclose(y, y_orig))

        y2 = model_opt(input)
        # everytime we run the model, the weights get updated, so we expect the output to be different
        assert(not torch.allclose(y, y2))

    def testSimpleEval(self):
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 5)
                self.linear2 = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear1(x) + self.linear2(x)

        model = SimpleModule()
        input = torch.randn((3, 4), requires_grad=False)

        model.eval()
        model_opt = olla.optimize(model, input)

        y_orig = model(input)
        y = model_opt(input)
        assert(torch.allclose(y, y_orig))

        y2 = model_opt(input)
        # since this is inference mode, everytime we run the model with same input, we should get same output
        assert(torch.allclose(y, y2))

    def testSimpleTrain(self):
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 5)
                self.linear2 = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear1(x) + self.linear2(x)

        model = SimpleModule()
        input = torch.randn((3, 4), requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.MSELoss()

        model.train()
        model_opt = olla.optimize(model, input, optimizer=optimizer, loss_fn=loss_fn)

        y_orig = model(input)
        y = model_opt(input)
        assert(torch.allclose(y, y_orig))

        y2 = model_opt(input)
        # everytime we run the model, the weights get updated, so we expect the output to be different
        assert(not torch.allclose(y, y2))

    def testAlexNetEval(self):
        model = torchvision.models.alexnet()
        input = torch.randn((1, 3, 224, 224), requires_grad=False)

        model.eval()
        model_opt = olla.optimize(model, input)

        y_orig = model(input)
        y = model_opt(input)
        assert(torch.allclose(y, y_orig))

        y2_orig = model(input)
        y2 = model_opt(input)
        # since this is inference mode, everytime we run the model with same input, we should get same output
        assert(torch.allclose(y2_orig, y_orig))
        assert(torch.allclose(y, y2))

    def testResNet18Train(self):
        model = torchvision.models.resnet18()
        input = torch.randn((32, 3, 224, 224), requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.MSELoss()

        model.train()
        model_opt = olla.optimize(model, input, optimizer=optimizer, loss_fn=loss_fn)

        y_orig = model(input)
        y = model_opt(input)
        assert(torch.allclose(y, y_orig))

        y2 = model_opt(input)
        # everytime we run the model, the weights get updated, so we expect the output to be different
        assert(not torch.allclose(y, y2))
