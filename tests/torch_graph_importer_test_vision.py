import unittest

import torch
import torch.fx
import torchvision

from olla.torch import torch_graph_importer


class TorchGraphImporterTestVision(unittest.TestCase):
    def setUp(self):
        self.importer = torch_graph_importer.TorchGraphImporter()
        self.maxDiff = None

    def run_tests(
        self,
        model,
        *inputs,
        mode="eval",
        methods=None,
        optimizer=None,
        test_name="test",
    ):
        if methods is None:
            methods = ["aotautograd"]

        for method in methods:
            print(f"Import using {method}")
            g = self.importer.import_from_torch(
                model,
                *inputs,
                mode=mode,
                method=method,
                optimizer=optimizer,
            )
            self.assertTrue(g.check_consistency())
            # g.dump(f"/tmp/{test_name}.{method}.dot")

    def testAlexNetInference(self):
        model = torchvision.models.alexnet()
        input = torch.randn((1, 3, 224, 224))

        model.eval()
        self.run_tests(
            model,
            input,
            mode="eval",
            methods=[
                "fx",
                "aotautograd",
                "functorch",
                "acc_tracer",
            ],
            test_name="alexnet.inference",
        )

    def testAlexNetTrain(self):
        model = torchvision.models.alexnet()
        input = torch.randn((1, 3, 224, 224), requires_grad=True)

        model.train()
        self.run_tests(
            model,
            input,
            mode="train",
            methods=[
                "aotautograd",
                # "functorch", # FIXME: Error message: P523547974
            ],
            test_name="alexnet.train",
        )

    def testSqueezeNetInference(self):
        model = torchvision.models.squeezenet1_0()
        input = torch.randn((1, 3, 224, 224))

        model.eval()
        self.run_tests(
            model,
            input,
            mode="eval",
            methods=[
                "fx",
                "aotautograd",
                "functorch",
                "acc_tracer",
            ],
            test_name="squeezenet1_0.inference",
        )

    def testSqueezeNetTrain(self):
        model = torchvision.models.squeezenet1_0()
        input = torch.randn((1, 3, 224, 224), requires_grad=True)

        model.train()
        self.run_tests(
            model,
            input,
            mode="train",
            methods=[
                "aotautograd",
            ],
            test_name="squeezenet1_0.train",
        )

    def testDenseNetInference(self):
        model = torchvision.models.densenet161()
        input = torch.randn((1, 3, 224, 224))

        model.eval()
        self.run_tests(
            model,
            input,
            mode="eval",
            methods=[
                # "fx", # FIXME
                "aotautograd",
                "functorch",
                "acc_tracer",
            ],
            test_name="densenet161.inference",
        )

    def testDenseNetTrain(self):
        model = torchvision.models.densenet161()
        input = torch.randn((1, 3, 224, 224), requires_grad=True)

        model.train()
        self.run_tests(
            model,
            input,
            mode="train",
            methods=[
                "aotautograd",
                # "functorch", #FIXME: Error message: P523547974
            ],
            test_name="densenet161.train",
        )

    def testInceptionInference(self):
        model = torchvision.models.inception_v3()
        input = torch.randn((1, 3, 299, 299))

        model.eval()
        self.run_tests(
            model,
            input,
            mode="eval",
            methods=[
                # "fx", # FIXME
                "aotautograd",
                "functorch",
                "acc_tracer",
            ],
            test_name="inception_v3.inference",
        )

    def testInceptionTrain(self):
        # inception model returns a namedtuple (logits, aux_logits)
        # so will wrap the model to return their concatenation
        class InceptionWrapper(torch.nn.Module):
            def __init__(self):
                super(InceptionWrapper, self).__init__()
                self.model = torchvision.models.inception_v3()

            def forward(self, x):
                (logits, aux_logits) = self.model(x)
                return torch.cat([logits, aux_logits])

        model = InceptionWrapper()

        input = torch.randn((32, 3, 299, 299), requires_grad=True)

        model.train()
        self.run_tests(
            model,
            input,
            mode="train",
            methods=[
                "aotautograd",
                # "functorch", #FIXME: Error message: P523547974
            ],
            test_name="inception_v3.train",
        )

    def testGoogleNetInference(self):
        model = torchvision.models.googlenet()
        input = torch.randn((1, 3, 224, 224))

        model.eval()
        self.run_tests(
            model,
            input,
            mode="eval",
            methods=[
                # "fx", # FIXME
                "aotautograd",
                "functorch",
                "acc_tracer",
            ],
            test_name="googlenet.inference",
        )

    def testGoogleNetTrain(self):
        class GoogleNetWrapper(torch.nn.Module):
            def __init__(self):
                super(GoogleNetWrapper, self).__init__()
                self.model = torchvision.models.googlenet()

            def forward(self, x):
                rslt = self.model(x)
                return torch.concat((rslt[0], rslt[1], rslt[2]), dim=-1)

        model = GoogleNetWrapper()
        input = torch.randn((1, 3, 224, 224), requires_grad=True)

        model.train()
        self.run_tests(
            model,
            input,
            mode="train",
            methods=[
                "aotautograd",
            ],
            test_name="googlenet.train",
        )

    def testShuffleNetInference(self):
        model = torchvision.models.shufflenet_v2_x1_0()
        input = torch.randn((1, 3, 224, 224))

        model.eval()
        self.run_tests(
            model,
            input,
            mode="eval",
            methods=[
                # "fx", # FIXME
                "aotautograd",
                "functorch",
                # "acc_tracer", # FIXME: P524956696
            ],
            test_name="shufflenet_v2_x1_0.inference",
        )

    def testShuffleNetTrain(self):
        model = torchvision.models.shufflenet_v2_x1_0()
        input = torch.randn((1, 3, 224, 224), requires_grad=True)

        model.train()
        self.run_tests(
            model,
            input,
            mode="train",
            methods=[
                "aotautograd",
                # "functorch", # FIXME: Error message: P523546037
            ],
            test_name="shufflenet_v2_x1_0.train",
        )

    def testMobileNetv2Inference(self):
        model = torchvision.models.mobilenet_v2()
        input = torch.randn((1, 3, 224, 224))

        model.eval()
        self.run_tests(
            model,
            input,
            mode="eval",
            methods=[
                # "fx", # FIXME
                "aotautograd",
                "functorch",
                "acc_tracer",
            ],
            test_name="mobilenet_v2.inference",
        )

    def testMobileNetv2Train(self):
        model = torchvision.models.mobilenet_v2()
        input = torch.randn((32, 3, 224, 224), requires_grad=True)

        model.train()
        self.run_tests(
            model,
            input,
            mode="train",
            methods=[
                "aotautograd",
                # "functorch", # FIXME: Error message: P523547567
            ],
            test_name="mobilenet_v2.train",
        )

    def testResNet18Inference(self):
        model = torchvision.models.resnet18()
        input = torch.randn((1, 3, 224, 224))

        model.eval()
        self.run_tests(
            model,
            input,
            mode="eval",
            methods=[
                # "fx", # FIXME
                "aotautograd",
                "functorch",
                "acc_tracer",
            ],
            test_name="resnet18.inference",
        )

    def testResNet18Train(self):
        model = torchvision.models.resnet18()
        input = torch.randn((32, 3, 224, 224), requires_grad=True)

        model.train()
        self.run_tests(
            model,
            input,
            mode="train",
            methods=[
                "aotautograd",
                # "functorch", # FIXME: error message: P523546919
            ],
            test_name="resnet18.train",
        )

    def testResNext50_32x4dInference(self):
        model = torchvision.models.resnext50_32x4d()
        input = torch.randn((1, 3, 224, 224))

        model.eval()
        self.run_tests(
            model,
            input,
            mode="eval",
            methods=[
                # "fx", # FIXME
                "aotautograd",
                "functorch",
                "acc_tracer",
            ],
            test_name="resnext50_32x4d.inference",
        )

    def testResNext50_32x4dTrain(self):
        model = torchvision.models.resnext50_32x4d()
        input = torch.randn((1, 3, 224, 224), requires_grad=True)

        model.train()
        self.run_tests(
            model,
            input,
            mode="train",
            methods=[
                "aotautograd",
                # "functorch", # FIXME: Error message: P523546356
            ],
            test_name="resnext50_32x4d.train",
        )

    def testWideResNet50_2Inference(self):
        model = torchvision.models.wide_resnet50_2()
        input = torch.randn((1, 3, 224, 224))

        model.eval()
        self.run_tests(
            model,
            input,
            mode="eval",
            methods=[
                # "fx", # FIXME
                "aotautograd",
                "functorch",
                "acc_tracer",
            ],
            test_name="wide_resnet50_2.inference",
        )

    def testWideResNet50_2Train(self):
        model = torchvision.models.wide_resnet50_2()
        input = torch.randn((1, 3, 224, 224), requires_grad=True)

        model.train()
        self.run_tests(
            model,
            input,
            mode="train",
            methods=[
                "aotautograd",
                # "functorch", # FIXME: Error message: P523543780
            ],
            test_name="wide_resnet50_2.train",
        )

    def testMNasNetInference(self):
        model = torchvision.models.mnasnet1_0()
        input = torch.randn((1, 3, 224, 224))

        model.eval()
        self.run_tests(
            model,
            input,
            mode="eval",
            methods=[
                # "fx", # FIXME
                "aotautograd",
                "functorch",
                "acc_tracer",
            ],
            test_name="mnasnet1_0.inference",
        )

    def testMNasNetTrain(self):
        model = torchvision.models.mnasnet1_0()
        input = torch.randn((1, 3, 224, 224), requires_grad=True)

        model.train()
        self.run_tests(
            model,
            input,
            mode="train",
            methods=[
                "aotautograd",
                # "functorch", # FIXME: Error message: P523547974
            ],
            test_name="mnasnet1_0.train",
        )

    def testVGG11Inference(self):
        model = torchvision.models.vgg11()
        input = torch.randn((1, 3, 224, 224))

        model.eval()
        self.run_tests(
            model,
            input,
            mode="eval",
            methods=[
                "fx",
                "aotautograd",
                "functorch",
                "acc_tracer",
            ],
            test_name="vgg11.inference",
        )

    def testVGG11TrainDefaultOptimizer(self):
        model = torchvision.models.vgg11()
        input = torch.randn((1, 3, 224, 224), requires_grad=True)
        optimizer = True

        model.train()
        self.run_tests(
            model,
            input,
            mode="train",
            methods=[
                "aotautograd",
                # "functorch", # FIXME: Error message: P523543991
            ],
            optimizer=optimizer,
            test_name="vgg11.train",
        )
