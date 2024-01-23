import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


class ResNet50(torch.nn.Module):
    dims = {
        "l6": 3,
        "l5": 64,
        "l4": 256,
        "l3": 512,
        "l2": 1024,
        "l1": 2048,
    }
    pyramid_level_names = ["l6", "l5", "l4", "l3", "l2", "l1"]

    def __init__(self, drop_out: float = 0):
        super(ResNet50, self).__init__()

        model_pretrained = resnet50(weights=ResNet50_Weights.DEFAULT)
        layer_names = [name for name, _ in model_pretrained.named_children()]
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        for x in range(0, 4):
            self.stage1.add_module(str(x), getattr(model_pretrained, layer_names[x]))
        self.stage1.add_module("drop", nn.Dropout2d(drop_out))
        for x in range(4, 5):
            self.stage2.add_module(str(x), getattr(model_pretrained, layer_names[x]))
        self.stage2.add_module("drop", nn.Dropout2d(drop_out))
        for x in range(5, 6):
            self.stage3.add_module(str(x), getattr(model_pretrained, layer_names[x]))
        self.stage3.add_module("drop", nn.Dropout2d(drop_out))
        for x in range(6, 7):
            self.stage4.add_module(str(x), getattr(model_pretrained, layer_names[x]))
        self.stage4.add_module("drop", nn.Dropout2d(drop_out))
        for x in range(7, 9):
            self.stage5.add_module(str(x), getattr(model_pretrained, layer_names[x]))
        self.stage5.add_module("drop", nn.Dropout2d(drop_out))

    def forward(self, x):
        f5 = self.stage1(x)
        f4 = self.stage2(f5)
        f3 = self.stage3(f4)
        f2 = self.stage4(f3)
        f1 = self.stage5(f2)
        emb = torch.mean(f1, dim=(-2, -1))
        return {"emb": emb, "l1": f1, "l2": f2, "l3": f3, "l4": f4, "l5": f5}


class ResNet18(torch.nn.Module):
    dims = {
        "l6": 3,
        "l5": 64,
        "l4": 64,
        "l3": 128,
        "l2": 256,
        "l1": 512,
    }
    pyramid_level_names = ["l6", "l5", "l4", "l3", "l2", "l1"]

    def __init__(self, drop_out: float = 0.0):
        super(ResNet18, self).__init__()

        model_pretrained = resnet18(weights=ResNet18_Weights.DEFAULT)
        layer_names = [name for name, _ in model_pretrained.named_children()]
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        for x in range(0, 4):
            self.stage1.add_module(str(x), getattr(model_pretrained, layer_names[x]))
        self.stage1.add_module("drop", nn.Dropout2d(drop_out))
        for x in range(4, 5):
            self.stage2.add_module(str(x), getattr(model_pretrained, layer_names[x]))
        self.stage2.add_module("drop", nn.Dropout2d(drop_out))
        for x in range(5, 6):
            self.stage3.add_module(str(x), getattr(model_pretrained, layer_names[x]))
        self.stage3.add_module("drop", nn.Dropout2d(drop_out))
        for x in range(6, 7):
            self.stage4.add_module(str(x), getattr(model_pretrained, layer_names[x]))
        self.stage4.add_module("drop", nn.Dropout2d(drop_out))
        for x in range(7, 9):
            self.stage5.add_module(str(x), getattr(model_pretrained, layer_names[x]))
        self.stage5.add_module("drop", nn.Dropout2d(drop_out))

    def forward(self, x):
        f5 = self.stage1(x)
        f4 = self.stage2(f5)
        f3 = self.stage3(f4)
        f2 = self.stage4(f3)
        f1 = self.stage5(f2)
        emb = torch.mean(f1, dim=(-2, -1))

        return {"emb": emb, "l5": f5, "l4": f4, "l3": f3, "l2": f2, "l1": f1}
