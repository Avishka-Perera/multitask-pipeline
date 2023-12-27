import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


class ResNet50(torch.nn.Module):
    def __init__(self, drop_out: float):
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
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        embs = torch.flatten(f5, start_dim=-3)
        return {"embs": embs, "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5}


class ResNet18(torch.nn.Module):
    def __init__(self, drop_out: float):
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
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        embs = torch.flatten(f5, start_dim=-3)
        return {"embs": embs, "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5}
