import torch
import torchvision.models as models
import pnnx

def export_resnet50():
    model = models.resnet50(pretrained=True)
    x = torch.rand(1, 3, 224, 224)
    opt_model = pnnx.export(model, "resnet50.pt", x)

if __name__ == "__main__":
    export_resnet50()
