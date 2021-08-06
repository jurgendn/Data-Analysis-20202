import torchvision.models as models
import torch.nn as nn

_model_factory = {
    'resnet18':     models.resnet18,
    'resnet34':     models.resnet34,
    'resnet50':     models.resnet50,
}


def get_model(model_name: str, pretrained: bool = True):
    try:
        get_backbone = _model_factory[model_name]
    except:
        print(f'This model name {model_name} is not defined. Feature extractor will be None')
        return
    backbone = get_backbone(pretrained=pretrained)
    
    modules = list(backbone.children())[:-1]
    model = nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = False
    return model
