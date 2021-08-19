import torch

from .vgg16_bn import AutoEncoder
from .dummy_model import DummyAE

_autoencoder_factory = {
    'dummy':        DummyAE,
    'vgg16_bn':     AutoEncoder
}


def get_autoencoder(model_name: str, pretrained: bool = True, checkpoint: str = None):
    try:
        get_autoencoder = _autoencoder_factory[model_name]
    except:
        print(f'This model name {model_name} is not defined.')
        return
    autoencoder = get_autoencoder(pretrained=pretrained)
    if checkpoint is not None:
        try:
            state_dict = torch.load(checkpoint)['state_dict']
            autoencoder.load_state_dict(state_dict)
        except:
            print("Can't Load StateDict")
    return autoencoder
