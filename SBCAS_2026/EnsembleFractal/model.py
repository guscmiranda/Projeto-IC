import torch.nn as nn

from  SBCAS_2026.EnsembleFractal.utils import *
from torchvision import models
from torchvision.models import (
    MobileNet_V2_Weights,
    EfficientNet_B0_Weights,
)
import torch.nn as nn

def criar_modelo(backbone: str, num_classes: int, pretrained=True):
    backbone = backbone.lower()

    if backbone == "mobilenet":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    elif backbone == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    else:
        raise ValueError("backbone deve ser 'mobilenet' ou 'efficientnet_b0'")

    return model


def carregar_modelo(backbone, num_classes, path_weights):
    print(f"Carregando {backbone} de {path_weights}...")
    backbone = backbone.lower()
    if backbone == "mobilenet":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Carrega os pesos treinados
    try:
        model.load_state_dict(torch.load(path_weights, map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERRO: Arquivo {path_weights} não encontrado! Treine o modelo antes.")
        return None

    model.to(DEVICE)
    model.eval()
    return model