import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# =========================================================
# 🔧 1. Escolher camada automaticamente
# =========================================================
def get_target_layers(model):
    """
    Seleciona automaticamente uma camada adequada para Grad-CAM
    compatível com múltiplas arquiteturas.
    """

    # 🔹 ResNet-like
    if hasattr(model, "layer4"):
        return [model.layer4[-1]]

    # 🔹 EfficientNet / DenseNet / MobileNet
    if hasattr(model, "features"):
        if isinstance(model.features, nn.Sequential):
            return [model.features[-1]]

    # 🔹 Vision Transformers (timm)
    if hasattr(model, "blocks"):
        return [model.blocks[-1]]

    # 🔹 Última camada convolucional
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is not None:
        return [last_conv]

    # 🔹 Fallback: penúltima camada
    modules = list(model.children())
    if len(modules) > 1:
        return [modules[-2]]

    # 🔹 Último recurso
    return [modules[-1]]


# =========================================================
# 🚀 2. Classe wrapper (recomendado)
# =========================================================
class GradCAMWrapper:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

        self.model.eval()
        self.model.to(self.device)

        # 🔥 escolhe camada automaticamente
        self.target_layers = get_target_layers(model)

        print(f"[GradCAM] Using layer: {self.target_layers[0]}")

        # 🔥 cria extractor UMA VEZ
        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers
        )

    def generate(self, input_tensor, rgb_img, pred_class, save_path):
        """
        Gera e salva Grad-CAM

        Args:
            input_tensor: tensor (1, C, H, W)
            pred_class: int
            save_path: caminho para salvar imagem
        """

        input_tensor = input_tensor.to(self.device)

        targets = [ClassifierOutputTarget(pred_class)]

        # 🔥 gera CAM
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets
        )

        cam_img = grayscale_cam[0]

        # 🔥 overlay
        overlay = show_cam_on_image(
            rgb_img.astype(np.float32),
            cam_img,
            use_rgb=True
        )

        # 🔥 salvar
        Image.fromarray(overlay).save(save_path)