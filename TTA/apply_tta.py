import os
import csv
import random
import time
from PIL import Image
from pathlib import Path

from matplotlib import image
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

import numpy as np

from tqdm import tqdm

# 🔥 Grad-CAM
from TTA.gradcam_utils import GradCAMWrapper

from SBCAS_2026.EnsembleFractal.model import carregar_modelo


# =========================================================
# ⚙️ CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
N_TTA = 5

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
# =========================================================
# 🔁 TTA TRANSFORMS
# =========================================================
TTA_DICT = {
    "T_A": transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(1.7, 1.9)),

    "T_B": transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(1.7, 1.9)),

    "T_C": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((IMG_SIZE, IMG_SIZE))
    ]),

    "T_D": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(1.7, 1.9)),
    ]),

    "T_E": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(1.7, 1.9)),
    ]),

    "T_F": transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.95,1.0), ratio=(1.7, 1.9))
    ]),

    "T_G": transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9,1.0), ratio=(1.7, 1.9))
    ]),

    "T_H": transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85,1.0), ratio=(1.7, 1.9))
    ]),

    "T_I": transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.98,1.0), ratio=(1.7, 1.9))
    ])
}



# =========================================================
# 🖼️ BASE TRANSFORM
# =========================================================
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# =========================================================
# 💾 CSV INIT
# =========================================================
def init_csv(csv_path):
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id",
            "tta_type",
            "sample_id",
            "logit_0",
            "logit_1",
            "prob_0",
            "prob_1",
            "pred",
            "label"
        ])

# def init_csv(csv_path):
#     with open(csv_path, mode="a", newline="") as f:
#         writer = csv.writer(f)

#         if f.tell() == 0:
#             writer.writerow([
#                 "image_id",
#                 "tta_type",
#                 "sample_id",
#                 "logit_0",
#                 "logit_1",
#                 "prob_0",
#                 "prob_1",
#                 "pred",
#                 "label"
#             ])


from pathlib import Path
from PIL import Image
from torchvision.utils import save_image

from torchvision.transforms.functional import resized_crop

def generate_tta_dataset(image_paths, save_root):
    for img_path in image_paths:
        p = Path(img_path)

        class_name = p.parent.parent.name
        image_id = p.stem

        image = Image.open(img_path).convert("RGB")

        save_dir = Path(save_root) / class_name
        save_dir.mkdir(parents=True, exist_ok=True)

        x_orig = transforms.Resize((IMG_SIZE, IMG_SIZE))(image)
        x_orig = transforms.ToTensor()(x_orig)

        save_image(
            x_orig,
            save_dir / f"{image_id}_orig.png"
        )

        for tta_name, transform in TTA_DICT.items():
            for i in range(N_TTA):

                x_aug = transform(image)
                x_aug = transforms.ToTensor()(x_aug)

                save_image(
                    x_aug,
                    save_dir / f"{image_id}_{tta_name}_{i}.png"
                )

def tensor_to_rgb(tensor, mean, std):
    img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    return img.astype(np.float32)

# =========================================================
# 🚀 MAIN PIPELINE
# =========================================================


def run_tta_pipeline(model, image_paths, save_dir, csv_path):
    model.to(DEVICE)
    model.eval()

    # 🔥 cria Grad-CAM UMA VEZ
    # gradcam = GradCAMWrapper(model, device=DEVICE)

    init_csv(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        for img_path in tqdm(image_paths):
            
            p = Path(img_path)
            label = p.parent.name  # healthy / severe
            image_id = p.stem
            
            # 🔍 tentar extrair tta_name e índice do nome
            # ex: 1_T_A_0.png
            
            parts = image_id.split("_")
            if len(parts) >= 4:
                tta_name = parts[1] + "_" + parts[2]
                i = parts[-1]
                base_image_id = parts[0]
            else:
                tta_name = "none"
                i = 0
                base_image_id = parts[0]
            

            # 📥 carregar imagem
            image = Image.open(img_path).convert("RGB")
            x = base_transform(image)
            x = x.unsqueeze(0).to(DEVICE)

            # 🎨 imagem original (para CAM)
            
            # image = image.resize((IMG_SIZE, IMG_SIZE))
            # rgb_img = np.array(image).astype(np.float32) / 255.0

            rgb_img = tensor_to_rgb(x, mean, std)

            # 🔮 forward
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)

            logit_0 = logits[0, 0].item()
            logit_1 = logits[0, 1].item()
            prob_0 = probs[0, 0].item()
            prob_1 = probs[0, 1].item()
            pred = int(torch.argmax(logits, dim=1).item())

            # 💾 salvar Grad-CAM
            # save_subdir = os.path.join(save_dir, label)
            # os.makedirs(save_subdir, exist_ok=True)

            # cam_save_path = os.path.join(
            #     save_subdir, f"{image_id}_cam.png"
            # )

            # gradcam.generate(
            #     input_tensor=x,
            #     rgb_img=rgb_img,
            #     pred_class=pred,
            #     save_path=cam_save_path
            # )

            # 💾 CSV
            writer.writerow([
                base_image_id,
                tta_name,
                i,
                logit_0,
                logit_1,
                prob_0,
                prob_1,
                pred,
                label
            ])


# Usando modelos treinados na seed 7

# =========================================================
# ▶️ EXEMPLO DE USO
# =========================================================
if __name__ == "__main__":
    import glob

    # =====================================================
    # 📂 DATASET
    # =====================================================
    DATASET_DIR = "C:/projects/IC/codigos_fractal/datasets/dataset_displasia/teste" 

    image_paths = []

    for class_name in ["healthy", "severe"]:
        class_dir = os.path.join(DATASET_DIR, class_name, "originais")

        for img_path in glob.glob(os.path.join(class_dir, "*")):
            image_paths.append(img_path)

    # print(f"Total de imagens: {len(image_paths)}")


    save_root = os.path.join("TTA", "tta_dataset")
    # generate_tta_dataset(image_paths, save_root)


    image_paths = []

    for class_name in ["healthy", "severe"]:
        class_dir = os.path.join(save_root, class_name)

        for img_path in glob.glob(os.path.join(class_dir, "*")):
            image_paths.append(img_path)

    # print(f"Total de imagens: {len(image_paths)}")

    

    # =====================================================
    # 🤖 MODELOS (somente originais)
    # =====================================================
    seeds = [7, 12, 42, 65, 87, 93, 107, 121] 

    inicio_total = time.time()
    for seed in tqdm(seeds):
        set_seed(seed)
        MODELS_DIR = f"C:/projects/IC/TTA/models_updated/{seed}"

        model_paths = [
            os.path.join(MODELS_DIR, f)
            for f in os.listdir(MODELS_DIR)
            if "F-RecPlot_transform.pth" not in f
        ]

        # print("Modelos encontrados:")
        # for m in model_paths:
        #     print(" -", m)


        # =====================================================
        # 🚀 LOOP POR MODELO
        # =====================================================
        for model_path in model_paths:
            # print(f"\n🚀 Rodando modelo: {model_path}")

            # detectar backbone
            if "efficientnet" in model_path:
                backbone = "efficientnet_b0"
            elif "mobilenet" in model_path:
                backbone = "mobilenet"
            else:
                raise ValueError(f"Backbone desconhecido: {model_path}")

            # carregar modelo corretamente
            model = carregar_modelo(
                backbone=backbone,
                num_classes=2,
                path_weights=model_path
            )

            if model is None:
                continue

            # pasta específica por modelo
            model_name = os.path.basename(model_path).replace(".pth", "")

            save_dir = os.path.join("TTA", "tta_outputs_updated", model_name, "seeds", str(seed))
            os.makedirs(save_dir, exist_ok=True)

            csv_path = os.path.join(save_dir, "results.csv")

            # rodar pipeline
            run_tta_pipeline(model, image_paths, save_dir, csv_path)

    fim_total = time.time()
    print(f"\nTempo total de execução: {fim_total - inicio_total:.2f} segundos")