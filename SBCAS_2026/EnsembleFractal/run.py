from SBCAS_2026.EnsembleFractal.metrics import metrics_to_csv
from SBCAS_2026.EnsembleFractal.model import carregar_modelo

from SBCAS_2026.EnsembleFractal.utils import *
from SBCAS_2026.EnsembleFractal.dataset import *
from SBCAS_2026.EnsembleFractal.train import *

displasia_dataset_path = 'codigos_fractal/datasets/dataset_displasia/treino_e_validacao'
displasia_classes = ['healthy', 'severe']

from tqdm import tqdm
import time

# lung_dataset_path = '../datasets/dataset_pulmao/treino_e_validacao'
# lung_classes = ['aca_md', 'nor', 'scc_md']

displasia_test_dir = 'codigos_fractal/datasets/dataset_displasia/teste'
# lung_test_dir = '../datasets/dataset_pulmao/teste'

####### Testando com o dataset displasia #######
classes = displasia_classes
dataset_path = displasia_dataset_path
TEST_DIR = displasia_test_dir

inicio_total = time.time()

for transform_name in tqdm(TRANSFORM_DICT.keys(), desc = "Transformações"):

    inicio_transform = time.time()

    if transform_name == 'T_C' or transform_name == 'T_D': continue
    
    transform = TRANSFORM_DICT[transform_name]

    # test_dataset = EnsembleTestDataset(TEST_DIR, classes, transform=TRANSFORM_DICT["transform"])
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # print(f"Total de pares de imagens para teste: {len(test_dataset)}")

    results = train_seeds(SEEDS, dataset_path, classes, transform, transform_name)

    # num_classes = len(classes) 

    # for seed in tqdm(SEEDS, desc=f"Seeds ({transform_name})", leave=False):
    #     results[seed] = {}
    #     results[seed]['mobnet_orig'] = carregar_modelo('mobilenet', num_classes,f"TTA/models/{seed}/mobilenet_originais_{transform_name}.pth")
    #     results[seed]['effnet_orig'] = carregar_modelo('efficientnet_b0', num_classes,f"TTA/models/{seed}/efficientnet_b0_originais_{transform_name}.pth")
    #     results[seed]['mobnet_recplot'] = carregar_modelo('mobilenet', num_classes,f"TTA/models/{seed}/mobilenet_F-RecPlot.pth")
    #     results[seed]['effnet_recplot'] = carregar_modelo('efficientnet_b0', num_classes,f"TTA/models/{seed}/efficientnet_b0_F-RecPlot.pth")

    # metrics_to_csv(SEEDS, results, test_loader)

    fim_transform = time.time()
    print(f"Tempo da transformação '{transform_name}': {fim_transform - inicio_transform:.2f} s")

fim_total = time.time()
print(f"\nTempo total: {fim_total - inicio_total:.2f} s")
