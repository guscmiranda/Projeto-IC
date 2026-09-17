import sys
from pathlib import Path

# ============================================================
# ESTRUTURA DO PROJETO
# ============================================================
#
# C:\projects\IC\
# ├── SBCAS\
# │   └── SBCAS_2026\
# │       └── EnsembleFractal\
# │
# └── TTA\
#     ├── ensemble.py
#     ├── tta_outputs_v2\
#     └── codigos_fractal\
#
# ============================================================

TTA_DIR = Path(__file__).resolve().parent
IC_DIR = TTA_DIR.parent
SBCAS_DIR = IC_DIR / "SBCAS"

# Permite:
# from SBCAS_2026.EnsembleFractal...
sys.path.insert(0, str(SBCAS_DIR))


# ============================================================
# IMPORTS
# ============================================================

import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from tqdm import tqdm

from SBCAS_2026.EnsembleFractal.model import carregar_modelo
from SBCAS_2026.EnsembleFractal.dataset import (
    load_data_from_folders,
    ImageDataset
)
from SBCAS_2026.EnsembleFractal.utils import (
    DEVICE,
    BATCH_SIZE,
    SEEDS,
    TRANSFORM_DICT
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CLASSES = ["healthy", "severe"]
NUM_CLASSES = len(CLASSES)

TTA_AGGREGATION = "mean_prob"

# ------------------------------------------------------------
# Diretórios
# ------------------------------------------------------------

TEST_DIR = (
    IC_DIR
    / "codigos_fractal"
    / "datasets"
    / "dataset_displasia"
    / "teste"
)

TTA_OUTPUTS_DIR = (
    TTA_DIR / "tta_outputs_v2"
)

RP_MODELS_DIR = (
    IC_DIR
    / "codigos_fractal"
    / "ensemble_displasia_2.0"
    / "models"
)

OUTPUT_DIR = (
    TTA_DIR / f"ensemble_results_{TTA_AGGREGATION}_baseline"
)

# ------------------------------------------------------------
# Estratégias escolhidas para cada arquitetura
# ------------------------------------------------------------

TTA_STRATEGIES = {
    "mobilenet": "none+T_F",
    "efficientnet_b0": "none+T_G"
}

# ------------------------------------------------------------
# Nomes das pastas dos modelos originais
# ------------------------------------------------------------
#
# Ajuste SOMENTE se os nomes reais das pastas forem diferentes.
#
# Exemplo:
#
# tta_outputs_v2/
# ├── mobilenet/
# └── efficientnet_b0/
#
# ------------------------------------------------------------

ORIGINAL_MODEL_NAMES = {
    "mobilenet": "mobilenet_originais_T_D",
    "efficientnet_b0": "efficientnet_b0_originaisT_D"
}


# ============================================================
# INFORMAÇÕES INICIAIS
# ============================================================

print("=" * 70)
print("ENSEMBLE - DISPLASIA")
print("=" * 70)

print(f"Diretório TTA:       {TTA_DIR}")
print(f"Diretório SBCAS:     {SBCAS_DIR}")
print(f"Dataset de teste:    {TEST_DIR}")
print(f"Resultados TTA:      {TTA_OUTPUTS_DIR}")
print(f"Modelos F-RecPlot:   {RP_MODELS_DIR}")
print(f"Saída dos ensembles: {OUTPUT_DIR}")
print(f"Dispositivo:         {DEVICE}")

print("\nEstratégias escolhidas:")
print(
    f"  MobileNetV2:      {TTA_STRATEGIES['mobilenet']}"
)
print(
    f"  EfficientNet-B0:  {TTA_STRATEGIES['efficientnet_b0']}"
)

print("=" * 70)


# ============================================================
# 1. CARREGAR CSV DO MODELO ORIGINAL
# ============================================================

def load_original_results(model_name, seed):
    """
    Carrega o aggregated_results.csv de um modelo original.

    Estrutura esperada:

    tta_outputs_v2/
        modelo/
            seeds/
                seed/
                    aggregated_results.csv
    """

    path = (
        TTA_OUTPUTS_DIR
        / model_name
        / "seeds"
        / str(seed)
        / "aggregated_results.csv"
    )

    print(f"\nCarregando resultados:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nArquivo não encontrado:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"  {len(df)} registros encontrados."
    )

    return df


# ============================================================
# 2. TRANSFORMAR VOTOS EM PROBABILIDADES
# ============================================================

def add_vote_probabilities(df):
    """
    Converte:

        votes_for_0
        votes_for_1

    em:

        vote_prob_0
        vote_prob_1

    Exemplo:

        votes_for_0 = 5
        votes_for_1 = 1

        vote_prob_0 = 5/6
        vote_prob_1 = 1/6
    """

    df = df.copy()

    total_votes = (
        df["votes_for_0"]
        + df["votes_for_1"]
    )

    if (total_votes == 0).any():
        raise ValueError(
            "Encontrada amostra com zero votos."
        )

    df["vote_prob_0"] = (
        df["votes_for_0"] /
        total_votes
    )

    df["vote_prob_1"] = (
        df["votes_for_1"] /
        total_votes
    )

    return df


# ============================================================
# 3. PREPARAR RESULTADOS DO MODELO ORIGINAL
# ============================================================

def prepare_original_predictions_vote(
    df,
    tta_strategy
):
    """
    Agregação TTA por votação.

    P(classe 0) = votes_for_0 / total_votes
    P(classe 1) = votes_for_1 / total_votes
    """

    df = df.copy()

    df = df[
        df["tta_strategy"] == tta_strategy
    ].copy()

    if df.empty:
        raise ValueError(
            f"Estratégia '{tta_strategy}' não encontrada."
        )

    total_votes = (
        df["votes_for_0"] +
        df["votes_for_1"]
    )

    if (total_votes == 0).any():
        raise ValueError(
            "Encontrada imagem com zero votos."
        )

    df["prob_0"] = (
        df["votes_for_0"] /
        total_votes
    )

    df["prob_1"] = (
        df["votes_for_1"] /
        total_votes
    )

    df["pred"] = np.where(
        df["prob_1"] > df["prob_0"],
        1,
        0
    )

    df["sample_id"] = list(
        zip(
            df["true_label"].astype(int),
            df["image_id"].astype(int)
        )
    )

    return df[
        [
            "sample_id",
            "image_id",
            "true_label",
            "prob_0",
            "prob_1",
            "pred"
        ]
    ].copy()


def prepare_original_predictions_mean_prob(
    df,
    tta_strategy
):
    """
    Agregação TTA pela média das probabilidades.

    Utiliza diretamente:

        mean_prob_0
        mean_prob_1

    presentes no aggregated_results.csv.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Selecionar estratégia TTA
    # --------------------------------------------------------

    df = df[
        df["tta_strategy"] == tta_strategy
    ].copy()

    if df.empty:
        raise ValueError(
            f"Estratégia '{tta_strategy}' não encontrada."
        )

    # --------------------------------------------------------
    # Verificar se as colunas existem
    # --------------------------------------------------------

    required_columns = [
        "mean_prob_0",
        "mean_prob_1",
        "image_id",
        "true_label"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Colunas ausentes no CSV: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Utilizar diretamente a média das probabilidades
    # --------------------------------------------------------

    df["prob_0"] = df["mean_prob_0"]
    df["prob_1"] = df["mean_prob_1"]

    # --------------------------------------------------------
    # Predição baseada na maior probabilidade média
    # --------------------------------------------------------

    df["pred"] = np.where(
        df["prob_1"] > df["prob_0"],
        1,
        0
    )

    # --------------------------------------------------------
    # Identificador da amostra
    # --------------------------------------------------------

    df["sample_id"] = list(
        zip(
            df["true_label"].astype(int),
            df["image_id"].astype(int)
        )
    )

    # --------------------------------------------------------
    # Retorno padronizado
    # --------------------------------------------------------

    return df[
        [
            "sample_id",
            "image_id",
            "true_label",
            "prob_0",
            "prob_1",
            "pred"
        ]
    ].copy()


def prepare_original_predictions(
    df,
    tta_strategy,
    aggregation="mean_prob"
):
    """
    Seleciona o método de agregação TTA.
    """

    if aggregation == "vote":

        return prepare_original_predictions_vote(
            df,
            tta_strategy
        )

    elif aggregation == "mean_prob":

        return prepare_original_predictions_mean_prob(
            df,
            tta_strategy
        )

    else:

        raise ValueError(
            "aggregation deve ser "
            "'vote' ou 'mean_prob'."
        )
# ============================================================
# 4. CARREGAR MODELO F-RECPLOT
# ============================================================

def load_rp_model(backbone, seed):
    """
    Carrega o modelo F-RecPlot correspondente à seed.

    Estrutura esperada:

    models/
        7/
            mobilenet_F-RecPlot.pth
            efficientnet_b0_F-RecPlot.pth

        12/
            ...
    """

    filename = (
        f"{backbone}_F-RecPlot.pth"
    )

    path = (
        RP_MODELS_DIR
        / str(seed)
        / filename
    )

    print(
        f"\nCarregando modelo F-RecPlot:"
    )
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nModelo não encontrado:\n{path}"
        )

    model = carregar_modelo(
        backbone=backbone,
        num_classes=NUM_CLASSES,
        path_weights=str(path)
    )

    if model is None:
        raise RuntimeError(
            f"Não foi possível carregar:\n{path}"
        )

    return model


# ============================================================
# 5. INFERÊNCIA F-RECPLOT
# ============================================================

def predict_recplot(model):
    """
    Realiza inferência nas imagens F-RecPlot.

    IMPORTANTE:
    utiliza exatamente:

        TRANSFORM_DICT["transform"]

    que corresponde a:

        Resize(224, 224)
        ToTensor()
        Normalize(ImageNet)
    """

    # --------------------------------------------------------
    # Carregar imagens F-RecPlot
    # --------------------------------------------------------

    data_list = load_data_from_folders(
        str(TEST_DIR),
        CLASSES,
        "F-RecPlot"
    )

    if len(data_list) == 0:
        raise ValueError(
            "\nNenhuma imagem F-RecPlot encontrada."
        )

    print(
        f"  Imagens F-RecPlot: {len(data_list)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = ImageDataset(
        data_list,
        transform=TRANSFORM_DICT["transform"]
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Inferência
    # --------------------------------------------------------

    model.eval()

    results = []

    offset = 0

    with torch.no_grad():

        for images, labels in tqdm(
            loader,
            desc="Inferência F-RecPlot",
            leave=False
        ):

            images = images.to(DEVICE)

            logits = model(images)

            probs = torch.softmax(
                logits,
                dim=1
            )

            preds = torch.argmax(
                probs,
                dim=1
            )

            probs = (
                probs
                .detach()
                .cpu()
                .numpy()
            )

            preds = (
                preds
                .detach()
                .cpu()
                .numpy()
            )

            labels = (
                labels
                .detach()
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Salvar resultado de cada imagem
            # ------------------------------------------------

            for i in range(len(labels)):

                img_path = data_list[
                    offset + i
                ][0]

                # nome sem extensão
                stem = Path(
                    img_path
                ).stem

                try:
                    image_id = int(stem)
                except ValueError:
                    raise ValueError(
                        f"\nNão foi possível converter "
                        f"'{stem}' para image_id.\n"
                        f"Arquivo: {img_path}"
                    )

                true_label = int(
                    labels[i]
                )

                results.append({
                    "sample_id": (
                        true_label,
                        image_id
                    ),

                    "image_id": image_id,

                    "true_label": true_label,

                    "prob_0": float(
                        probs[i, 0]
                    ),

                    "prob_1": float(
                        probs[i, 1]
                    ),

                    "pred": int(
                        preds[i]
                    )
                })

            offset += len(labels)

    return pd.DataFrame(results)


# ============================================================
# 6. ENSEMBLE POR MÉDIA DAS PROBABILIDADES
# ============================================================

def average_ensemble(
    df_a,
    df_b,
    name_a,
    name_b
):
    """
    Ensemble entre dois modelos.

    P_ensemble(c) =
        (P_modelo_A(c) + P_modelo_B(c)) / 2

    A classe escolhida é aquela com maior
    probabilidade média.
    """

    merged = pd.merge(
        df_a,
        df_b,

        on=[
            "sample_id",
            "image_id",
            "true_label"
        ],

        how="inner",

        suffixes=("_a", "_b"),

        validate="one_to_one"
    )

    if len(merged) == 0:
        raise ValueError(
            f"\nNenhuma amostra em comum entre "
            f"{name_a} e {name_b}."
        )

    # --------------------------------------------------------
    # Verificação
    # --------------------------------------------------------

    if len(merged) != len(df_a):

        print(
            f"\nAVISO:"
            f"\n{name_a}: {len(df_a)} amostras"
            f"\n{name_b}: {len(df_b)} amostras"
            f"\nEnsemble: {len(merged)} amostras"
        )

    # --------------------------------------------------------
    # Média das probabilidades
    # --------------------------------------------------------

    merged["ensemble_prob_0"] = (
        merged["prob_0_a"]
        + merged["prob_0_b"]
    ) / 2.0

    merged["ensemble_prob_1"] = (
        merged["prob_1_a"]
        + merged["prob_1_b"]
    ) / 2.0

    # --------------------------------------------------------
    # Predição
    # --------------------------------------------------------

    merged["ensemble_pred"] = np.where(
        merged["ensemble_prob_1"]
        > merged["ensemble_prob_0"],

        1,

        0
    )

    # --------------------------------------------------------
    # Informações dos modelos
    # --------------------------------------------------------

    merged["model_a"] = name_a
    merged["model_b"] = name_b

    return merged


# ============================================================
# 7. MÉTRICAS
# ============================================================

def calculate_metrics(df):

    y_true = df["true_label"]
    y_pred = df["ensemble_pred"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    metrics = {

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),

        "n_samples": len(df)
    }

    return metrics


# ============================================================
# 8. EXECUTAR ENSEMBLES
# ============================================================

def run_ensembles():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_metrics = []

    # ========================================================
    # LOOP DAS SEEDS
    # ========================================================

    for seed in SEEDS:

        print("\n\n")
        print("#" * 70)
        print(f"# SEED {seed}")
        print("#" * 70)

        # ====================================================
        # 1. MODELOS ORIGINAIS
        # ====================================================

        print("\n" + "-" * 70)
        print("MODELOS ORIGINAIS")
        print("-" * 70)

        # ----------------------------------------------------
        # MobileNet Original
        # ----------------------------------------------------

        mob_orig_raw = load_original_results(
            ORIGINAL_MODEL_NAMES["mobilenet"],
            seed
        )

        mob_orig = prepare_original_predictions(
            mob_orig_raw,
            tta_strategy=TTA_STRATEGIES["mobilenet"],
            aggregation=TTA_AGGREGATION
        )

        print(
            f"MobileNet Original:"
            f" {TTA_STRATEGIES['mobilenet']}"
        )

        # ----------------------------------------------------
        # EfficientNet Original
        # ----------------------------------------------------

        eff_orig_raw = load_original_results(
            ORIGINAL_MODEL_NAMES[
                "efficientnet_b0"
            ],
            seed
        )

        eff_orig = prepare_original_predictions(
            eff_orig_raw,
            tta_strategy=TTA_STRATEGIES["efficientnet_b0"],
            aggregation=TTA_AGGREGATION
        )

        print(
            f"EfficientNet-B0 Original:"
            f" {TTA_STRATEGIES['efficientnet_b0']}"
        )

        # ====================================================
        # 2. MODELOS F-RECPLOT
        # ====================================================

        print("\n" + "-" * 70)
        print("MODELOS F-RECPLOT")
        print("-" * 70)

        # ----------------------------------------------------
        # MobileNet F-RecPlot
        # ----------------------------------------------------

        mob_rp_model = load_rp_model(
            "mobilenet",
            seed
        )

        print(
            "\nInferindo MobileNet F-RecPlot..."
        )

        mob_rp = predict_recplot(
            mob_rp_model
        )

        del mob_rp_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # EfficientNet-B0 F-RecPlot
        # ----------------------------------------------------

        eff_rp_model = load_rp_model(
            "efficientnet_b0",
            seed
        )

        print(
            "\nInferindo EfficientNet-B0 F-RecPlot..."
        )

        eff_rp = predict_recplot(
            eff_rp_model
        )

        # # ============================================================
        # # AUDITORIA: MOBILE NET RP VS EFFICIENT NET RP
        # # ============================================================

        # rp_compare = pd.merge(
        #     mob_rp,
        #     eff_rp,
        #     on=[
        #         "sample_id",
        #         "image_id",
        #         "true_label"
        #     ],
        #     suffixes=("_mob", "_eff"),
        #     how="inner",
        #     validate="one_to_one"
        # )

        # # ------------------------------------------------------------
        # # Comparação das previsões
        # # ------------------------------------------------------------

        # same_predictions = (
        #     rp_compare["pred_mob"]
        #     ==
        #     rp_compare["pred_eff"]
        # )

        # n_same_predictions = same_predictions.sum()
        # n_total = len(rp_compare)

        # print("\n" + "=" * 70)
        # print("AUDITORIA MOBILE NET F-RECPLOT × EFFICIENTNET F-RECPLOT")
        # print("=" * 70)

        # print(
        #     f"Imagens comparadas: {n_total}"
        # )

        # print(
        #     f"Predições iguais: "
        #     f"{n_same_predictions}/{n_total}"
        # )

        # print(
        #     f"Percentual de predições iguais: "
        #     f"{100 * n_same_predictions / n_total:.2f}%"
        # )

        # print(
        #     f"Predições diferentes: "
        #     f"{n_total - n_same_predictions}"
        # )

        # # ------------------------------------------------------------
        # # Comparação das probabilidades
        # # ------------------------------------------------------------

        # prob_diff_0 = (
        #     rp_compare["prob_0_mob"]
        #     -
        #     rp_compare["prob_0_eff"]
        # ).abs()

        # prob_diff_1 = (
        #     rp_compare["prob_1_mob"]
        #     -
        #     rp_compare["prob_1_eff"]
        # ).abs()

        # print(
        #     f"\nDiferença média P(classe 0): "
        #     f"{prob_diff_0.mean():.6f}"
        # )

        # print(
        #     f"Diferença média P(classe 1): "
        #     f"{prob_diff_1.mean():.6f}"
        # )

        # print(
        #     f"Maior diferença P(classe 0): "
        #     f"{prob_diff_0.max():.6f}"
        # )

        # print(
        #     f"Maior diferença P(classe 1): "
        #     f"{prob_diff_1.max():.6f}"
        # )
        

        # del eff_rp_model

        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        # ====================================================
        # 3. ORGANIZAR MODELOS
        # ====================================================

        models_dict = {

            # Original
            "MO": mob_orig,
            "EO": eff_orig,

            # RecPlot
            "MR": mob_rp,
            "ER": eff_rp
        }

        # ====================================================
        # 4. ENSEMBLES
        # ====================================================
        #
        # MO = MobileNet Original
        # EO = EfficientNet Original
        # MR = MobileNet F-RecPlot
        # ER = EfficientNet F-RecPlot
        #
        # ====================================================

        ensemble_pairs = [

            # modelos individuais
            ("MO", "MO"),
            ("EO", "EO"),
            ("MR", "MR"),
            ("ER", "ER"),

            # mesma representação
            ("MO", "EO"),
            ("MR", "ER"),

            # mesma arquitetura
            ("MO", "MR"),
            ("EO", "ER"),

            # arquitetura + representação diferentes
            ("MO", "ER"),
            ("EO", "MR")
        ]

        # ====================================================
        # DIRETÓRIO DA SEED
        # ====================================================

        seed_output_dir = (
            OUTPUT_DIR
            / f"seed_{seed}"
        )

        seed_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ====================================================
        # EXECUTAR CADA ENSEMBLE
        # ====================================================

        for name_a, name_b in ensemble_pairs:

            ensemble_name = (
                f"{name_a}+{name_b}"
            )

            print("\n" + "=" * 60)
            print(
                f"ENSEMBLE: {ensemble_name}"
            )
            print("=" * 60)

            df_ensemble = average_ensemble(
                models_dict[name_a],
                models_dict[name_b],
                name_a,
                name_b
            )

            # ------------------------------------------------
            # Métricas
            # ------------------------------------------------

            metrics = calculate_metrics(
                df_ensemble
            )

            metrics.update({

                "seed": seed,

                "ensemble": ensemble_name,

                "mobilenet_tta": (
                    TTA_STRATEGIES["mobilenet"]
                ),

                "efficientnet_tta": (
                    TTA_STRATEGIES[
                        "efficientnet_b0"
                    ]
                )
            })

            all_metrics.append(
                metrics
            )

            # ------------------------------------------------
            # Salvar previsões
            # ------------------------------------------------

            output_file = (
                seed_output_dir
                / f"{ensemble_name}.csv"
            )

            df_ensemble.to_csv(
                output_file,
                index=False
            )

            # ------------------------------------------------
            # Mostrar resultado
            # ------------------------------------------------

            print(
                f"\nAmostras: "
                f"{metrics['n_samples']}"
            )

            print(
                f"Accuracy: "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"F1 Macro: "
                f"{metrics['f1_macro']:.4f}"
            )

            print(
                f"Precision Macro: "
                f"{metrics['precision_macro']:.4f}"
            )

            print(
                f"Recall Macro: "
                f"{metrics['recall_macro']:.4f}"
            )

            print(
                f"Matriz de confusão:"
            )

            print(
                f"[[{metrics['tn']} {metrics['fp']}]"
            )

            print(
                f" [{metrics['fn']} {metrics['tp']}]]"
            )

            print(
                f"\nSalvo em:"
                f"\n{output_file}"
            )

    # ========================================================
    # 5. RESULTADOS GERAIS
    # ========================================================

    df_metrics = pd.DataFrame(
        all_metrics
    )

    metrics_file = (
        OUTPUT_DIR
        / "ensemble_metrics.csv"
    )

    df_metrics.to_csv(
        metrics_file,
        index=False
    )

    print("\n\n")
    print("#" * 70)
    print("# RESULTADOS FINAIS")
    print("#" * 70)

    print(
        df_metrics[
            [
                "seed",
                "ensemble",
                "accuracy",
                "f1_macro",
                "precision_macro",
                "recall_macro"
            ]
        ].to_string(index=False)
    )

    print(
        f"\nResultados salvos em:"
        f"\n{metrics_file}"
    )

    return df_metrics


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":

    results = run_ensembles()