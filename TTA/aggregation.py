import os
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score
)

# =========================================================
# LABELS
# =========================================================

LABEL_MAP = {
    "healthy": 0,
    "severe": 1
}


# =========================================================
# UTILS
# =========================================================

def softmax_from_logits(logit0, logit1):

    exp0 = np.exp(logit0)
    exp1 = np.exp(logit1)

    s = exp0 + exp1

    return exp0 / s, exp1 / s


def mean_std_str(values, decimals=4):

    values = np.array(values)

    mean = values.mean()
    std = values.std()

    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# =========================================================
# AGREGAÇÃO DE UMA IMAGEM
# =========================================================

def aggregate_group(group, orig_pred):

    # -----------------------------------------------------
    # MÉDIA DAS PROBABILIDADES
    # -----------------------------------------------------

    mean_prob_0 = group["prob_0"].mean()
    mean_prob_1 = group["prob_1"].mean()

    std_prob_0 = group["prob_0"].std()
    std_prob_1 = group["prob_1"].std()

    mean_prob_pred = int(mean_prob_1 > mean_prob_0)

    # -----------------------------------------------------
    # MÉDIA DOS LOGITS
    # -----------------------------------------------------

    mean_logit_0 = group["logit_0"].mean()
    mean_logit_1 = group["logit_1"].mean()

    std_logit_0 = group["logit_0"].std()
    std_logit_1 = group["logit_1"].std()

    _, softmax_1 = softmax_from_logits(
        mean_logit_0,
        mean_logit_1
    )

    mean_logit_pred = int(softmax_1 > 0.5)

    # -----------------------------------------------------
    # VOTAÇÃO
    # -----------------------------------------------------

    votes_for_0 = (group["pred"] == 0).sum()
    votes_for_1 = (group["pred"] == 1).sum()

    # desempate usando baseline
    if votes_for_0 == votes_for_1:
        vote_pred = orig_pred
    else:
        vote_pred = int(votes_for_1 > votes_for_0)

    return {

        # probs
        "mean_prob_0": mean_prob_0,
        "std_prob_0": std_prob_0,

        "mean_prob_1": mean_prob_1,
        "std_prob_1": std_prob_1,

        # logits
        "mean_logit_0": mean_logit_0,
        "std_logit_0": std_logit_0,

        "mean_logit_1": mean_logit_1,
        "std_logit_1": std_logit_1,

        # votos
        "votes_for_0": votes_for_0,
        "votes_for_1": votes_for_1,

        # preds
        "mean_prob_pred": mean_prob_pred,
        "mean_logit_pred": mean_logit_pred,
        "vote_pred": vote_pred,
    }


# =========================================================
# AGREGAÇÃO PRINCIPAL
# =========================================================

def aggregate_tta_results(arch_path):

    csv_path = os.path.join(
        arch_path,
        "results.csv"
    )

    df = pd.read_csv(csv_path)

    # baseline
    orig_df = df[df["tta_type"] == "none"]

    tta_types = sorted([
        x for x in df["tta_type"].unique()
        if x != "none"
    ])

    results = []

    # =====================================================
    # ESTRATÉGIAS
    # =====================================================

    strategies = [["none"]]

    # none + TTA individual
    for tta in tta_types:
        strategies.append(["none", tta])

    # -----------------------------------------------------
    # TODAS AS COMBINAÇÕES
    # descomente se quiser
    # -----------------------------------------------------

    # for r in range(1, len(tta_types)+1):
    #     for comb in combinations(tta_types, r):
    #         strategies.append(["none", *comb])

    # =====================================================
    # AGREGA
    # =====================================================

    for strategy in strategies:

        strategy_name = "+".join(strategy)

        strategy_df = df[
            df["tta_type"].isin(strategy)
        ]

        grouped = strategy_df.groupby(
            ["image_id", "label"]
        )

        for (image_id, label), group in grouped:

            true_label = LABEL_MAP[label]

            orig_pred = orig_df[
                (orig_df["image_id"] == image_id) &
                (orig_df["label"] == label)
            ]["pred"].iloc[0]

            agg = aggregate_group(
                group,
                orig_pred
            )

            row = {

                "tta_strategy": strategy_name,

                "image_id": image_id,
                "label": label,

                "true_label": true_label,
                "orig_pred": orig_pred,

                **agg
            }

            results.append(row)

    agg_df = pd.DataFrame(results)

    save_path = os.path.join(
        arch_path,
        "aggregated_results.csv"
    )

    agg_df.to_csv(save_path, index=False)

    print(f"Saved: {save_path}")

    return agg_df


# =========================================================
# AVALIAÇÃO
# =========================================================

def evaluate_tta(arch_path):

    csv_path = os.path.join(
        arch_path,
        "aggregated_results.csv"
    )

    df = pd.read_csv(csv_path)

    methods = [
        "mean_prob_pred",
        "mean_logit_pred",
        "vote_pred"
    ]

    results = []

    # =====================================================
    # POR ESTRATÉGIA
    # =====================================================

    for strategy in df["tta_strategy"].unique():

        strategy_df = df[
            df["tta_strategy"] == strategy
        ]

        y_true = strategy_df["true_label"]

        baseline = strategy_df["orig_pred"]

        for method in methods:

            y_pred = strategy_df[method]

            # -------------------------------------------------
            # métricas principais
            # -------------------------------------------------

            acc = accuracy_score(
                y_true,
                y_pred
            )

            rec = recall_score(
                y_true,
                y_pred
            )

            f1 = f1_score(
                y_true,
                y_pred,
                average="macro"
            )

            # -------------------------------------------------
            # comparação com baseline
            # -------------------------------------------------

            corrected = (
                (baseline != y_true) &
                (y_pred == y_true)
            ).sum()

            corrupted = (
                (baseline == y_true) &
                (y_pred != y_true)
            ).sum()

            total = len(strategy_df)

            corrected_pct = corrected / total

            corrupted_pct = corrupted / total

            net_gain = (
                corrected_pct -
                corrupted_pct
            )

            # -------------------------------------------------
            # estabilidade
            # -------------------------------------------------

            prob_std = (
                strategy_df["std_prob_1"]
                .fillna(0)
                .mean()
            )

            logit_std = (
                strategy_df["std_logit_1"]
                .fillna(0)
                .mean()
            )

            results.append({

                "tta_strategy": strategy,

                "method": method,

                "accuracy": acc,
                "recall": rec,
                "f1": f1,

                "corrected": corrected,
                "corrupted": corrupted,

                "corrected_pct": corrected_pct,
                "corrupted_pct": corrupted_pct,

                "net_gain": net_gain,

                "avg_prob_std": prob_std,
                "avg_logit_std": logit_std
            })

    eval_df = pd.DataFrame(results)

    save_path = os.path.join(
        arch_path,
        "aggregated_evaluation.csv"
    )

    eval_df.to_csv(save_path, index=False)

    print(f"Saved: {save_path}")

    return eval_df


# =========================================================
# AGREGA ENTRE SEEDS
# =========================================================

def summarize_across_seeds(
    root_dir,
    architectures,
    seeds
):

    for architecture in architectures:

        final_results = []

        arch_seed_results = []

        # =====================================================
        # CARREGA RESULTADOS DE TODAS AS SEEDS
        # =====================================================

        for seed in seeds:

            eval_path = os.path.join(
                root_dir,
                architecture,
                "seeds",
                str(seed),
                "aggregated_evaluation.csv"
            )

            df = pd.read_csv(eval_path)

            df["seed"] = seed

            arch_seed_results.append(df)

        all_df = pd.concat(
            arch_seed_results,
            ignore_index=True
        )

        # =====================================================
        # AGRUPA
        # =====================================================

        grouped = all_df.groupby(
            ["tta_strategy", "method"]
        )

        for (
            strategy,
            method,
        ), group in grouped:

            row = {

                "architecture": architecture,

                "tta_strategy": strategy,
                "method": method,

                "accuracy": mean_std_str(
                    group["accuracy"]
                ),

                "recall": mean_std_str(
                    group["recall"]
                ),

                "f1": mean_std_str(
                    group["f1"]
                ),

                "corrected_pct": mean_std_str(
                    group["corrected_pct"]
                ),

                "corrupted_pct": mean_std_str(
                    group["corrupted_pct"]
                ),

                "net_gain": mean_std_str(
                    group["net_gain"]
                ),
            }

            final_results.append(row)

        # =====================================================
        # SALVA CSV DA ARQUITETURA
        # =====================================================

        summary_df = pd.DataFrame(final_results)

        save_path = os.path.join(
            root_dir,
            architecture,
            "tta_summary_across_seeds.csv"
        )

        summary_df.to_csv(
            save_path,
            index=False
        )

        print(f"Saved summary: {save_path}")

    return


# =========================================================
# EXECUÇÃO
# =========================================================

seeds = [7, 12, 42, 65, 87, 93, 107, 121]

architectures = [
    "efficientnet_b0_originais",
    "mobilenet_originais",
    "efficientnet_b0_originais_T_A",
    "efficientnet_b0_originais_T_C",
    "efficientnet_b0_originais_T_D",
    "mobilenet_originais_T_A",
    "mobilenet_originais_T_C",
    "mobilenet_originais_T_D"
]

ROOT_DIR = "TTA/tta_outputs_v2"

# =========================================================
# GERA RESULTADOS POR SEED
# =========================================================

for architecture in architectures:

    for seed in seeds:

        ARCH_PATH = os.path.join(
            ROOT_DIR,
            architecture,
            "seeds",
            str(seed)
        )

        aggregate_tta_results(
            ARCH_PATH
        )

        evaluate_tta(
            ARCH_PATH
        )

# =========================================================
# RESUMO FINAL ENTRE SEEDS
# =========================================================

summary_df = summarize_across_seeds(
    ROOT_DIR,
    architectures,
    seeds
)
