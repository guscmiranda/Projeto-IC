import os
import pandas as pd

# =========================================================
# CONFIGURAÇÃO
# =========================================================

ROOT_DIR = "tta_outputs_v2"

architectures = [
    "efficientnet_b0_originais",
    "mobilenet_originais",
    "efficientnet_b0_originais_T_A",
    "efficientnet_b0_originais_T_C",
    "efficientnet_b0_originais_T_D",
    "mobilenet_originais_T_A",
    "mobilenet_originais_T_C",
    "mobilenet_originais_T_D",
]

# =========================================================
# LIMPA OS CSVs DE RESUMO
# =========================================================

for architecture in architectures:

    csv_path = os.path.join(
        ROOT_DIR,
        architecture,
        "tta_summary_across_seeds.csv"
    )

    if not os.path.exists(csv_path):
        print(f"Arquivo não encontrado: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    # Mantém apenas o método mean_prob_pred
    df = df[df["method"] == "mean_prob_pred"].copy()

    # Remove o prefixo "none+" da estratégia
    df["tta_strategy"] = (
        df["tta_strategy"]
        .str.replace("none+", "", regex=False)
    )

    # Remove colunas desnecessárias
    df = df.drop(
        columns=["architecture", "method"],
        errors="ignore"
    )

    # Salva o novo CSV
    save_path = os.path.join(
        ROOT_DIR,
        architecture,
        "tta_summary_mean_prob_pred.csv"
    )

    df.to_csv(save_path, index=False)

    print(f"Salvo: {save_path}")

print("\nConcluído!")