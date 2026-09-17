import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURAÇÕES
# ============================================================

INPUT_FILE = Path("TTA/ensemble_results_mean_prob_baseline/ensemble_metrics.csv")

OUTPUT_FILE = Path(
    "TTA/ensemble_results_mean_prob_baseline/ensemble_metrics_across_seeds.csv"
)


# ============================================================
# CARREGAR RESULTADOS
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Arquivo carregado:")
print(INPUT_FILE)

print(f"\nTotal de linhas: {len(df)}")
print(f"Total de ensembles: {df['ensemble'].nunique()}")
print(f"Seeds encontradas: {sorted(df['seed'].unique())}")


# ============================================================
# MÉTRICAS QUE SERÃO GENERALIZADAS
# ============================================================

metrics = [
    "accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro"
]


# ============================================================
# AGRUPAR POR ENSEMBLE
# ============================================================

grouped = (
    df
    .groupby("ensemble")[metrics]
    .agg(["mean", "std"])
)


# ============================================================
# ORGANIZAR RESULTADO
# ============================================================

result = pd.DataFrame()

# Mantém a ordem dos ensembles
ensemble_order = [
    "MO+MO",
    "EO+EO",
    "MR+MR",
    "ER+ER",
    "MO+EO",
    "MR+ER",
    "MO+MR",
    "EO+ER",
    "MO+ER",
    "EO+MR"
]

result["ensemble"] = ensemble_order


# ============================================================
# MÉDIA ± DESVIO PADRÃO
# ============================================================

for metric in metrics:

    means = grouped[metric]["mean"]
    stds = grouped[metric]["std"]

    result[metric] = [
        f"{means[ensemble]:.4f} ± {stds[ensemble]:.4f}"
        for ensemble in ensemble_order
    ]


# ============================================================
# SALVAR
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# EXIBIR
# ============================================================

print("\n" + "=" * 80)
print("RESULTADOS ACROSS SEEDS")
print("=" * 80)

print(
    result.to_string(index=False)
)

print("\nArquivo salvo em:")
print(OUTPUT_FILE)