# Projeto IC - Ensemble Fractal & Test-Time Augmentation (TTA)

Este repositório contém o código-fonte desenvolvido para o projeto de Iniciação Científica (IC), focado no treinamento de arquiteturas de redes neurais, aplicação de técnicas de **Test-Time Augmentation (TTA)** e combinação de modelos utilizando diferentes representações das imagens, incluindo a representação **F-RecPlot**.

O pipeline contempla o treinamento dos modelos, geração das imagens aumentadas, inferência, agregação dos resultados de TTA e formação de ensembles entre modelos treinados com imagens originais e imagens F-RecPlot.

---

## 📊 Dataset

O dataset utilizado no projeto é o **OralEpitheliumDB**, disponibilizado no repositório:

**[LIPAI-Org/OralEpitheliumDB_Dataset](https://github.com/LIPAI-Org/OralEpitheliumDB_Dataset)**

---

## 📁 Estrutura do Repositório

```text
.

├── Treinamento/                 # Módulo de treinamento e modelos
│   ├── dataset.py               # Carregamento e preparação dos dados
│   ├── metrics.py               # Cálculo de métricas de avaliação
│   ├── model.py                 # Construção e carregamento dos modelos
│   ├── train.py                 # Lógica de treinamento e validação
│   ├── run.py                   # Script principal de execução do treinamento
│   ├── utils.py                 # Funções e configurações auxiliares
│   └── __init__.py
│
└── TTA/                         # Módulo de TTA, inferência e ensembles
    ├── apply_tta.py             # Geração das imagens TTA e inferência
    ├── aggregation.py           # Agregação e avaliação dos resultados de TTA
    ├── gradcam_utils.py         # Funções auxiliares para Grad-CAM
    ├── limpa_csv.py             # Filtragem e organização dos resultados de TTA
    ├── ensemble.py              # Formação e avaliação dos ensembles
    └── limpa_ensembles.py       # Consolidação dos resultados entre seeds
```

---

## Módulos do Projeto

### 1. `Treinamento/` — Treinamento dos Modelos

Responsável pela definição dos modelos e execução do pipeline de treinamento.

* **`run.py`**: ponto de entrada para executar o fluxo de treinamento.

* **`train.py`**: implementa a lógica de treinamento, validação e execução do K-Fold.

* **`model.py`**: responsável pela construção e carregamento das arquiteturas utilizadas.

* **`dataset.py`**: gerencia o carregamento dos dados de treino e validação.

* **`utils.py`**: reúne configurações, transformações e funções auxiliares utilizadas pelo treinamento.

---

### 2. `TTA/` — Test-Time Augmentation e Inferência

Responsável pela geração das imagens aumentadas, execução dos modelos treinados e armazenamento das predições.

* **`apply_tta.py`**: gera as imagens utilizadas no TTA e executa a inferência dos modelos originais. São geradas múltiplas versões de cada imagem para cada estratégia de TTA. O resultado inicial é armazenado em `results.csv`.

> ⚠️ **Aviso de desempenho:** é recomendado manter o código de Grad-CAM comentado durante as execuções de rotina, caso a geração dos mapas de ativação não seja necessária. O `apply_tta.py` possui suporte ao Grad-CAM por meio do `gradcam_utils.py`.

* **`aggregation.py`**: processa os `results.csv` gerados pelo TTA. Para cada estratégia, calcula a média e o desvio padrão das probabilidades e logits, além da agregação por votação. Também calcula as métricas das diferentes formas de agregação.

* **`limpa_csv.py`**: filtra os resultados da agregação, mantendo atualmente os resultados referentes ao método `mean_prob_pred` e removendo informações que não são necessárias para a análise final.

---

### 3. `ensemble.py` — Ensemble

Responsável pela combinação dos modelos treinados com imagens originais e modelos treinados com a representação F-RecPlot.

O script:

1. Carrega os `aggregated_results.csv` dos modelos originais.
2. Seleciona as estratégias de TTA definidas para cada arquitetura.
3. Obtém as probabilidades agregadas do TTA.
4. Carrega os modelos F-RecPlot correspondentes a cada seed.
5. Executa a inferência nos dados F-RecPlot.
6. Combina os resultados dos modelos por média das probabilidades.
7. Calcula as métricas dos ensembles.
8. Salva as predições e métricas para cada seed.

Atualmente, as estratégias utilizadas no ensemble são:

```text
MobileNetV2:       none+T_F
EfficientNet-B0:  none+T_G
```

enquanto os modelos F-RecPlot são utilizados sem TTA adicional.

Os modelos são identificados no ensemble como:

```text
MO = MobileNetV2 Original

EO = EfficientNet-B0 Original

MR = MobileNetV2 F-RecPlot

ER = EfficientNet-B0 F-RecPlot
```

São avaliadas dez combinações:

```text
MO+MO
EO+EO
MR+MR
ER+ER
MO+EO
MR+ER
MO+MR
EO+ER
MO+ER
EO+MR
```

---

### 4. `limpa_ensembles.py` — Consolidação dos resultados

Responsável pela consolidação dos resultados obtidos para as diferentes seeds.

O script lê o `ensemble_metrics.csv`, agrupa os resultados por ensemble e calcula **média ± desvio padrão** para:

* Accuracy
* F1 Macro
* Precision Macro
* Recall Macro

O resultado final é salvo em:

```text
ensemble_metrics_across_seeds.csv
```

---

## 🛠️ Como Executar

O pipeline deve ser executado na seguinte ordem:

### 1. Treinamento dos modelos

```bash
python Treinamento/run.py
```

Essa etapa gera os modelos treinados que serão utilizados posteriormente pelo TTA e pelo ensemble.

### 2. Aplicação do TTA e inferência

```bash
python TTA/apply_tta.py
```

Gera as imagens aumentadas e executa a inferência dos modelos originais, produzindo os arquivos `results.csv`.

### 3. Agregação dos resultados do TTA

```bash
python TTA/aggregation.py
```

Processa os `results.csv` e gera os resultados agregados e avaliados, incluindo `aggregated_results.csv` e `aggregated_evaluation.csv`.

### 4. Organização dos resultados do TTA

```bash
python TTA/limpa_csv.py
```

Filtra os resultados para a análise baseada na média das probabilidades.

### 5. Formação dos ensembles

```bash
python TTA/ensemble.py
```

Combina os modelos originais com os modelos F-RecPlot e gera as métricas e predições dos ensembles.

### 6. Consolidação entre seeds

```bash
python TTA/limpa_ensembles.py
```

Gera o resultado final consolidado com média e desvio padrão das métricas entre as diferentes seeds.

---

## 🔄 Fluxo do Pipeline

```text
Treinamento/run.py
       │
       ▼
Modelos treinados
       │
       ▼
TTA/apply_tta.py
       │
       ▼
results.csv
       │
       ▼
TTA/aggregation.py
       │
       ├── aggregated_results.csv
       └── aggregated_evaluation.csv
       │
       ▼
TTA/limpa_csv.py
       │
       ▼
Resultados TTA organizados
       │
       ▼
TTA/ensemble.py
       │
       ▼
ensemble_metrics.csv
       │
       ▼
TTA/limpa_ensembles.py
       │
       ▼
ensemble_metrics_across_seeds.csv
```
