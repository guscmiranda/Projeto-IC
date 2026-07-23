# Projeto IC - Ensemble Fractal & Test-Time Augmentation (TTA)

Este repositório contém o código-fonte desenvolvido para o projeto de Iniciação Científica (IC), focado em treinamento de arquiteturas baseadas em **Ensemble Fractal** e aplicação de técnicas de **Test-Time Augmentation (TTA)** para inferência e explicabilidade (Grad-CAM).

---

## 📁 Estrutura do Repositório

```text
.
├── EnsembleFractal/          # Módulo de treinamento e modelos
│   ├── dataset.py            # Carregamento e pré-processamento dos dados
│   ├── metrics.py            # Cálculo de métricas de avaliação
│   ├── model.py              # Arquitetura dos modelos
│   ├── train.py              # Estrutura do loop de treinamento
│   ├── run.py                # Script principal de execução do treino
│   ├── utils.py              # Funções utilitárias de suporte
│   └── __init__.py
│
└── TTA/                      # Módulo de Test-Time Augmentation e Inferência
    ├── apply_tta.py          # Gera dataset de inferência e roda testes nos modelos
    ├── aggregation.py        # Calcula e avalia resultados por estratégia de agregação
    ├── gradcam_utils.py      # Utilitários para geração dos mapas de ativação (Grad-CAM)
    └── limpa_csv.py          # Formata dados em CSV para análise da agregação por votação

```

---

## 🚀 Módulos do Projeto

### 1. `EnsembleFractal/` — Treinamento

Responsável pela definição da rede e execução do pipeline de treinamento dos modelos.

* **`run.py`**: Ponto de entrada para iniciar o fluxo de treinamento.
* **`train.py`** & **`model.py`**: Implementam a lógica de otimização, validação e construção da rede.
* **`dataset.py`**: Gerencia a ingestão e preparação dos dados de treino/validação.

### 2. `TTA/` — Inferência, Agregação e Explicabilidade

Aplica estratégias de Test-Time Augmentation para melhorar a robustez das predições em tempo de teste.

* **`apply_tta.py`**: Executa os testes em todos os modelos e gera o dataset de inferência.
> ⚠️ **Aviso de Desempenho:** É altamente recomendado comentar a chamada do **Grad-CAM** neste script durante execuções de rotina para otimizar o tempo de processamento.


* **`aggregation.py`**: Processa e consolida as predições de acordo com diferentes estratégias de agregação do TTA.
* **`gradcam_utils.py`**: Funções de suporte para interpretação do modelo através de visualizações Grad-CAM.
* **`limpa_csv.py`**: Trata os arquivos de saída para melhorar a visualização e análise das estratégias baseadas em votação.

---

## 🛠️ Como Executar

### 1. Treinamento do Modelo

Para iniciar a fase de treinamento:

```bash
python EnsembleFractal/run.py

```

### 2. Aplicação do TTA e Inferência

Para rodar os testes e gerar os dados de inferência:

```bash
python TTA/apply_tta.py

```

### 3. Agregação dos Resultados

Para processar e formatar as métricas de agregação:

```bash
python TTA/aggregation.py
python TTA/limpa_csv.py

```