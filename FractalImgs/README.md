# Fractal Descriptors and Recurrence Plots

Este projeto contém os scripts utilizados para a **extração de atributos fractais a partir de imagens** e sua posterior conversão em imagens de **Recurrence Plot (RP)**.

O processamento é dividido em duas etapas principais:

```text
Imagens originais
       ↓
Extração dos atributos fractais
       ↓
CSV de atributos
       ↓
Conversão dos atributos
       ↓
Recurrence Plot RGB
```

A primeira etapa é responsável pelo cálculo dos atributos de **Percolação (PERC), Lacunaridade (LAC) e Dimensão Fractal (DF)** e pela organização desses resultados em arquivos CSV.

A segunda etapa utiliza o CSV gerado para construir imagens RGB de Recurrence Plot a partir dos atributos selecionados.

---

# Quick Start

Se o objetivo é apenas **reproduzir o processo e obter as imagens de Recurrence Plot**, são necessárias duas etapas.

## 1. Gerar o CSV de atributos

Coloque as imagens que serão processadas em um diretório, por exemplo:

```text
imagens/
├── imagem_01.png
├── imagem_02.png
├── imagem_03.png
└── ...
```

Execute:

```bash
python saveCSVPercCLACDF3Distances.py ./imagens ./features true
```

O comando recebe:

```text
<diretorio_origem> <diretorio_destino> <All_FEATURES>
```

O valor `true` indica que serão calculados os atributos de:

```text
PERC + LAC + DF
```

Ao final, serão gerados:

```text
features/
├── result_aca_md_v.csv
└── result_aca_md.csv
```

## 2. Gerar as imagens de Recurrence Plot

Utilize o CSV gerado na etapa anterior:

```bash
python create_imgs.py ./features/result_aca_md_v.csv ./dataset recplot
```

As imagens serão salvas em:

```text
dataset/
└── F-RecPlot/
    ├── F-RecPlot1.png
    ├── F-RecPlot2.png
    ├── F-RecPlot3.png
    └── ...
```

O resultado final é uma coleção de imagens **PNG RGB de 100 × 100 pixels**, produzidas a partir dos atributos fractais extraídos das imagens originais.

> **Se você só precisa reproduzir o resultado, os dois comandos acima são suficientes. As seções seguintes explicam detalhadamente o que acontece em cada etapa.**

---

# 1. Estrutura do projeto

O processo está dividido em dois conjuntos de scripts:

```text
fractal_descriptors_and_plots_CNN/
│
├── extract_fractal_features/
│   ├── saveCSVPercCLACDF3Distances.py
│   ├── ScriptPerc.py
│   ├── ScriptLACDF3Distances.py
│   ├── util.py
│   ├── pmr.py
│   ├── pmrEucl.py
│   ├── pmrManh.py
│   ├── clustperc.py
│   ├── clustpercEucl.py
│   ├── clustpercManh.py
│   ├── lacunaridade.py
│   ├── N.py
│   └── ...
│
└── feature_to_image/
    ├── create_imgs.py
    ├── reshapeRecPlot.py
    ├── create_recorrence_plot.py
    ├── reshapeClassical.py
    └── ...
```

A pasta `extract_fractal_features` concentra a etapa de **extração dos atributos e geração do CSV**.

A pasta `feature_to_image` concentra a etapa de **conversão dos atributos em imagens**.

---

# 2. Requisitos

O código utiliza bibliotecas Python para processamento de imagens, manipulação de dados, cálculo numérico, regressão e geração das imagens.

Entre as principais dependências estão:

```text
numpy
pandas
Pillow
scipy
scikit-learn
numba
imageio
```

A instalação pode ser realizada com:

```bash
pip install numpy pandas Pillow scipy scikit-learn numba imageio
```

Além das bibliotecas externas, os módulos próprios do projeto precisam estar disponíveis no ambiente de execução.

Entre eles estão:

```text
clustperc
clustpercEucl
clustpercManh
pmr
pmrEucl
pmrManh
lacunaridade
N
create_recorrence_plot
reshapeRecPlot
reshapeClassical
```

---

# 3. Entrada: imagens originais

A primeira etapa recebe um diretório contendo as imagens que serão analisadas.

Os scripts de extração procuram arquivos com as extensões:

```text
.png
.tif
.jpg
```

Por exemplo:

```text
imagens/
├── imagem_01.png
├── imagem_02.png
├── imagem_03.jpg
└── imagem_04.tif
```

As imagens são carregadas utilizando o `Pillow`.

Antes do cálculo dos atributos, cada imagem é redimensionada para:

```text
224 × 224 pixels
```

utilizando interpolação bilinear:

```python
img_pil = img_pil.resize((224, 224), Image.BILINEAR)
```

Em seguida, a imagem é convertida para um array NumPy:

```python
PIC = np.array(img_pil)
```

Esse array é utilizado como entrada para os métodos de extração.

---

# 4. Extração dos atributos

A geração do CSV é coordenada pelo script:

```text
saveCSVPercCLACDF3Distances.py
```

A função principal é:

```python
saveCSVPercCLACDF3Distances(
    origem,
    destino,
    All_FEATURES=True
)
```

Quando `All_FEATURES=True`, o script executa duas etapas de extração:

```python
resultado_perc = scriptPerc(origem)
resultado_LACDF = scriptLACDF3Distances(origem)
```

Assim, os atributos são obtidos por meio de:

```text
scriptPerc()
        ↓
      PERC

scriptLACDF3Distances()
        ↓
     LAC + DF
```

Os resultados das duas etapas são posteriormente combinados.

---

# 5. Atributos de Percolação (PERC)

A função `scriptPerc()` é responsável pela extração dos atributos relacionados à **percolação**.

Para cada imagem, são executados os métodos correspondentes às três métricas de distância:

```python
Minsk_perc = clustperc_jit(PIC, maxr)
Eucl_perc = clustpercEucl_jit(PIC, maxr)
Manh_perc = clustpercManh_jit(PIC, maxr)
```

Os resultados são combinados:

```python
resultado_parc = {
    **Minsk_perc,
    **Eucl_perc,
    **Manh_perc
}
```

Portanto, a etapa de PERC considera três configurações de distância:

```text
Minkowski/Chessboard
Euclidiana
Manhattan
```

O parâmetro utilizado para o maior raio é:

```python
maxr = 41
```

---

# 6. Atributos de Lacunaridade (LAC) e Dimensão Fractal (DF)

A função `scriptLACDF3Distances()` é responsável pelo cálculo dos atributos de **lacunaridade** e **dimensão fractal**.

Para cada uma das três métricas de distância, é calculada uma matriz de probabilidades por meio dos respectivos módulos:

```text
pmr.py       → Minkowski/Chessboard
pmrEucl.py   → Euclidiana
pmrManh.py   → Manhattan
```

Por exemplo:

```python
MatrizProb = pmr(PIC, maxr)
```

Depois, a matriz é utilizada para calcular o vetor de lacunaridade:

```python
MinkLAC = lacunaridade(MatrizProb)
```

O mesmo processo é realizado para as distâncias Euclidiana e Manhattan.

---

# 7. Métricas de distância

O processo utiliza três métricas de distância.

## Minkowski / Chessboard

A implementação correspondente é utilizada por:

```text
pmr.py
clustperc.py
```

e seus respectivos métodos de cálculo.

## Euclidiana

A implementação correspondente é utilizada por:

```text
pmrEucl.py
clustpercEucl.py
```

## Manhattan

A implementação correspondente é utilizada por:

```text
pmrManh.py
clustpercManh.py
```

Dessa forma, os atributos são calculados considerando diferentes formas de medir distância entre os elementos da imagem.

---

# 8. Escalas utilizadas

O código define:

```python
maxr = 41
```

Para os cálculos de LAC e DF, são utilizadas as escalas:

```python
r = list(range(3, maxr + 1, 2))
```

O resultado é:

```text
3, 5, 7, 9, 11, 13, 15, 17, 19, 21,
23, 25, 27, 29, 31, 33, 35, 37, 39, 41
```

Total:

```text
20 escalas
```

Os atributos que resultam em vetores ao longo dessas escalas são posteriormente expandidos para colunas individuais no CSV.

Por exemplo:

```text
MinkLAC
```

é convertido em:

```text
MinkLAC1
MinkLAC2
...
MinkLAC20
```

---

# 9. Características de LAC

Para cada métrica de distância, o código calcula um vetor de lacunaridade.

São produzidos:

```text
MinkLAC
EuclLAC
ManhLAC
```

Além dos vetores, são calculadas características derivadas:

```text
AreaLAC
SkewnessLAC
AreaRatioLAC
MaxLAC
MaxLACIndex
```

Essas características são armazenadas separadamente para cada métrica de distância.

Por exemplo:

```text
MinkAreaLAC
MinkSkewnessLAC
MinkAreaRatioLAC
MinkMaxLAC
MinkMaxLACIndex
```

e, de maneira equivalente:

```text
EuclAreaLAC
EuclSkewnessLAC
EuclAreaRatioLAC
EuclMaxLAC
EuclMaxLACIndex
```

```text
ManhAreaLAC
ManhSkewnessLAC
ManhAreaRatioLAC
ManhMaxLAC
ManhMaxLACIndex
```

---

# 10. Dimensão Fractal

A dimensão fractal é obtida a partir dos valores de `nn`.

Para cada distância existe um vetor correspondente:

```text
Minknn
Euclnn
Manhnn
```

Para calcular a dimensão fractal, o código utiliza:

```python
x = np.log(r)
y = -np.log(Minknn)
```

e realiza um ajuste utilizando `HuberRegressor`:

```python
modelo = HuberRegressor()
modelo.fit(X, y)
```

O coeficiente do modelo é armazenado como:

```text
MinkDF
EuclDF
ManhDF
```

Assim, cada imagem possui uma dimensão fractal associada a cada uma das três métricas.

---

# 11. Combinação dos atributos

Quando `All_FEATURES=True`, os resultados de PERC e LAC/DF são combinados para cada imagem:

```python
combinado = {**d_perc, **d_lacdf}
```

Os registros são então transformados em um `DataFrame`:

```python
df = pd.DataFrame(resultado)
```

Antes da reorganização, os valores que são arrays NumPy são convertidos para listas:

```python
for col in df.columns:
    df[col] = df[col].apply(
        lambda x: x.tolist()
        if isinstance(x, np.ndarray)
        else x
    )
```

---

# 12. Organização do CSV

A função:

```text
reorganizar_e_expandir_df()
```

é responsável por organizar os atributos em uma ordem definida e expandir os atributos vetoriais.

Entre os atributos vetoriais processados estão:

```text
Minkp
Minkg
Minkh
MinkLAC
Minknn

Euclp
Euclg
Euclh
EuclLAC
Euclnn

Manhp
Manhg
Manhh
ManhLAC
Manhnn
```

Um vetor com 20 elementos, por exemplo:

```text
MinkLAC
```

é transformado em:

```text
MinkLAC1
MinkLAC2
...
MinkLAC20
```

Os atributos escalares permanecem como colunas individuais.

---

# 13. Arquivos CSV gerados

Ao final da etapa de extração, são salvos dois arquivos:

```text
result_aca_md_v.csv
result_aca_md.csv
```

O primeiro é salvo utilizando vírgula como separador:

```python
df.to_csv(
    caminho_csv_final_v,
    index=False,
    sep=','
)
```

O segundo utiliza ponto e vírgula:

```python
df.to_csv(
    caminho_csv_final,
    index=False,
    sep=';',
    decimal='.'
)
```

Assim, após a execução:

```bash
python saveCSVPercCLACDF3Distances.py ./imagens ./features true
```

a estrutura esperada é:

```text
features/
├── result_aca_md_v.csv
└── result_aca_md.csv
```

Para a etapa de geração do Recurrence Plot, utiliza-se o CSV compatível com o formato esperado pelo `create_imgs.py`.

---

# 14. Conversão dos atributos em Recurrence Plot

A segunda etapa é realizada pela pasta:

```text
feature_to_image/
```

O script principal é:

```text
create_imgs.py
```

Ele recebe o CSV, identifica as colunas de atributos e as transforma em uma matriz NumPy:

```python
features = df[feature_cols].to_numpy(dtype=np.float64)
```

Quando `relative_path` e `image_name` estão presentes, essas informações são tratadas separadamente para preservar a organização dos arquivos de saída.

Em seguida, para o modo `recplot`, o script chama:

```python
reshapeRecPlot(
    destino_base,
    features,
    out_paths=rec_out_paths
)
```

---

# 15. Organização dos 300 atributos

O `reshapeRecPlot.py` trabalha com **300 atributos por imagem**.

Esses atributos são divididos em três grupos de 100:

```text
Atributos 1–100       → Canal R
Atributos 101–200     → Canal G
Atributos 201–300     → Canal B
```

Essa divisão é realizada diretamente no código:

```python
featuresSplit[:,:,0] = features[:, 0:100]
featuresSplit[:,:,1] = features[:, 100:200]
featuresSplit[:,:,2] = features[:, 200:300]
```

O resultado é uma estrutura:

```text
n imagens × 100 atributos × 3 canais
```

ou:

```text
(n, 100, 3)
```

### Observação

A composição exata das 300 colunas utilizadas nessa etapa depende da ordem dos atributos presente no CSV produzido pela etapa de extração.

O `reshapeRecPlot.py` pressupõe que as 300 colunas utilizadas como entrada estejam organizadas na ordem esperada.

---

# 16. Normalização dos atributos

Antes da criação dos Recurrence Plots, os valores são normalizados para o intervalo `[0, 1]`.

A função utilizada é:

```python
mat2gray()
```

A normalização segue:

```text
x' = (x - min(x)) / (max(x) - min(x))
```

Caso todos os valores sejam iguais, a função retorna zeros para evitar divisão por zero.

Os 100 atributos de cada canal são processados em cinco blocos:

```text
0–19
20–39
40–59
60–79
80–99
```

Cada bloco possui:

```text
20 atributos
```

totalizando:

```text
5 × 20 = 100 atributos por canal
```

---

# 17. Geração dos Recurrence Plots

Depois da organização e normalização, os atributos de cada canal são tratados como um sinal.

Para cada imagem:

```python
signal_r = new_features[i, :, 0].reshape(-1, 1)
signal_g = new_features[i, :, 1].reshape(-1, 1)
signal_b = new_features[i, :, 2].reshape(-1, 1)
```

Cada sinal possui:

```text
100 valores
```

Cada sinal é então convertido individualmente em um Recurrence Plot:

```python
r_channel = create_recorrence_plot(signal_r)
g_channel = create_recorrence_plot(signal_g)
b_channel = create_recorrence_plot(signal_b)
```

Portanto:

```text
100 valores do canal R → RP R
100 valores do canal G → RP G
100 valores do canal B → RP B
```

---

# 18. Construção da imagem RGB

Os três Recurrence Plots são utilizados como os três canais de uma imagem RGB:

```python
imgs[i,:,:,0] = r_channel
imgs[i,:,:,1] = g_channel
imgs[i,:,:,2] = b_channel
```

A estrutura final é:

```text
                 Imagem RGB
                 100 × 100
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
    Canal R       Canal G       Canal B
    100 × 100     100 × 100     100 × 100
       │             │             │
       ▼             ▼             ▼
      RP R          RP G          RP B
```

Os valores são limitados ao intervalo `[0,1]`:

```python
imgs[i,:,:,:] = np.clip(
    imgs[i,:,:,:],
    0,
    1
)
```

Antes do salvamento, são convertidos para `uint8`:

```python
img_uint8 = (img * 255).astype(np.uint8)
```

Assim, a imagem PNG final possui valores de pixel no intervalo:

```text
0–255
```

---

# 19. Organização das imagens de saída

Quando o CSV contém as colunas:

```text
relative_path
image_name
```

o `create_imgs.py` utiliza essas informações para preservar a hierarquia dos arquivos.

Por exemplo, caso o CSV contenha imagens pertencentes às classes:

```text
healthy/
severe/
```

a saída pode manter essa organização:

```text
dataset/
└── F-RecPlot/
    ├── healthy/
    │   ├── imagem_01.png
    │   └── imagem_02.png
    │
    └── severe/
        ├── imagem_03.png
        └── imagem_04.png
```

Quando essas informações não estão disponíveis, o script salva os arquivos com nomes sequenciais:

```text
F-RecPlot1.png
F-RecPlot2.png
F-RecPlot3.png
...
```

---

# 20. Pipeline completo

O processo completo pode ser representado da seguinte maneira:

```text
┌─────────────────────────────┐
│      Imagens originais      │
│       PNG / TIF / JPG       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Redimensionamento           │
│        224 × 224            │
└──────────────┬──────────────┘
               │
               ▼
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌──────────────┐ ┌────────────────────┐
│ scriptPerc() │ │scriptLACDF3...()   │
│              │ │                    │
│    PERC      │ │     LAC + DF       │
└──────┬───────┘ └─────────┬──────────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
┌─────────────────────────────┐
│ Combinação dos resultados   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Reorganização e expansão    │
│       dos atributos         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       CSV de features       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       create_imgs.py        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       300 atributos         │
│                             │
│   100 R + 100 G + 100 B     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       Normalização          │
└──────────────┬──────────────┘
               │
               ▼
       ┌───────┼───────┐
       ▼       ▼       ▼
      RP R    RP G    RP B
       │       │       │
       └───────┼───────┘
               │
               ▼
┌─────────────────────────────┐
│     Imagem RGB 100 × 100    │
│            PNG              │
└─────────────────────────────┘
```

---

# 21. Scripts principais

| Script                           | Responsabilidade                                                   |
| -------------------------------- | ------------------------------------------------------------------ |
| `saveCSVPercCLACDF3Distances.py` | Coordena a extração e salva os CSVs                                |
| `ScriptPerc.py`                  | Calcula os atributos de Percolação (PERC)                          |
| `ScriptLACDF3Distances.py`       | Calcula os atributos de Lacunaridade (LAC) e Dimensão Fractal (DF) |
| `util.py`                        | Reorganiza e expande os atributos para o formato tabular           |
| `create_imgs.py`                 | Lê o CSV e coordena a geração das imagens                          |
| `reshapeRecPlot.py`              | Organiza os 300 atributos e constrói os três canais                |
| `create_recorrence_plot.py`      | Gera o Recurrence Plot a partir de cada sinal                      |

---

# 22. Modos disponíveis no `create_imgs.py`

O script `create_imgs.py` aceita um terceiro argumento que define o tipo de imagem a ser gerado:

```text
recplot
classical
both
```

Para gerar somente Recurrence Plots:

```bash
python create_imgs.py ./features/result_aca_md_v.csv ./dataset recplot
```

Para gerar somente a representação clássica:

```bash
python create_imgs.py ./features/result_aca_md_v.csv ./dataset classical
```

Para gerar ambas:

```bash
python create_imgs.py ./features/result_aca_md_v.csv ./dataset both
```

Quando o terceiro argumento não é informado, o modo padrão é:

```text
both
```

Este README está focado especificamente no fluxo de geração de **Recurrence Plots**.

---

# 23. Resumo

O processo de geração das imagens de Recurrence Plot pode ser resumido em:

```text
Imagem original
      ↓
224 × 224
      ↓
PERC + LAC + DF
      ↓
CSV
      ↓
300 atributos
      ↓
3 grupos de 100
      ↓
Normalização
      ↓
3 sinais
      ↓
3 Recurrence Plots
      ↓
3 canais RGB
      ↓
PNG 100 × 100
```

Assim, cada imagem original dá origem a uma imagem RGB de Recurrence Plot construída a partir dos atributos fractais extraídos durante a primeira etapa do processamento.
