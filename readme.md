# Regressão Linear com Dados de Energia Solar (PVGIS)

**Grupo / Aluno(s):**  Renato sandreschi rm 569156, Andre shiguematsu rm569158, Guilherme belo rm570079, Conrado gracie rm569157


Exercício de preparação de Machine Learning: obter dados horários reais de geração fotovoltaica pela API pública do PVGIS, organizá-los em um DataFrame, treinar dois modelos de Regressão Linear e compará-los com MAE, MSE e R².

Notebook: `Regressao_Linear_Energia_Solar_Basica.ipynb` (feito para rodar no Google Colab)

---

## 1. Fonte dos dados

| Item | Informação |
|---|---|
| Fonte | PVGIS (Photovoltaic Geographical Information System), Joint Research Centre, Comissão Europeia |
| Recurso | Dados horários (*Hourly Data*), endpoint `seriescalc`, versão `v5_3` |
| Cidade | Campinas, SP |
| Latitude / Longitude | -22,9056 / -47,0608 |
| Período consultado | 2019 a 2020 (01/01/2019 a 31/12/2020) |
| Status da consulta | 200 (sucesso) |

**Endereço utilizado na requisição:**

```
https://re.jrc.ec.europa.eu/api/v5_3/seriescalc?lat=-22.9056&lon=-47.0608&startyear=2019&endyear=2020&pvcalculation=1&peakpower=1&loss=14&optimalangles=1&outputformat=json
```

| Parâmetro | Valor | Significado |
|---|---|---|
| `lat`, `lon` | -22.9056, -47.0608 | localização |
| `startyear`, `endyear` | 2019, 2020 | período |
| `pvcalculation` | 1 | inclui o cálculo da potência fotovoltaica (`P`) |
| `peakpower` | 1 | sistema de 1 kWp, então `P` é dada em W |
| `loss` | 14 | perdas do sistema, em % |
| `optimalangles` | 1 | o PVGIS calcula inclinação e azimute ótimos |
| `outputformat` | json | formato da resposta |

A resposta JSON tem três blocos: `inputs`, `outputs` (a série horária está em `outputs.hourly`) e `meta`. Exemplo de registro bruto:

```
{'time': '20190101:0003', 'P': 0.0, 'G(i)': 0.0, 'H_sun': 0.0, 'T2m': 23.71, 'WS10m': 4.76, 'Int': 0.0}
```

## 2. Variáveis do DataFrame

A coluna `time` foi convertida para `datetime` (formato `%Y%m%d:%H%M`; os horários do PVGIS caem no minuto 03). A coluna `Int` (indicador de dado reconstruído) foi descartada.

| Coluna | Descrição | Unidade |
|---|---|---|
| `time` | data e hora | - |
| `P` | potência do sistema fotovoltaico (**variável alvo**) | W |
| `G(i)` | irradiância no plano dos painéis | W/m² |
| `H_sun` | altura do Sol | graus |
| `T2m` | temperatura do ar a 2 m | °C |
| `WS10m` | velocidade do vento a 10 m | m/s |

## 3. Inspeção inicial

- **Dimensão:** 17.544 linhas e 6 colunas (731 dias × 24 horas, pois 2020 foi bissexto)
- **Tipos:** `time` é `datetime64[ns]` e as demais colunas são `float64`
- **Valores ausentes:** nenhum, em todas as colunas

| Estatística | P (W) | G(i) (W/m²) | T2m (°C) | WS10m (m/s) | H_sun (°) |
|---|---|---|---|---|---|
| Média | 178,76 | 236,80 | 21,32 | 2,57 | 18,78 |
| Desvio padrão | 255,81 | 336,58 | 5,08 | 1,24 | 24,44 |
| Mínimo | 0,00 | 0,00 | 5,42 | 0,00 | 0,00 |
| Mediana (50%) | 0,00 | 0,00 | 21,13 | 2,34 | 0,00 |
| Máximo | 841,19 | 1113,63 | 38,07 | 7,38 | 89,70 |

A mediana de `P`, `G(i)` e `H_sun` é zero, o que mostra que mais da metade dos registros corresponde a períodos sem irradiância (noite).

### Decisão: remover os registros sem irradiância

Foram mantidos apenas os registros com `G(i) > 0`. Como o objetivo é estimar a potência **durante a geração solar**, as linhas noturnas, em que `G(i) = 0`, `H_sun = 0` e `P = 0`, não ajudam a explicar a geração. Elas tornariam a previsão trivial para metade dos dados, o que inflaria as métricas, e formariam um aglomerado em (0, 0) que distorceria o ajuste da reta.

| | Registros |
|---|---|
| Antes do filtro | 17.544 |
| Depois do filtro | 8.627 |
| Removidos | 8.917 (cerca de 50,8%) |

## 4. Definição do problema

- **Alvo (y):** `P`, a potência fotovoltaica
- **Entradas (X) do Modelo 1:** `G(i)`, `H_sun`, `T2m` e `WS10m`

Justificativa: a irradiância é a energia solar que chega ao painel, a altura do Sol se relaciona com o ângulo de incidência, a temperatura afeta a eficiência dos módulos e o vento ajuda a resfriá-los.

## 5. Visualização e correlação

O notebook traz o gráfico de dispersão **Irradiância × Potência Fotovoltaica** e a matriz de correlação (mapa de calor).

| | P | G(i) | H_sun | T2m | WS10m |
|---|---|---|---|---|---|
| **P** | 1,0000 | 0,9978 | 0,6937 | 0,2202 | 0,0353 |
| **G(i)** | 0,9978 | 1,0000 | 0,7090 | 0,2586 | 0,0108 |
| **H_sun** | 0,6937 | 0,7090 | 1,0000 | 0,4684 | 0,0755 |
| **T2m** | 0,2202 | 0,2586 | 0,4684 | 1,0000 | -0,1102 |
| **WS10m** | 0,0353 | 0,0108 | 0,0755 | -0,1102 | 1,0000 |

- **Maior correlação com `P`:** `G(i)`, com 0,9978, uma relação linear praticamente perfeita
- **Correlação moderada:** `H_sun`, com 0,69
- **Correlação baixa:** `T2m` (0,22) e `WS10m` (0,035, praticamente nula)
- `G(i)` e `H_sun` também são correlacionadas entre si (0,71), ou seja, trazem informação parcialmente redundante

## 6. Separação treino e teste

Divisão com `train_test_split(test_size=0.20, random_state=42)`:

| Conjunto | Dimensão (Modelo 1) |
|---|---|
| `X_train` | (6901, 4) |
| `X_test` | (1726, 4) |
| `y_train` | (6901,) |
| `y_test` | (1726,) |

O Modelo 2 usa o mesmo `random_state`, então os dois modelos são avaliados exatamente nas mesmas linhas de teste.

## 7. Modelos

| Modelo | Variáveis utilizadas |
|---|---|
| Modelo 1 (`modelo1`) | `G(i)`, `H_sun`, `T2m`, `WS10m` |
| Modelo 2 (`modelo2`) | `G(i)`, `T2m` |

O Modelo 2 foi criado para verificar se `H_sun` e `WS10m` acrescentam informação além da irradiância e da temperatura.

## 8. Resultados

| Modelo | Variáveis utilizadas | MAE (W) | MSE (W²) | R² |
|---|---|---|---|---|
| Modelo 1 | G(i), H_sun, T2m, WS10m | 9,8660 | 152,0427 | 0,9977 |
| Modelo 2 | G(i), T2m | 10,3919 | 177,8959 | 0,9973 |

Valores completos: Modelo 1 com MAE = 9,866027, MSE = 152,042744 e R² = 0,997654; Modelo 2 com MAE = 10,391927, MSE = 177,895905 e R² = 0,997255. A raiz do MSE (RMSE) é cerca de 12,33 W no Modelo 1 e 13,34 W no Modelo 2, contra uma potência máxima observada de 841,19 W.

## 9. Análise dos resultados

1. **Maior R²:** Modelo 1 (0,9977 contra 0,9973).
2. **Menor MAE:** Modelo 1 (9,87 W contra 10,39 W).
3. **Menor MSE:** Modelo 1 (152,04 contra 177,90).
4. **As três métricas apontam para o mesmo modelo?** Sim, todas indicam o Modelo 1 como melhor. A diferença é pequena (cerca de 0,5 W no MAE e 0,0004 no R²), porque `G(i)` sozinha já explica quase toda a variação da potência.
5. **Maior correlação implica melhor modelo?** Não necessariamente. `G(i)` tem correlação de 0,998 e sustenta os dois modelos. Mesmo assim, o Modelo 1 ficou um pouco melhor ao incluir `H_sun` e `WS10m`, sendo que `WS10m` tem correlação quase nula com `P` (0,035). A correlação mede o efeito de cada variável isoladamente. Em um modelo múltiplo, o que conta é a informação nova que cada variável traz, e isso não se enxerga só na matriz de correlação. Como os dois modelos diferem em duas variáveis ao mesmo tempo, não dá para dizer qual delas causou a melhora.
6. **Relações aproximadamente lineares?** A relação entre `G(i)` e `P` é praticamente linear, o que o coeficiente de 0,998 e o gráfico de dispersão indicam, e é o que explica R² próximo de 1. `H_sun` tem relação moderada, e `T2m` e `WS10m` têm relações fracas com a potência.
7. **Fatores que dificultam a previsão linear:**
   - a eficiência dos módulos cai com a temperatura e varia com a irradiância e o ângulo de incidência, o que não é rigorosamente linear;
   - interações e redundância entre variáveis, como `G(i)` e `H_sun` (correlação de 0,71) e `T2m` e `H_sun` (0,47);
   - nebulosidade e variações climáticas rápidas;
   - ausência de variáveis que influenciam a geração (umidade, sombreamento, sujeira dos painéis) e de informação de data e sazonalidade;
   - incertezas inerentes aos dados de satélite e reanálise.

## 10. Conclusão

O **Modelo 1** foi o melhor nas três métricas (maior R² e menores MAE e MSE) e é o escolhido. Ele erra em média cerca de 10 W em uma potência que chega a 841 W. A vantagem sobre o Modelo 2 é, porém, muito pequena, e o Modelo 2, mais simples, já tem desempenho muito próximo. Isso mostra que a irradiância `G(i)` é, de longe, o fator mais importante para estimar a potência fotovoltaica.

**Limitações:** os resultados valem para Campinas-SP e para os anos de 2019 e 2020. Como as linhas sem irradiância foram removidas, o modelo não deve ser usado para prever a potência em períodos noturnos. A divisão foi aleatória, e horas vizinhas são parecidas entre si, o que pode deixar as métricas um pouco otimistas.

## 11. Como executar

1. Abra o Google Colab e faça o upload de `Regressao_Linear_Energia_Solar_Basica.ipynb`.
2. Execute **Ambiente de execução → Executar tudo**. É necessário acesso à internet para consultar a API.
3. As bibliotecas usadas (`requests`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`) já vêm instaladas no Colab.

## Estrutura do notebook

1. Importação das bibliotecas
2. Consulta dos dados no PVGIS
3. Organização dos dados (DataFrame)
4. Análise inicial dos dados
5. Remoção dos registros sem geração
6. Definição das variáveis X e y
7. Visualização dos dados e matriz de correlação
8. Separação treino e teste
9. Modelo 1
10. Modelo 2
11. Comparação dos modelos
12. Conclusão

