# Propuestas de Diagramas para el Documento

## Resumen
Este documento propone lugares donde un diagrama o figura mejoraría significativamente la comprensión del lector, evitando descripciones textuales extensas o ambiguas.

---

## 1. Diagrama de Arquitectura del Sistema (Obligatorio)

**Ubicación propuesta:** Sección "Methodology → System Architecture", justo después del primer párrafo de introducción.

**Descripción:** Un diagrama de flujo horizontal mostrando el pipeline completo:
```
[Raw Data] → [Ingestion] → [Enrichment Layer] → [Prompt Builder] → [Phi-4-mini] → [Action]
   ↓              ↓                ↓                  ↓              ↓
Yahoo CSVs    CSV Granular    FinBERT + Chronos   grpo_cot_v1      BUY/HOLD/SELL
CLEF dataset   Parquet                                          + Reward
```

**Por qué es necesario:** El texto actual describe 5 pasos en una lista numerada (enumerate). Un diagrama visual permitiría al lector entender el pipeline en 3 segundos en lugar de leer 5 párrafos. Es especialmente útil para reviewers de CEUR-WS que evalúan muchos papers.

---

## 2. Diagrama del Pipeline de Entrenamiento de Dos Fases

**Ubicación propuesta:** Sección "Methodology → Model Optimization and Training Setup", antes de describir Phase 1 y Phase 2.

**Descripción:** Un diagrama temporal vertical:
```
┌─────────────────────────────────────────┐
│  Phase 0: Dataset Construction (~1 min) │
│  1082 train / 896 val rows              │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Phase 1: SFT Warm-up (30 min)          │
│  • XML format anchoring                 │
│  • Ultra-concise completions (60-80 ch) │
│  • LR = 1.155e-5                        │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Phase 2: GRPO Training (6 min)         │
│  • Group size G=4                       │
│  • Asymmetric reward matrix             │
│  • Token rescue mechanism               │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Phase 3: Evaluation (Fast + Full)      │
│  • 100 rows (fast) / 896 rows (full)    │
└─────────────────────────────────────────┘
```

**Por qué es necesario:** El texto describe las fases pero el lector pierde la perspectiva temporal. El diagrama muestra que el entrenamiento total fue solo ~37 minutos, lo cual es un dato impactante que justifica el claim de "commodity hardware".

---

## 3. Diagrama de la Estructura del Prompt `grpo_cot_v1`

**Ubicación propuesta:** Sección "Methodology → Prompt Orchestration", después de la descripción del template.

**Descripción:** Un mockup visual del prompt con bloques coloreados:
```
┌────────────────────────────────────────────────┐
│  INSTRUCTION BLOCK (System Prompt)              │
│  "You are a professional financial analyst..." │
└────────────────────────────────────────────────┘
┌────────────────────────────────────────────────┐
│  <technicals>                                  │
│  • Price action summary                        │
│  • Momentum signal                             │
│  • Key indicators                              │
│  </technicals>                                 │
└────────────────────────────────────────────────┘
┌────────────────────────────────────────────────┐
│  <news>                                        │
│  • Sentiment analysis (FinBERT)                │
│  • News catalysts                              │
│  </news>                                       │
└────────────────────────────────────────────────┘
┌────────────────────────────────────────────────┐
│  <conclusion>                                  │
│  • Synthesis of technicals + news              │
│  </conclusion>                                 │
└────────────────────────────────────────────────┘
┌────────────────────────────────────────────────┐
│  [[[ACTION]]]  ← BUY / HOLD / SELL            │
└────────────────────────────────────────────────┘
```

**Por qué es necesario:** El template es el corazón del sistema. Aunque incluimos el texto del template en el paper, un diagrama visual con bloques coloridos haría que el lector entienda inmediatamente la estructura XML que el modelo debe aprender.

---

## 4. Diagrama de la Matriz de Recompensas Asimétrica 3×3

**Ubicación propuesta:** Sección "Methodology → Reward Function Design", en lugar de (o junto a) la tabla actual `tab:reward_matrix`.

**Descripción:** Una matriz visual con colores de semáforo:
- Verde (positivo): diagonal principal (+1.0)
- Rojo intenso: penalizaciones severas (-2.25, -2.0)
- Rojo suave: penalizaciones moderadas (-1.0)

```
           REAL
         BUY  HOLD  SELL
      ┌─────┬─────┬─────┐
 BUY  │ +1.0│ -1.0│ -2.25│
      ├─────┼─────┼─────┤
PRED  │     │     │      │
 HOLD │ -2.0│ +1.0│ -2.0 │
      ├─────┼─────┼─────┤
 SELL │ -2.0│ -1.0│ +1.0 │
      └─────┴─────┴─────┘
```

**Por qué es necesario:** La tabla actual es funcional pero aburrida. Un diagrama con heatmap de colores resaltaría inmediatamente la asimetría (BUY mal cuando debería ser SELL es peor que SELL mal cuando debería ser BUY) y la severidad de las penalizaciones de HOLD (-2.0).

---

## 5. Gráfico Comparativo de Modelos (Fitness vs Cumulative Return)

**Ubicación propuesta:** Sección "Results → Analysis", reemplazando o complementando la tabla `tab:validation_results`.

**Descripción:** Un scatter plot con:
- Eje X: Maximum Drawdown (MDD)
- Eje Y: Cumulative Return (CR)
- Tamaño de burbuja: Fitness
- Color: Verde (Hall of Fame, MDD < 20%), Rojo (fuera de umbral)
- Labels: Nombres de los 3 mejores modelos

**Por qué es necesario:** La tabla con 9 modelos es densa y difícil de interpretar. Un gráfico mostraría visualmente la tensión riesgo/retorno: el modelo ganador (085649) estaría en la esquina superior derecha (alto CR, alto MDD), mientras que 111504 estaría en la zona verde (alto fitness, MDD bajo). Comunica la decisión de selección manual mucho mejor que texto.

---

## 6. Diagrama del Dataset Split y Orígenes

**Ubicación propuesta:** Sección "Dataset → Data sources", después de describir las fuentes.

**Descripción:** Un diagrama de Venn o de cajas anidadas mostrando:
```
┌─────────────────────────────────────────────────────────┐
│              THEFINAI/DAILY_NEWS (CLEF)                 │
│  12 assets: TSLA, MSFT, AAPL, AMZN, GOOGL, META,       │
│  NVDA, ADBE, BMRN, MRNA, BTC, ETH                       │
│  Contains: News + SEC filings + prices                  │
│                                                          │
│  ┌─────────────────────────────────────────┐            │
│  │     YAHOO FINANCE (complementary)       │            │
│  │  3 assets: BTC, ETH, SOL                │            │
│  │  Contains: Prices ONLY (no news)        │            │
│  │                                          │            │
│  │  ┌──────────────────────────────────┐   │            │
│  │  │  DATASET USED FOR TRAINING       │   │            │
│  │  │  Filtered to: has_news == 1      │   │            │
│  │  │  Train: BTC, ETH, SOL (YAHOO)    │   │            │
│  │  │  Val:   BTC, ETH, SOL (YAHOO)    │   │            │
│  │  │  Test:  CLEF assets (held out)   │   │            │
│  │  └──────────────────────────────────┘   │            │
│  └─────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

**Por qué es URGENTE y NECESARIO:** Hay confusión actual sobre de dónde vienen los datos de SOL si Yahoo no tiene noticias. El diagrama aclararía que SOL viene de CLEF (que sí tiene noticias), no de Yahoo. Este es un punto crítico que los revisores podrían cuestionar.

---

## 7. Diagrama de la Curva de Equity (Simulada)

**Ubicación propuesta:** Sección "Results → Full Validation Results", después de describir las métricas.

**Descripción:** Una gráfica de línea mostrando:
- Curva de equity acumulada del modelo ganador (085649) sobre los 3 folds
- Picos y caídas que ilustren el MDD de 48.05%
- Posiblemente una línea comparativa del modelo de mejor fitness (111504) para contrastar

**Por qué es necesario:** Las métricas numéricas no comunican la volatilidad del modelo. Un gráfico de equity haría tangible el riesgo: el lector vería visualmente cómo el modelo acumula 90.67% de retorno pero sufre caídas del 48% en el camino.

---

## Prioridad de Implementación

| Prioridad | Diagrama | Impacto en claridad | Esfuerzo |
|-----------|----------|---------------------|----------|
| **P0 (Crítico)** | Dataset Split y Orígenes | Alto | Medio |
| **P0 (Crítico)** | Arquitectura del Sistema | Alto | Medio |
| **P1 (Importante)** | Pipeline de Entrenamiento 2 Fases | Alto | Bajo |
| **P1 (Importante)** | Scatter Plot Modelos (CR vs MDD) | Alto | Medio |
| **P2 (Deseable)** | Estructura del Prompt | Medio | Bajo |
| **P2 (Deseable)** | Matriz de Recompensas Visual | Medio | Bajo |
| **P3 (Nice to have)** | Curva de Equity | Medio | Alto |
