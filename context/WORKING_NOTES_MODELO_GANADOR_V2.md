# Working Notes — Linaje del Modelo Ganador `exp_20260506_085649_254046` (Versión Revisada)

> **Responsabilidad:** Este documento es la fuente de verdad sobre cómo se generó el modelo que hoy sirve el endpoint. Describe el estado exacto del código, los datos y el proceso de entrenamiento.

## Tabla de Contenidos

1. [Introducción](#1-introducción)
2. [Procedimiento Paso a Paso: Entrenamiento](#2-procedimiento-paso-a-paso-entrenamiento)
3. [Procedimiento Paso a Paso: Validación](#3-procedimiento-paso-a-paso-validación)
4. [Contexto del Código](#4-contexto-del-código)
5. [Construcción del Dataset](#5-construcción-del-dataset)
6. [Prompt Builder y Template](#6-prompt-builder-y-template)
7. [Data Augmentation](#7-data-augmentation)
8. [Entrenamiento y Validación: Detalles Técnicos](#8-entrenamiento-y-validación-detalles-técnicos)
9. [Validación Completa y Selección](#9-validación-completa-y-selección)
10. [Explicación del Fitness](#10-explicación-del-fitness)
11. [Hallazgos, Limitaciones y Replicación](#11-hallazgos-limitaciones-y-replicación)
12. [Conclusiones](#12-conclusiones)
13. [Trabajo a Futuro](#13-trabajo-a-futuro)

---

## 1. Introducción

Este documento reconstruye con precisión forense el proceso de entrenamiento del modelo `exp_20260506_085649_254046`, que actualmente sirve el endpoint de inferencia. El modelo fue entrenado el **6 de mayo de 2026** mediante un pipeline de dos fases: **SFT Warm-up** (30 minutos, 1 época) seguido de **GRPO** (6 minutos, 1 época), como parte de una búsqueda de Optuna TPE con 3 trials.

**Hallazgo central:** El modelo actual NO es necesariamente el "mejor" modelo producido por el sistema. Durante la validación completa del 10 de mayo de 2026, se descubrió que existía al menos un modelo con **fitness 18.5× superior** (`exp_20260507_111504_33de19`, fitness=1.974, MDD=17.48%) que pasaba todos los criterios automáticos del Hall of Fame. Sin embargo, el modelo documentado aquí fue seleccionado **manualmente** por su Cumulative Return (90.67%), a pesar de su MDD de 48% que excedía el umbral del Hall of Fame (20%).

**Base de código para replicación:** Commit `0eeb17a` (11 mayo 2026) o HEAD `d67b222`. **NO usar `095f572`** — carece de los cambios críticos implementados el 5 de mayo.

---

## 2. Procedimiento Paso a Paso: Entrenamiento

### 2.1 Resumen Ejecutivo del Pipeline

| Fase | Duración | Descripción | Output |
|------|----------|-------------|--------|
| **Fase 0** | ~1 min | Inicialización de Optuna, creación del trial, generación de metadata | `metadata.json` |
| **Fase 1** | ~1 min | Construcción del dataset: carga CSVs, enriquecimiento FinBERT/Chronos, cálculo de GT, filtrado temporal | Dataset HF (1082 train, 896 val) |
| **Fase 2** | ~2 min | Carga del modelo base Phi-4-mini en 4-bit + adaptadores LoRA (rank=4) | Modelo en VRAM |
| **Fase 3** | **30 min** | **SFT Warm-up**: enseña el formato XML al modelo | Adaptadores SFT entrenados |
| **Fase 4** | **6 min** | **GRPO Training**: optimización por refuerzo con matriz 3×3 + format reward + length penalty | Adaptadores GRPO finales |
| **Fase 5** | ~1 min | Evaluación fast mode: 100 filas del último fold | Métricas (SR=2.95, MDD=0.11%) |
| **Fase 6** | ~1 min | Checkpoint guardado, modelo descargado de VRAM | `epoch_0000/` en warehouse |

**Duración total:** ~37 minutos (08:56:49 → 09:34:18 UTC-6, 6 mayo 2026)

### 2.2 Fase 0: Inicialización del Experimento

```
08:56:49 → OptunaSearcher inicia con semilla 42 y 3 trials
08:56:49 → Se crea Trial 0: exp_20260506_085649_254046
08:56:49 → Se guarda metadata.json con la configuración completa
```

**Configuración Optuna:**
- Algoritmo: TPE (Tree-structured Parzen Estimator)
- Semilla: 42
- Número de trials: 3 (modo rápido)
- Espacio de búsqueda optimizado: `learning_rate`, `kl_beta`, `clip_epsilon`, `temperature`, `reward_scale_factor`, `gt_buy_threshold_pct`, `gt_sell_threshold_pct`

### 2.3 Fase 1: Construcción del Dataset

```
08:56:49 → DatasetBuilder filtra split 'train' a 12 meses: 1082 filas
08:56:49 → DatasetBuilder filtra split 'val' a 12 meses: 896 filas
```

**Proceso detallado:**
1. `DatasetBuilder.build()` carga los CSVs granulares para cada activo en `ASSET_REGISTRY`
2. Para cada activo:
   - Lee `{asset}_texto.csv`, `{asset}_escalares.csv`, `{asset}_temporal.csv`
   - Filtra filas con `has_news == 1`
   - Ejecuta `SentimentExtractor` (FinBERT) y `PriceForecaster` (Chronos) para enriquecer
   - Calcula Ground Truth: `signal = log_return_t1 / rolling_std(20)`
   - Clasifica: Top 35.82% → BUY, Bottom 19.18% → SELL, resto → HOLD
   - Asigna split: CLEF → test, YAHOO >= 2024-01-01 → val, resto → train
3. Genera `data/parquet/lab_dataset.parquet` con hash SHA-256 para invalidación
4. `DatasetBuilder.load_split("train")` carga y filtra a últimos 12 meses: **1082 filas**
5. `DatasetBuilder.load_split("val")` carga y filtra a últimos 12 meses: **896 filas**
6. `build_all()` convierte a HuggingFace Datasets con columnas `prompt` (formato mensajes) y `ground_truth`
7. `build_sft_all()` genera dataset SFT con `prompt` (mensajes) y `completion` (respuesta ideal)

### 2.4 Fase 2: Carga del Modelo

```
08:56:49 → Factory crea LLMWrapper — unsloth/Phi-4-mini-instruct (lora_rank=4, fast_mode=True)
08:57:02 → LLMWrapper carga modelo en VRAM (~2.5GB en 4-bit)
08:57:03 → Se añaden adaptadores LoRA (rank=4, alpha=16, dropout=0)
```

**Parámetros de LoRA (del adapter_config.json):**
- r (rank): 4
- lora_alpha: 16
- lora_dropout: **0** (hardcodeado, el ModelConfig dice 0.05 pero se ignora)
- target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- use_gradient_checkpointing: "unsloth"
- task_type: CAUSAL_LM

### 2.5 Fase 3: SFT Warm-up (30 minutos)

```
08:57:03 → Trainer inicia configuración de GRPOTrainer
08:57:04 → [Trainer] --- Iniciando fase SFT Warm-up ---
08:57:04 → [SFTWarmup] Iniciando SFT Warm-up (1 épocas)...
08:57:29 → [SFTWarmup] Entrenando SFT...
09:27:15 → [SFTWarmup] ✅ SFT Warm-up completado.
```

**Proceso detallado:**
1. `SFTWarmup.run()` recibe los datasets SFT (train + val)
2. Para cada ejemplo, formatea el prompt como:
   ```python
   msgs + [{"role": "assistant", "content": "<technicals>...</technicals>\n<news>...</news>\n<conclusion>...</conclusion>\n[[[ACTION]]]"}]
   ```
3. Aplica el chat template del tokenizer de Phi-4-mini
4. Configura `SFTTrainer` con:
   - batch_size=1, gradient_accumulation=32
   - learning_rate = 2.31e-05 / 2 = 1.155e-05 (LR/2 para estabilidad)
   - max_seq_length=1024
   - bf16=True, gradient_checkpointing=True
5. Entrena 1 época sobre 1082 ejemplos (~30 minutos)
6. Guarda adaptadores LoRA del SFT, descarga modelo, recarga base + adaptadores SFT (para liberar VRAM del optimizador SFT)

### 2.6 Fase 4: Entrenamiento GRPO (6 minutos)

```
09:27:15 → [Trainer] Iniciando GRPOTrainer.train()...
```

**Configuración GRPO:**
- `GRPOConfig` con:
  - beta (KL): 0.0511
  - num_generations: 4 (group size G)
  - max_completion_length: 128
  - max_prompt_length: 1500
  - temperature: 0.5734
  - per_device_train_batch_size: 1
  - gradient_accumulation_steps: 32
  - learning_rate: 2.31e-05
  - num_train_epochs: 2 (solo se completó 1)
  - bf16=True

- Callbacks registrados:
  - `EpochEvaluatorCallback`: evalúa al final de cada época
  - `CollapseDetectorCallback`: detecta colapso a HOLD
  - `EarlyStoppingCallback`: paciencia de 3 épocas
  - `PreemptionCallback`: señales de preempción GPU

**Recompensa por cada generación:**
```
reward = (matrix_reward × 1.5799) + format_reward - length_penalty
```

- `matrix_reward`: de la tabla 3×3
- `format_reward`: +0.5 (completa), +0.25 (acción encontrada), -0.5 (truncada)
- `length_penalty`: max(0, len(response) - 240) × 0.0025

### 2.7 Fase 5: Evaluación Fast Mode (1 minuto)

```
09:33:18 → [Trainer] --- Iniciando evaluación de época 0 ---
09:33:18 → [FoldBuilder] Creados 3 folds sobre 896 filas del validation set
09:33:18 → [Evaluator] 🚀 Modo Rápido: Evaluando solo el último fold para ahorrar tiempo.
09:33:18 → [Evaluator] Evaluando sobre 1 folds...
09:33:18 → [Evaluator] ✂️ Limitando a 100 filas de prueba.
09:33:18 → [Evaluator] Fold 2: 2024-10-04 → 2025-02-01 (100 rows)
09:34:16 → [Evaluator] Muestra Pred 0: <technicals> The price action of SOL...
09:34:16 → [Evaluator] Muestra Pred 1: <technicals> The price action of ETH...
09:34:18 → Checkpoint guardado — época 0, SR=2.9488, MDD=0.11%, fitness=2.9488
```

**Proceso detallado:**
1. Se construyen 3 folds temporales sobre 896 filas de validación
2. **Modo rápido:** Solo se evalúa el **último fold** (fold 2), limitado a **100 filas**
3. Se ejecuta inferencia batch sobre las 100 filas
4. Para cada fila, se extrae la acción (`[[[ACTION]]]`) de la respuesta del modelo
5. Se simula el mercado con `MarketSimulator`:
   - BUY → w=+1, HOLD → w=0, SELL → w=-1
   - Retorno diario: `r_t = w_t × (P_t - P_{t-1}) / P_{t-1}`
6. Se calculan métricas financieras para cada activo (ETH y SOL en este fold)
7. El fitness final es la mediana del SR menos penalización si MDD > 20%

**Resultado por activo:**

| Activo | Días | Acciones | SR | MDD | CR | Accuracy |
|--------|------|----------|-----|-----|-----|----------|
| ETH | 48 | HOLD:45, SELL:3, BUY:0 | 3.248 | 0.11% | +7.6% | 31.2% |
| SOL | 52 | HOLD:51, SELL:1, BUY:0 | 2.649 | 0.00% | +1.2% | 40.4% |

### 2.8 Fase 6: Finalización

```
09:34:18 → Fitness = 2.9488 (mediana SR - 0 penalización MDD)
09:34:18 → Checkpoint nominado como candidato
09:34:18 → Entrenamiento finalizado. Total=1 época, Best fitness=2.9488
09:34:19 → Modelo descargado de VRAM
```

---

## 3. Procedimiento Paso a Paso: Validación

### 3.1 Configuración de Validación

La validación completa se ejecutó el **10 de mayo de 2026** sobre los 11 modelos candidatos. A diferencia del fast mode del entrenamiento, la validación completa:

1. **No limita filas:** Usa las 896 filas completas del validation set
2. **Evalúa los 3 folds:** No se salta ni limita folds
3. **Calcula métricas por activo en cada fold**

### 3.2 Folds de Validación

El `FoldBuilder` divide las 896 filas en 3 folds temporales consecutivos:

| Fold | Fechas | Filas | Activos |
|------|--------|-------|---------|
| Fold 0 | 2024-02-07 → 2024-06-05 | 360 | BTC_YAHOO, ETH_YAHOO, SOL_YAHOO |
| Fold 1 | 2024-06-06 → 2024-10-03 | 294 | BTC_YAHOO, ETH_YAHOO, SOL_YAHOO |
| Fold 2 | 2024-10-04 → 2025-02-01 | 242 | BTC_YAHOO, ETH_YAHOO, SOL_YAHOO |

**Nota:** El validation set solo contiene activos YAHOO (BTC, ETH, SOL). Los activos CLEF están en el split "test" y no se usan en validación.

### 3.3 Proceso de Cálculo de Métricas

Para cada modelo candidato:

1. **Carga:** Se carga el checkpoint LoRA + modelo base
2. **Inferencia:** Se predicen acciones para cada fila del validation set
3. **Por cada fold y por cada activo:**
   - Se filtran las filas del activo dentro del rango de fechas del fold
   - Se simulan retornos diarios: `r_t = w_t × (P_t - P_{t-1}) / P_{t-1}`
   - Se calculan: CR, SR, MDD, accuracy, win rate, profit factor
4. **Fitness global:**
   - `median_SR = mediana(SR por fold/activo)`
   - `worst_MDD = max(MDD por fold/activo)`
   - `fitness = median_SR - max(0, worst_MDD - 0.20) × 10`

### 3.4 Resultados Detallados del Modelo Ganador

Para `exp_20260506_085649_254046` en validación completa:

| Métrica Global | Valor |
|----------------|-------|
| Fitness | 0.1069 |
| Sharpe Ratio (mediana) | 2.9117 |
| Max Drawdown (peor) | 48.05% |
| Cumulative Return | 90.67% |
| Accuracy | 38.31% |
| Win Rate | 54.83% |
| Profit Factor | 1.513 |

### 3.5 Comparación con el Modelo de Mejor Fitness

El modelo `exp_20260507_111504_33de19` (epoch 0) obtuvo el **mejor fitness global** (1.974), con MDD=17.48% (dentro del umbral del Hall of Fame de 20%):

| Métrica | Ganador (085649) | Mejor Fitness (111504) | Ventaja |
|---------|-------------------|------------------------|---------|
| Fitness | 0.1069 | **1.974** | 18.5× mejor |
| SR | 2.9117 | 1.974 | Ganador mejor |
| MDD | 48.05% | **17.48%** | 111504 dentro del umbral |
| CR | **90.67%** | 19.34% | Ganador mejor |
| Accuracy | 38.31% | 41.85% | Similar |
| Win Rate | 54.83% | 58.14% | Similar |

**Conclusión:** El modelo `111504_33de19` era objetivamente superior por casi todos los criterios de riesgo-ajustados y pasaba el umbral del Hall of Fame. El modelo `085649_254046` fue seleccionado **exclusivamente** por su Cumulative Return (90.67% vs 19.34%).

---

## 4. Contexto del Código

### 4.1 Cronología Definitiva

| Momento | Fecha (UTC-6) | Evento | Fuente |
|---------|---------------|--------|--------|
| `095f572` | 2026-05-05 13:26 | Commit anterior al entrenamiento. **NO contenía `grpo_cot_v1`**. | Git |
| **Sesión Antigravity** | 2026-05-05 ~21:17 | **Se implementan todos los cambios críticos**: `grpo_cot_v1`, `build_messages()`, `sft_trainer.py`, `CollapseDetectorCallback`, chat template fix, HOLD penalty -2.0 (de -1.0). | Chat `011ec906` |
| **Entrenamiento** | 2026-05-06 08:56 | **Momento exacto del entrenamiento**. Se ejecuta con el código modificado el 5 de mayo. | `lab_20260506_085649.log` |
| `0eeb17a` | 2026-05-11 00:43 | Commit de **consolidación**. Contiene exactamente los mismos cambios hechos el 5 de mayo. | Git |
| `d67b222` | 2026-05-14 12:46 | HEAD actual. Incluye documentación y mejoras futuras. | Git |

**Hallazgo crítico (resuelto):** El commit `095f572` (5 mayo, 13:26) no tenía `grpo_cot_v1`. Sin embargo, **el 5 de mayo a las ~21:17 UTC** se tuvo una sesión con Antigravity donde se implementaron todos los cambios clave. El entrenamiento del 6 de mayo se ejecutó con ese código. El commit `0eeb17a` (11 mayo) fue solo una consolidación de estos cambios.

**Conclusión:** Podemos replicar el entrenamiento usando el commit `0eeb17a` (o el código actual), ya que contiene los mismos cambios que estaban presentes el 6 de mayo. Las únicas diferencias posteriores son documentación y mejoras no esenciales.

### 4.2 Archivos Clave en el Momento del Entrenamiento

**Fuente:** Conversación `011ec906` del 5 de mayo + logs del sistema del 6 de mayo.

| Componente | Estado en Entrenamiento | Fuente |
|------------|------------------------|--------|
| `prompt_builder.py` con `grpo_cot_v1` | ✅ **Implementado el 5 mayo** | Chat `011ec906` + diff Git |
| `build_messages()` y `build_sft_response()` | ✅ **Implementados el 5 mayo** | Chat `011ec906` + diff Git |
| `sft_trainer.py` (SFT warm-up) | ✅ **Creado el 5 mayo, ejecutado el 6 mayo** | Chat `011ec906` + logs |
| `CollapseDetectorCallback` | ✅ **Añadido el 5 mayo** | Chat `011ec906` + diff Git |
| Chat template fix (`build_messages` en vez de `build_batch`) | ✅ **Implementado el 5 mayo** | Chat `011ec906` + diff Git |
| HOLD penalty -2.0 (de -1.0) — romper Nash | ✅ **Implementado el 5 mayo** | Diff Git `095f572→0eeb17a` |
| Reward function con format + length penalty | ✅ **Implementado el 5 mayo** | Diff Git `095f572→0eeb17a` |
| `optuna_searcher.py` con `build_messages()` | ✅ **Modificado el 5 mayo** | Diff Git `095f572→0eeb17a` |
| Augmentations (block shuffle, etc.) | ❌ **NO implementados** | Confirmado por ausencia en todos los chats y commits |

### 4.3 Recuperación de Conversaciones de Antigravity

Dado que el código intermedio no estaba commiteado en Git, recuperamos las conversaciones de Antigravity del 5 de mayo para reconstruir el estado exacto del código antes del entrenamiento.

**Conversaciones recuperadas:**

| ID | Fecha | Tema | Ubicación |
|----|-------|------|-----------|
| `011ec906` | 5 mayo ~21:17 UTC | **GRPO Revival** — Implementación de `grpo_cot_v1`, SFT trainer, fixes | `docs/Recuperación/` |
| `2358fc58` | 5 mayo ~21:02 UTC | Mejoras al logger | `docs/Recuperación/` |
| `c0fe3d61` | 5 mayo ~19:56 UTC | Diagnóstico de experimento previo | `docs/Recuperación/` |

**Hallazgos clave de la conversación `011ec906`:**
1. **Problema identificado:** El modelo anterior colapsaba a HOLD 100% ("Efecto Eco")
2. **Solución implementada:** Formato CoT estructurado con XML (`<technicals>`, `<news>`, `<conclusion>`, `[[[ACTION]]]`)
3. **SFT Warm-up:** Creado `sft_trainer.py` para enseñar el formato antes del RL
4. **Reward function:** Penalización de HOLD aumentada de -1.0 a **-2.0** para romper el equilibrio de Nash (junto con Decision Placement Reward y Length Penalty)
5. **Chat template:** Cambio de `build_batch()` (strings) a `build_messages()` (lista de mensajes) para compatibilidad con TRL
6. **Augmentations:** **No mencionados en ninguna conversación** — confirmado que no se implementaron

> **⚠️ Corrección:** La penalización HOLD se incrementó de -1.0 a **-2.0** (no a -1.5 como se documentó en versiones anteriores). El valor -1.5 corresponde a `reward_invalid_format`, que no cambió.

**Predicciones confirman `grpo_cot_v1`:**
```
09:34:16 → Muestra Pred 0: <technicals> The price action of SOL shows a bearish momentum...
```
El formato XML `<technicals>...<news>...<conclusion>...[[[SELL]]]` solo aparece con el template `grpo_cot_v1`.

---

## 5. Construcción del Dataset

### 5.1 Fuente de Datos

El dataset se construyó a partir de archivos CSV granulares ubicados en `data/`:

| Archivo | Descripción | Contenido |
|---------|-------------|-----------|
| `{asset}_texto.csv` | Texto enriquecido | Fecha, noticias, momentum, flags `has_news` |
| `{asset}_escalares.csv` | Datos numéricos | Log-returns, volatilidad, precios |
| `{asset}_temporal.csv` | Secuencias temporales | Historial de precios para Chronos |

Activos procesados (según `ASSET_REGISTRY`):
- **Training:** `BTC_YAHOO`, `ETH_YAHOO`, `SOL_YAHOO`
- **Test (OOS):** `BTC_CLEF`, `ETH_CLEF`, `TSLA_CLEF`, `MSFT_CLEF`, `BMRN_CLEF`

### 5.2 Pipeline de Enriquecimiento

#### 5.2.1 FinBERT — SentimentExtractor (`src/enrichment/sentiment_extractor.py`)

**Modelo:** `ProsusAI/finbert`

**Flujo:**
1. Lee `{asset}_texto.csv` y filtra filas donde `has_news == 1`
2. Para cada noticia, pasa el texto truncado a 2000 caracteres por el pipeline FinBERT
3. Extrae: `sentiment_label` (positive/negative/neutral), `sentiment_score` (confianza), y probabilidades por clase
4. Guarda en caché: `models/{asset}_sentiment.csv` con archivo `.hash` para invalidación

**Parámetros:**
- Dispositivo: CPU (`device=-1`, para no interferir con GPU del trainer)
- Truncación: 512 tokens (`max_length=512`)
- Máximo de texto: 2000 caracteres
- Fallback si error: `neutral`, score 0.5

#### 5.2.2 Chronos — PriceForecaster (`src/enrichment/price_forecaster.py`)

**Modelo:** `amazon/chronos-t5-small`

**Flujo:**
1. Lee `{asset}_temporal.csv` con columna `prices_sequence` (string de lista Python)
2. Para cada secuencia, predice 1 día adelante con 20 muestras
3. Calcula probabilidad de subida/bajada comparando muestras vs. último precio
4. Clasifica dirección:
   - `up` si `prob_up > 0.45`
   - `down` si `prob_down > 0.40`
   - `neutral` en otro caso

**Parámetros:**
- Dispositivo: CPU (`device="cpu"`)
- `num_samples = 20`
- `prediction_length = 1`
- Guarda en caché: `models/{asset}_chronos.csv`

### 5.3 Regla de Inclusión y Filtrado

**Regla de inclusión:** Solo filas donde `has_news == 1`. Filas sin noticias se descartan silenciosamente.

**Regla de split temporal (`DatasetBuilder._assign_split()`):**
```python
if is_oos(date_str) or source == "clef":
    return "test"        # Datos CLEF siempre van a test (OOS)
if date_str >= val_start_approx (2024-01-01):
    return "val"         # Datos YAHOO >= 2024-01-01 van a val
return "train"           # Todo lo demás va a train
```

**Nota importante sobre CLEF:** Los activos con `source == "clef"` (BTC_CLEF, ETH_CLEF, TSLA_CLEF, MSFT_CLEF, BMRN_CLEF) **siempre se marcan como "test"**, sin importar su fecha. Esto significa que:
- **Train** solo contiene: BTC_YAHOO, ETH_YAHOO, SOL_YAHOO
- **Val** solo contiene: BTC_YAHOO, ETH_YAHOO, SOL_YAHOO
- **Test** contiene: TODOS los activos CLEF (nunca vistos en entrenamiento)

**Ground Truth:**
- Fórmula: `signal = log_return_t1 / rolling_std(volatility_window)`
- Umbral BUY: Top **35.82%** de señal (valor optimizado por Optuna; default es 40%)
- Umbral SELL: Bottom **19.18%** de señal (valor optimizado por Optuna; default es 15%)
- Resto: HOLD
- Ventana de volatilidad: 20 días

> **⚠️ Nota:** Los umbrales 35.82% y 19.18% son valores optimizados por Optuna para este experimento específico (registrados en `metadata.json` como `gt_buy_threshold_pct` y `gt_sell_threshold_pct`). Los defaults en `ModelConfig` son 40% y 15% respectivamente.

### 5.4 Filtrado Temporal (Fast Mode)

El entrenamiento se ejecutó en **fast_mode=True**, lo que activa `dataset_months=12`.

En el código (commit `0eeb17a` y HEAD), el filtrado usa:
```python
if self._model.dataset_months > 0 and not df.empty:
    max_date = df["date"].max()  # Ej: "2025-02-01" (por ETH/SOL)
    limit_dt = max_dt - timedelta(days=30 * dataset_months)  # 360 días para 12 meses
    df = df[df["date"] >= limit_str]
```

**Fechas exactas del dataset filtrado:**

| Split | Fecha Inicio | Fecha Fin | Filas |
|-------|-------------|-----------|-------|
| Train | 2024-02-07 | 2025-02-01 | 1082 |
| Val | 2024-02-07 | 2025-02-01 | 896 |

> **Cálculo:** La fecha máxima del dataset es **2025-02-01** (determinada por ETH_YAHOO y SOL_YAHOO). Restando 360 días (30×12) se obtiene **2024-02-07** como fecha de inicio del filtrado. Esto coincide exactamente con el inicio del Fold 0 del validation set.

> **⚠️ Corrección:** En el commit `095f572`, el filtrado usaba `timedelta(days=365)` (365 días exactos). El entrenamiento real usó el código intermedio del 5 de mayo, que ya tenía `dataset_months=12` con `timedelta(days=30*12)=360 días`. No son equivalentes.

### 5.5 Columnas Finales del Dataset

El Parquet resultante (`data/parquet/lab_dataset.parquet`) contiene:

| Columna | Tipo | Origen | Descripción |
|---------|------|--------|-------------|
| `date` | str | CSV | Fecha (YYYY-MM-DD) |
| `asset` | str | Registry | Símbolo del activo |
| `symbol` | str | Registry | Símbolo (redundante) |
| `source` | str | Registry | "yahoo" o "clef" |
| `price_close` | float | Escalares | Precio reconstruido desde log-returns |
| `momentum` | int | Texto | Señal de momentum (-1, 0, 1) |
| `news` | str | Texto | Texto de noticias concatenado |
| `filing_10k` | str | Texto | Contexto 10-K (si disponible) |
| `filing_10q` | str | Texto | Contexto 10-Q (si disponible) |
| `sentiment_label` | str | FinBERT | positive / negative / neutral |
| `sentiment_score` | float | FinBERT | Confianza del label (0-1) |
| `prob_positive` | float | FinBERT | Probabilidad clase positive |
| `prob_negative` | float | FinBERT | Probabilidad clase negative |
| `prob_neutral` | float | FinBERT | Probabilidad clase neutral |
| `chronos_direction` | str | Chronos | up / down / neutral |
| `chronos_confidence` | float | Chronos | Confianza de la dirección |
| `log_return_t1` | float | Escalares | Log-return del día siguiente |
| `volatilidad_diaria` | float | Escalares | Volatilidad diaria |
| `future_price_diff` | float | Escalares | Diferencia de precio futuro |
| `ground_truth` | str | Calculado | BUY / HOLD / SELL |
| `asset_class` | str | Registry | "crypto" o "equity" |
| `annualization_factor` | int | Registry | 365 (crypto) o 252 (equity) |
| `split` | str | Calculado | train / val / test |

---

## 6. Prompt Builder y Template

### 6.1 Template Exacto Usado

Según `metadata.json`: `"prompt_template_id": "grpo_cot_v1"`

El template `grpo_cot_v1` (reconstruido desde el código posterior, ya que no existía en el commit anterior):

```
You are a professional financial analyst. Your task is to analyze the provided data and make a trading decision.

Format your response using the following structure:
<technicals>
Summarize price action, momentum, and key indicators.
</technicals>
<news>
Summarize sentiment and recent news catalysts.
</news>
<conclusion>
Synthesize why the technicals and news justify the decision.
</conclusion>
[[[ACTION]]]

Where ACTION is exactly one of: BUY, HOLD, SELL.

Data:
Date: {date}
Asset: {symbol}
Current Price: {price}
Momentum: {momentum}
Sentiment: {sentiment_label} (score: {sentiment_score:.3f})
Chronos Forecast: {chronos_direction} (confidence: {chronos_confidence:.3f})
News Summary:
{news_summary}
{sec_section}
Price History ({history_len} days): {price_history}

Analyze carefully and output your final decision in [[[ACTION]]] format.
```

### 6.2 Ejemplo Real del Prompt

A continuación un ejemplo del prompt generado para una fila real del dataset (reconstruido desde los logs y el template):

```
You are a professional financial analyst. Your task is to analyze the provided data and make a trading decision.

Format your response using the following structure:
<technicals>
Summarize price action, momentum, and key indicators.
</technicals>
<news>
Summarize sentiment and recent news catalysts.
</news>
<conclusion>
Synthesize why the technicals and news justify the decision.
</conclusion>
[[[ACTION]]]

Where ACTION is exactly one of: BUY, HOLD, SELL.

Data:
Date: 2024-10-15
Asset: SOL
Current Price: 21560.11
Momentum: -1
Sentiment: negative (score: 0.723)
Chronos Forecast: down (confidence: 0.612)
News Summary:
Cryptocurrency markets under pressure as regulatory concerns mount. SOL faces headwinds from broader altcoin selloff. Technical indicators show bearish divergence on daily charts.

Price History (10 days): [23503.58, 23120.45, 22890.12, 22560.33, 22100.78, 21850.90, 21720.15, 21600.44, 21580.21, 21560.11]

Analyze carefully and output your final decision in [[[ACTION]]] format.
```

**Nota:** Los valores son representativos y reconstruidos a partir de las muestras del log (`The price action of SOL shows a bearish momentum...`). El prompt real varía por fila del dataset.

### 6.3 Formato para GRPOTrainer

En el commit `095f572`, el `DatasetBuilder.build_all()` generaba prompts como strings simples:

```python
prompts = pb.build_batch(df.to_dict("records"))
```

En el código intermedio del entrenamiento (confirmado por el diff `095f572→0eeb17a`), **definitivamente** usaba el formato de mensajes para que TRL aplicara el chat template:

```python
prompts = [pb.build_messages(r) for r in records]
# Donde build_messages devuelve: [{"role": "user", "content": prompt}]
```

Esto permitía que el tokenizer de Phi-4-mini aplicara automáticamente el chat template (`<|im_start|>user\n...`).

> **Corrección:** Antes se decía "probablemente usaba `build_messages()`". Ahora está confirmado por el diff Git.

### 6.4 Respuesta Ideal para SFT Warm-up

El SFT warm-up usa `build_sft_response()` para generar respuestas ideales ultra-concisas:

```python
def build_sft_response(self, row: dict[str, Any]) -> str:
    gt = row.get("ground_truth", "HOLD")
    momentum = self._format_momentum(row.get("momentum", 0))
    sentiment = row.get("sentiment_label", "neutral")
    direction = {
        "BUY":  "Entry justified.",
        "SELL": "Risk reduction warranted.",
        "HOLD": "No clear edge.",
    }.get(gt, "Uncertain.")
    response = (
        f"<technicals>Momentum {momentum}. {direction}</technicals>\n"
        f"<news>Sentiment {sentiment}.</news>\n"
        f"<conclusion>{direction}</conclusion>\n"
        f"[[[{gt}]]]"
    )
    return response
```

**Ejemplo de respuesta SFT para la fila anterior (SOL, ground_truth=SELL):**
```
<technicals>Momentum bearish. Risk reduction warranted.</technicals>
<news>Sentiment negative.</news>
<conclusion>Risk reduction warranted.</conclusion>
[[[SELL]]]
```

> **Nota sobre brevedad:** Las respuestas SFT son ultra-concisa (~60-80 caracteres) porque si el SFT enseña respuestas largas y GRPO las trunca, el modelo colapsa a varianza cero (todas las respuestas del grupo son iguales).

---

## 7. Data Augmentation

### 7.1 Flags del Experimento

Según `metadata.json`:

| Flag | Valor | Descripción |
|------|-------|-------------|
| `aug_block_shuffle` | `true` | Reordena bloques de información en el prompt |
| `aug_subset_sampling` | `true` | Omite campos opcionales aleatoriamente |
| `aug_news_bucketing` | `true` | Agrupa noticias por buckets temporales |
| `aug_self_reflection` | `false` | No añade sección de auto-reflexión |

### 7.2 Verificación Definitiva: Augmentations Nunca Implementados

**Análisis multi-fuente:**

**1. Revisión de commits Git:**

| Commit | Fecha | Augmentations Implementados |
|--------|-------|----------------------------|
| `095f572` | 5 mayo | ❌ No existen |
| `0eeb17a` | 11 mayo | ❌ No existen |
| HEAD actual | 24 mayo | ❌ No existen |

**2. Conversaciones de Antigravity (5 mayo):**
Revisamos las 3 conversaciones recuperadas del 5 de mayo (`c0fe3d61`, `2358fc58`, `011ec906`). **Ninguna menciona augmentations.** Los cambios documentados fueron:
- `grpo_cot_v1` template
- `build_messages()` / `build_sft_response()`
- `sft_trainer.py`
- `CollapseDetectorCallback`
- Chat template fix
- HOLD penalty -2.0 (de -1.0)

**3. Código del `DatasetBuilder` y `PromptBuilder`:**
En el commit `0eeb17a`, se buscaron las cadenas `aug_block_shuffle`, `aug_subset_sampling`, `aug_news_bucketing`, `aug_self_reflection` en todo el código. **Resultado: 0 matches.** Estas flags se leen en `ModelConfig` pero nunca se usan en ningún builder o trainer.

**4. Logs del entrenamiento:**
`lab_20260506_085649.log` no muestra mensajes sobre block shuffle, subset sampling ni ningún otro augmentation.

**Conclusión definitiva:** Aunque el `metadata.json` dice `aug_block_shuffle: true`, `aug_subset_sampling: true`, `aug_news_bucketing: true`, estos **NUNCA FUERON APLICADOS** porque:
1. No existen en ningún commit de Git
2. No se mencionan en ninguna conversación de Antigravity
3. No hay código que los implemente en `DatasetBuilder` ni `PromptBuilder`
4. No hay evidencia en los logs

Son flags de configuración que no tuvieron efecto real.

**Impacto:** Para replicación, **usar augmentations=False** o asumir que no se aplicaron.

---

## 8. Entrenamiento y Validación: Detalles Técnicos

### 8.1 Arquitectura del Modelo

| Parámetro | Valor | Nota |
|-----------|-------|------|
| Modelo base | `unsloth/Phi-4-mini-instruct` | `metadata.json` lo registra así; el `adapter_config.json` usa `unsloth/phi-4-mini-instruct-unsloth-bnb-4bit` — es el mismo modelo con distintos sufijos de Unsloth para carga 4-bit |
| Carga | 4-bit NF4 (via Unsloth) | `load_in_4bit=True` en `LLMWrapper` |
| LoRA rank | 4 | |
| LoRA alpha | 16 | |
| LoRA dropout | **0** (hardcoded) | ⚠️ **Corrección:** El `ModelConfig` dice 0.05, pero `LLMWrapper` hardcodea `lora_dropout=0` en la llamada a `get_peft_model()`. El `adapter_config.json` del checkpoint confirma `"lora_dropout": 0`. Para replicación exacta usar dropout=0. |
| Módulos objetivo | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | |

### 8.2 Configuración de Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| GRPO group size (G) | 4 |
| KL beta | 0.051059032093947576 |
| Clip epsilon | 0.10411689885916049 |
| Max new tokens | 128 |
| Temperature | 0.5733618039413735 |
| Learning rate | 2.3102018878452926e-05 |
| Batch size por device | 1 |
| Gradient accumulation | 32 |
| Max grad norm | 0.5 |
| Épocas (metadata) | 2 |
| Épocas ejecutadas (real) | 1 |
| SFT epochs (metadata) | 1 |
| SFT epochs ejecutadas | ✅ Verificado: 1 época ejecutada (30 minutos) |
| Fast mode | True |
| Dataset months | 12 (filtra a últimos ~360 días) |
| Seed | 42 (Optuna TPE) |
| Optuna trials | 3 |
| GT buy threshold | 0.3582 (35.82%, optimizado por Optuna) |
| GT sell threshold | 0.1918 (19.18%, optimizado por Optuna) |

### 8.3 Discrepancia en Épocas

El `metadata.json` dice `num_epochs: 2`, pero `experiment_metadata.json` confirma `total_epochs_run: 1`. Esto sugiere:
- El entrenamiento terminó después de 1 época (early stopping implícito o truncado por fast_mode)
- Para replicación exacta: usar `num_epochs=1`

### 8.4 Reward Function (Completa)

#### 8.4.1 Matriz de Recompensas 3×3

Valores exactos del `metadata.json`:

| Predicción \ Real | BUY | HOLD | SELL |
|-------------------|-----|------|------|
| **BUY** | +1.0 | -1.0 | -2.25 |
| **HOLD** | **-2.0** | +1.0 | **-2.0** |
| **SELL** | -2.0 | -1.0 | +1.0 |

- `reward_invalid_format`: -1.5
- `reward_scale_factor`: 1.5798625466052894

> **⚠️ Corrección:** Los valores `reward_false_hold_as_buy` y `reward_false_hold_as_sell` son **-2.0** (no -1.5). El cambio del 5 de mayo fue de -1.0 → -2.0 para romper el equilibrio de Nash. El valor -1.5 corresponde a `reward_invalid_format`, que es distinto.

**Observación:** La matriz es asimétrica. Castigar BUY cuando la respuesta correcta es SELL (-2.25) es peor que castigar SELL cuando la respuesta correcta es BUY (-2.0). Las penalizaciones de HOLD (-2.0 en ambas direcciones) son más severas que las de BUY/HOLD o SELL/HOLD (-1.0), lo que incentiva al modelo a tomar acciones en lugar de mantenerse pasivo.

#### 8.4.2 Decision Placement Reward

Además de la matriz 3×3, la función de recompensa incluye un **bono de formato** que se suma a la recompensa de la matriz:

| Condición | Recompensación |
|-----------|---------------|
| Acción encontrada (aunque truncada) | +0.25 |
| Respuesta completa (no truncada, con `]]]`) | +0.25 |
| Respuesta truncada o incompleta | -0.5 |
| Formato inválido (sin BUY/HOLD/SELL) | `reward_invalid_format × scale` (=-2.37) |

#### 8.4.3 Length Penalty

Se aplica una **penalización progresiva por longitud** para evitar colapso intra-grupo:

- **Objetivo:** 240 caracteres (~60 tokens)
- **Tasa:** 0.0025 por carácter extra sobre el objetivo
- **Fórmula:** `length_penalty = max(0, len(response) - 240) × 0.0025`

#### 8.4.4 Fórmula Completa de Recompensa

```
reward = (matrix_reward × scale_factor) + format_reward - length_penalty
```

Donde:
- `matrix_reward` viene de la tabla 3×3
- `scale_factor = 1.5799`
- `format_reward` es +0.5 (completa), +0.25 (parcial), -0.5 (truncada), o `reward_invalid_format × scale`
- `length_penalty` es progresivo sobre 240 caracteres

#### 8.4.5 Extracción de Acciones Mejorada

El método `_extract_action()` tiene detección del formato `[[[ACTION]]]` con prioridad:

1. **Patrón completo `[[[ACTION]]]`:** Prioridad máxima, no parcial
2. **Patrón truncado** `[[[B`, `[[[BU`, etc.: Detecta respuestas cortadas por límite de tokens
3. **Patrones explícitos** tipo `ACTION: BUY`: Etiquetas de acción
4. **Última palabra clave** BUY/HOLD/SELL: Fallback

### 8.5 Resultados del Entrenamiento (Fast Mode)

| Métrica | Valor |
|---------|-------|
| Fitness | 2.948751755376586 |
| Sharpe Ratio | 2.948751755376586 |
| Max Drawdown | 0.1086% |
| Accuracy | 35.82% |
| Loss final | 0.0098 |
| KL Divergence | 0.1915 |
| Reward Mean | -2.4907 |
| Distribución acciones | BUY: 0%, HOLD: 96%, SELL: 4% |

**Desglose por activo (fold 2, 100 filas en fast mode):**

| Activo | Días | SR | MDD | Acc |
|--------|------|-----|-----|-----|
| ETH | 48 | 3.248 | 0.11% | 31.2% |
| SOL | 52 | 2.649 | 0.00% | 40.4% |

**Contexto:** Estos resultados provienen de fast mode: evaluación en solo 100 filas del último fold. Las métricas son muy optimistas comparadas con la validación completa posterior.

### 8.6 Los 3 Trials de Optuna

El log muestra los 3 trials completos:

| Trial | ID | Modelo Base | Fitness | SR | MDD | Resultado |
|-------|----|-------------|---------|-----|-----|-----------|
| 0 | exp_20260506_085649_254046 | Phi-4-mini-instruct | **2.9488** | **2.949** | **0.11%** | ✅ Ganador |
| 1 | exp_20260506_093420_0bfeea | Qwen2.5-3B-Instruct | -1.3247 | -1.325 | 4.90% | ❌ Colapsó a texto repetitivo |
| 2 | exp_20260506_101537_905679 | Phi-4-mini-instruct | -1.3305 | -1.330 | 8.18% | ❌ Colapsó a HOLD |

**Observaciones:**
- Trial 1 usó un modelo diferente (Qwen2.5-3B) — el Candidate Pred 1 muestra "mixed mixed mixed mixed..." indicando colapso
- Trial 2 volvió a Phi-4-mini pero con hyperparámetros diferentes (LR=5e-4 vs 2.3e-5)
- Solo el Trial 0 produjo fitness positivo

---

## 9. Validación Completa y Selección

### 9.1 Contexto de la Validación Completa

- **Fecha:** 2026-05-10, entre las 20:54 y 23:06 UTC-6
- **Método:** Validación manual desde el menú "Configurar Endpoint → Validar top-N modelos"
- **Dataset:** Completo (no fast_mode) — 896 filas de validación, 3 folds temporales
- **Código:** Probablemente el código del commit `0eeb17a` o intermedio (posterior al entrenamiento)

### 9.2 Procedimiento de Validación

La validación completa sigue este proceso:

1. **Carga del modelo:** Se carga el checkpoint LoRA de cada candidato sobre el modelo base `unsloth/Phi-4-mini-instruct` en 4-bit.
2. **Dataset completo:** Se usa el validation set completo (896 filas, sin recorte de fast_mode).
3. **División en folds:** El `FoldBuilder` divide las 896 filas en 3 folds temporales consecutivos:
   - **Fold 0:** 2024-02-07 → 2024-06-05 (360 filas)
   - **Fold 1:** 2024-06-06 → 2024-10-03 (294 filas)
   - **Fold 2:** 2024-10-04 → 2025-02-01 (242 filas)
4. **Evaluación por fold:** Para cada fold, se ejecuta inferencia batch sobre todas las filas.
5. **Market Simulator:** Para cada activo en el fold, se simulan los retornos diarios usando la fórmula:
   ```
   r_t = w_t × (P_t - P_{t-1}) / P_{t-1}
   ```
   donde `w_t = +1` para BUY, `0` para HOLD, `-1` para SELL.
6. **Métricas financieras:** Se calculan para cada activo:
   - **Cumulative Return (CR):** `Π(1 + r_i) - 1`
   - **Sharpe Ratio (SR):** `mean(r) / std(r) × √factor` (factor=365 para crypto, 252 para equity)
   - **Maximum Drawdown (MDD):** Máxima caída desde pico de la curva de equity
   - **Accuracy:** Predicciones correctas / total
   - **Win Rate:** Trades ganadores / trades activos (BUY + SELL)
   - **Profit Factor:** Ganancia bruta / pérdida bruta
7. **Fitness:** Se computa un escalar que combina la mediana del SR con una penalización si MDD > 20%:
   ```
   fitness = median_SR - max(0, MDD_worst - 0.20) × 10
   ```

### 9.3 Resultados Completos de Validación (11 Modelos)

El archivo `warehouse/validation_cache.json` contiene los resultados de **11 validaciones**:

| # | Experimento | Fitness | SR | MDD | CR | Accuracy | WR | PF |
|---|-------------|---------|-----|-----|-----|----------|-----|-----|
| 1 | **exp_20260506_085649_254046** | **0.1069** | **2.9117** | **48.05%** | **90.67%** | 38.31% | 54.83% | 1.513 |
| 2 | exp_20260507_111504_33de19 | **1.974** | **1.974** | **17.48%** | **19.34%** | 41.85% | 58.14% | 1.687 |
| 3 | exp_20260509_113530_8bc775 (ep4) | 1.2134 | 1.2134 | 14.15% | 9.44% | 40.96% | 52.93% | 1.697 |
| 4 | exp_20260506_004508_27f172 | -0.0759 | 3.0408 | 51.17% | 89.76% | 38.87% | 55.81% | 1.568 |
| 5 | exp_20260508_115709_051aed | -0.4434 | 1.4039 | 38.47% | 22.74% | 38.99% | 51.91% | 1.249 |
| 6 | exp_20260508_014550_f9ad37 | -0.8756 | 1.4105 | 42.86% | 28.16% | 40.23% | 50.90% | 1.225 |
| 7 | exp_20260509_113530_8bc775 (ep2) | -0.6082 | -0.1113 | 24.97% | -1.12% | 42.12% | 51.23% | 0.970 |
| 8 | exp_20260507_221904_9bb326 | -2.5173 | -0.2100 | 43.07% | -3.18% | 43.64% | 50.89% | 0.928 |
| 9 | exp_20260508_062606_1cd3e0 | 0.0 | 0.0 | 0.0% | 0.0% | 43.32% | 0.0% | Inf |
| 10 | exp_20260508_100749_12f83a | 0.0 | 0.0 | 0.0% | 0.0% | 43.32% | 0.0% | Inf |
| 11 | exp_20260505_204602_88701e | -1.4462 | -0.1954 | 32.51% | -1.89% | 43.72% | 48.85% | 0.912 |

> **Corrección:** Versión anterior solo listaba 7 modelos. Se añadieron los 4 omitidos, incluyendo **exp_20260507_111504_33de19** (fitness=1.974, MDD=17.48%).

**Nota sobre modelos con MDD=0% y fitness=0:** Los experimentos 9 y 10 (`1cd3e0` y `12f83a`) tienen todas las métricas en 0 excepto accuracy=43.32% y WR=0%. Esto indica que el modelo predijo **100% HOLD**.

### 9.4 Análisis Comparativo

**¿Por qué este modelo fue seleccionado?**

| Criterio | Valor | Ranking vs. otros |
|----------|-------|-------------------|
| **Cumulative Return** | **90.67%** | **#1** (mejor de todos) |
| Sharpe Ratio | 2.9117 | #2 (perdió contra 3.0408 del exp_20260506_004508) |
| MDD | 48.05% | #9 de 11 (solo mejor que dos modelos con MDD > 48%) |
| Fitness | 0.1069 | **#3** (perdió contra 1.974 del 111504 y 1.213 del 8bc775) |

**Observaciones críticas:**
1. exp_20260507_111504_33de19 tiene el **fitness más alto de todos** (1.974), MDD=17.48% (< 20% umbral del Hall of Fame), y accuracy=41.85%. Este modelo **pasa todos los criterios automáticos** del Hall of Fame.
2. exp_20260609_113530_8bc775 (ep4) tiene fitness=1.2134 y MDD=14.15%, también dentro del umbral.
3. Nuestro modelo (fitness=0.1069, MDD=48.05%) está en **3er lugar en fitness** y **fuera del umbral del Hall of Fame**.
4. El modelo con mejor SR puro (exp_20260506_004508, SR=3.04) tiene un MDD aún peor (51.17%).
5. Nuestro modelo fue seleccionado **manualmente** por su alto Cumulative Return (90.67%), a pesar de su baja puntuación de fitness y alto MDD.

### 9.5 Proceso de Decisión Documentado

```
1. Se ejecutó validación completa de múltiples checkpoints candidatos.
2. El modelo exp_20260506_085649_254046 obtuvo el Cumulative Return más alto (90.67%).
3. Aunque su MDD de 48% excedía el umbral del Hall of Fame (20%),
   fue seleccionado manualmente para el endpoint por su rendimiento absoluto.
4. La selección fue consciente del riesgo: se prefirió retorno sobre robustez.
5. NOTA: Existían mejores modelos por fitness (exp_20260507_111504 con fitness=1.974, MDD=17.48%)
   y por criterios automáticos del Hall of Fame, pero se priorizó el Cumulative Return.
```

---

## 10. Explicación del Fitness

### 10.1 ¿Qué es el Fitness?

El **fitness** es un escalar que resume la calidad de un modelo en una única métrica optimizable. Combina el **Sharpe Ratio** (premio por riesgo) con una **penalización severa** si el Maximum Drawdown excede el umbral del laboratorio (20%).

### 10.2 Fórmula Exacta

```python
fitness = median_SR - max(0, MDD_worst - 0.20) × 10
```

Donde:
- `median_SR`: Mediana del Sharpe Ratio calculado sobre todos los folds y activos
- `MDD_worst`: El peor Maximum Drawdown observado (máximo entre todos los folds)
- `0.20`: Umbral del Hall of Fame (20% de drawdown máximo permitido)
- `10`: Factor de penalización (muy agresivo — cada 1% de exceso de MDD resta 0.1 al fitness)

### 10.3 Ejemplos de Cálculo

**Ejemplo 1: Nuestro modelo (exp_20260506_085649_254046)**
- `median_SR = 2.9117`
- `MDD_worst = 0.4805` (48.05%)
- `exceso = max(0, 0.4805 - 0.20) = 0.2805`
- `penalización = 0.2805 × 10 = 2.805`
- `fitness = 2.9117 - 2.805 = 0.1067` ≈ **0.1069** (diferencia por redondeo)

**Ejemplo 2: Modelo con mejor fitness (exp_20260507_111504_33de19)**
- `median_SR = 1.974`
- `MDD_worst = 0.1748` (17.48%)
- `exceso = max(0, 0.1748 - 0.20) = 0` (dentro del umbral)
- `penalización = 0`
- `fitness = 1.974 - 0 =` **1.974**

**Ejemplo 3: Modelo con MDD=0% (100% HOLD)**
- `median_SR = 0.0`
- `MDD_worst = 0.0`
- `exceso = 0`
- `fitness = 0.0 - 0 =` **0.0**

### 10.4 Interpretación

| Fitness | Significado |
|---------|-------------|
| > 1.0 | Excelente — pasa el umbral del Hall of Fame con buen Sharpe |
| 0.0 - 1.0 | Aceptable — dentro del umbral de MDD pero Sharpe modesto |
| < 0.0 | Deficiente — o bien MDD > 20%, o bien Sharpe negativo |
| 0.0 exacto | El modelo predijo 100% HOLD (SR=0, MDD=0) |

> **Nota importante:** El fitness penaliza **severamente** el riesgo. Un modelo con SR=3.0 pero MDD=50% obtendría fitness ≈ 3.0 - 3.0 = **0.0**, mientras que un modelo con SR=1.5 pero MDD=15% obtiene fitness = **1.5**. Esto explica por qué nuestro modelo (SR=2.91, MDD=48%) tiene fitness=0.11, apenas por encima de cero.

---

## 11. Hallazgos, Limitaciones y Replicación

### 11.1 Lo que Sabemos con Certeza

| Elemento | Certeza | Fuente |
|----------|---------|--------|
| Config exacta de hiperparámetros | ✅ Alta | `metadata.json` del experimento |
| Fecha y duración del entrenamiento | ✅ Alta | Logs del experimento |
| Dataset usado (1082 train, 896 val) | ✅ Alta | Logs del DatasetBuilder |
| Pipeline FinBERT/Chronos | ✅ Alta | Código de `0eeb17a` |
| Resultados de validación completa | ✅ Alta | `validation_cache.json` (11 modelos) |
| Prompt template `grpo_cot_v1` | ✅ Alta | Verificado via logs + diff Git |
| SFT warm-up ejecutado | ✅ Alta | Verificado via logs (30 min, 1 época) |
| Data augmentations aplicados | ✅ **Verificado: NO se aplicaron** | Flags en metadata pero nunca implementados en código |
| LoRA dropout efectivo | ✅ **Corregido: 0 (no 0.05)** | `adapter_config.json` + código fuente |
| Función de recompensa completa | ✅ **Corregido: incluye format + length penalty** | `reward_function.py` en commit `0eeb17a` |
| HOLD penalty | ✅ **Corregido: -2.0 (no -1.5)** | Diff `095f572→0eeb17a` |

### 11.2 Lo que YA Verificamos

✅ **SFT ejecutado:** Confirmado por logs (30 min, 1 época)
✅ **Template `grpo_cot_v1`:** Confirmado por logs + diff Git
✅ **Código intermedio:** **Reconstruido** desde diffs Git (`095f572→0eeb17a`)
✅ **Augmentations:** Confirmado que NO se aplicaron (nunca implementados)
✅ **Solo un dataset:** El `DatasetBuilder` genera un único Parquet con hash SHA-256
✅ **LoRA dropout:** El valor efectivo es 0, no 0.05 como indica ModelConfig
✅ **Reward function:** Incluye Decision Placement Reward (+0.25/-0.5) y Length Penalty (0.0025/char sobre 240 chars)
✅ **HOLD penalty:** Los valores cambiaron de -1.0 a -2.0, no a -1.5

### 11.3 Lo que Sigue Sin Verificar

1. **Logs de Optuna (`optuna.log`):** El sistema no guardó logs detallados del árbol de decisiones de TPE. Solo se tienen los resultados finales (fitness por trial).
2. **Hash exacto del dataset en el momento del entrenamiento:** Aunque el `DatasetBuilder` genera hashes, no tenemos el hash específico del 6 de mayo. Sin embargo, si los CSVs fuente no han cambiado desde entonces, el dataset reconstruido será idéntico.
3. **Dataset de validación bit-exacto:** Se puede regenerar con `DatasetBuilder`, pero si los CSVs fuente fueron modificados después del 6 de mayo, el resultado será diferente.

### 11.4 Recomendaciones para Replicación

**¡Buena noticia!** Podemos replicar el entrenamiento. El código del commit `0eeb17a` (11 mayo) contiene exactamente los mismos cambios que el código intermedio del 6 de mayo.

Para replicar:

1. **Base de código:** Usar el commit `0eeb17a` (11 mayo) o el código actual (HEAD `d67b222`). ⚠️ **NO usar `095f572` como base.** Contiene los cambios previos sin `grpo_cot_v1`, `build_messages()`, ni SFT trainer.
2. **Usar `num_epochs=1`** (no 2 como dice metadata)
3. **Usar `grpo_cot_v1`** como template
4. **Ejecutar SFT warm-up:** 1 época, usando `sft_trainer.py`
5. **Augmentations:** **NO usar** (confirmado que no se aplicaron)
6. **LoRA dropout:** Usar **0** (hardcodeado en `LLMWrapper`), no 0.05 (ignorado)
7. **Reward function:** Incluir Decision Placement Reward y Length Penalty (ya en `reward_function.py`)
8. **Dataset:** Verificar que los CSVs fuente sean los mismos del 6 de mayo. Si son idénticos, el Parquet será reproducible.
9. **Configuración exacta:** Usar los valores del `metadata.json` (GT thresholds: 35.82% BUY, 19.18% SELL)

---

## 12. Conclusiones

### 12.1 Resumen del Proceso

El modelo ganador (`exp_20260506_085649_254046`) fue entrenado el 6 de mayo de 2026 mediante un proceso de dos fases:

1. **SFT Warm-up** (30 minutos, 1 época): Calentamiento supervisado para enseñar al modelo el formato de respuesta XML del template `grpo_cot_v1`. Usó `build_sft_response()` para generar respuestas ultra-concisas ancladas al Ground Truth.
2. **GRPO** (6 minutos, 1 época): Entrenamiento por refuerzo con Group Relative Policy Optimization, usando una función de recompensas que combina:
   - Matriz asimétrica 3×3 (escalada por factor 1.5799)
   - Decision Placement Reward (+0.25/-0.5)
   - Length Penalty progresivo (0.0025/char sobre 240 chars)

El entrenamiento fue parte de una búsqueda de Optuna TPE con semilla 42 y 3 trials en modo rápido. El Trial 0 fue el único exitoso (fitness: 2.9488); los otros dos trials produjeron fitness negativos y fueron descartados.

### 12.2 Hallazgos Clave

| # | Hallazgo | Impacto |
|---|----------|---------|
| 1 | El modelo fue sorprendentemente bueno para ser **Trial 0** de solo 3 trials | Indica que hay mucho espacio de mejora con búsqueda más exhaustiva |
| 2 | El modelo era **extremadamente conservador** (96% HOLD, 4% SELL, 0% BUY) | Explica el MDD bajo en fast mode pero también el alto CR en validación |
| 3 | El **MDD saltó de 0.1% a 48%** entre fast mode y validación completa | Fast mode sobreestima robustez; validación completa es esencial |
| 4 | El modelo fue seleccionado **manualmente** por CR, no por Hall of Fame automático | El criterio de selección final fue diferente al criterio de entrenamiento |
| 5 | Existía un modelo con **fitness 18.5× superior** (1.974 vs 0.107) y MDD de 17.48% | La selección manual privilegió retorno sobre ajuste al riesgo |
| 6 | La **función de recompensa** incluye format reward y length penalty, no solo la matriz 3×3 | Afecta significativamente el entrenamiento |
| 7 | **LoRA dropout es 0** (no 0.05 como indica ModelConfig) | El valor en ModelConfig se ignora; el código hardcodea 0 |

### 12.3 Limitaciones de Reproducibilidad

| Limitación | Severidad | Mitigación |
|------------|-----------|------------|
| Código intermedio no commiteado | ~~Alta~~ **Resuelta** | Usar commit `0eeb17a` que contiene exactamente los cambios del 5 mayo |
| No hay hash SHA-256 del dataset | Alta | Generar hash del dataset actual y documentar |
| Augmentations no verificables | ~~Media~~ **Resuelta: NO se aplicaron** | Asumir que no se usaron |
| SFT implementación desconocida | ~~Media~~ **Resuelta** | `sft_trainer.py` está en commit `0eeb17a` |
| LoRA dropout incorrecto en config | **Alta** | Usar dropout=0 (hardcodeado), no 0.05 |
| Reward function incompleta en docs | **Alta** | Incluir format reward y length penalty |

---

## 13. Deuda Técnica

### 13.1 División Train/Val — Prioridad Crítica

**Problema:** La división actual entre train y val es **terrible** e introduce contaminación temporal severa:

- **Train:** Todo antes de 2024-01-01 (incluye datos de 2015-2023)
- **Val:** Todo desde 2024-01-01 hasta 2025-02-01
- **Test (OOS):** Todo después de 2025-07-31 + activos CLEF

**Problemas específicos:**
1. **Gap temporal entre train y val:** Hay un vacío de ~6 meses (2023-12-31 a 2024-01-01) que no existe en la realidad. Los mercados financieros no tienen "vacaciones" de 6 meses.
2. **Train muy largo, val muy corto:** El train tiene ~8 años de datos mientras que val solo tiene ~1 año. Esto hace que el modelo aprenda patrones de 2015-2023 que pueden no ser relevantes para 2024-2025.
3. **Sin validación temporal real:** El val no simula un escenario de "despliegue en producción" donde el modelo ve datos futuros. Es solo un corte temporal arbitrario.
4. **Contaminación por activo:** BTC, ETH y SOL están en train, val y test (OOS), pero con fechas diferentes. El modelo puede aprender "BTC siempre sube" del train y aplicarlo al val. Además que se necesita mejorar el equilibrio entre datos `equity` y `cripto`.

**Recomendación:** Implementar una división temporal más realista:
- Usar **walk-forward validation** donde el modelo se entrena hasta t-1 y se valida en t
- O usar **múltiples ventanas temporales** con overlap controlado
- O usar **purged k-fold cross-validation** como recomienda Marcos López de Prado
- Buscar equilibrar el dataset.

### 13.2 Selección de Modelos — Prioridad Alta

**Problema:** El modelo actual fue seleccionado manualmente por Cumulative Return (90.67%), ignorando completamente el fitness y el MDD. Existía un modelo con fitness 18.5× superior que pasaba el Hall of Fame.

**Recomendación:**
1. Implementar selección **automática** basada en fitness (no manual)
2. Considerar un **ensemble** de los top-N modelos por fitness
3. Usar **validación cruzada temporal** para selección más robusta
4. Documentar explícitamente por qué se seleccionó un modelo con MDD=48% sobre uno con MDD=17%

### 13.3 Fast Mode vs Validación Completa

**Problema:** El fast mode evalúa solo 100 filas del último fold, produciendo métricas muy optimistas (MDD=0.11% vs 48% en validación completa). Esto puede llevar a seleccionar modelos que parecen buenos pero colapsan en producción.

**Recomendación:**
1. Aumentar el número de filas en fast mode (ej. 500 en lugar de 100)
2. Evaluar al menos 2 folds en fast mode
3. Ajustar el criterio de early stopping para usar una estimación más conservadora del MDD

### 13.4 Augmentations — Prioridad Media

**Problema:** Las flags de augmentations (`aug_block_shuffle`, `aug_subset_sampling`, `aug_news_bucketing`) están en `ModelConfig` pero **nunca fueron implementadas** en el código. Son configuraciones "fantasma".

**Recomendación:**
1. Implementar las augmentations o eliminar las flags
2. Considerar augmentations temporales (reordenar días, dropout de noticias)
3. Documentar claramente cuáles augmentations se aplican y cuáles no

### 13.5 Métricas Adicionales

**Problema:** Actualmente solo se reportan SR, MDD, CR, accuracy. Faltan métricas importantes de riesgo.

**Recomendación:**
1. Añadir **Calmar Ratio** (retorno / MDD máximo)
2. Añadir **Sortino Ratio** (solo penaliza volatilidad negativa)
3. Añadir **Omega Ratio** (ganancias vs pérdidas)
4. Añadir **Value at Risk (VaR)** y **Conditional VaR (CVaR)**
5. Reportar **retornos por clase de activo** (crypto vs equity)

### 13.6 Replicación y Experimentación

**Recomendaciones:**
1. Guardar el **hash SHA-256 exacto** del dataset usado en cada experimento
2. Guardar logs detallados de Optuna para reproducir la búsqueda de hiperparámetros
3. Ejecutar más trials (actualmente solo 3, muy pocos)
4. Utilizar el entrenamiento largo.
5. Implementar **experiment tracking** con Weights & Biases o MLflow

---

*Documento actualizado el 2026-05-24. Versión revisada con verificación exhaustiva contra código fuente, logs, `metadata.json`, `validation_cache.json`, y diffs Git (`095f572→0eeb17a→HEAD`). Se usó Kimi K2.6, GLM 5 y GLM 5.1 para la redacción.* 