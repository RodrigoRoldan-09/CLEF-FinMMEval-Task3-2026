# Inconsistencias entre main.tex y WORKING_NOTES_MODELO_GANADOR_V2.md

> Fecha de generación: 2026-05-26
> Estado: Requiere decisión del usuario sobre cómo resolver cada punto.

---

## 1. Integridad de Bibliografía (refs.bib)

**Estado: ✅ VERIFICADO — Sin problemas**

Todas las 16 claves de cita usadas en `main.tex` existen correctamente en `refs.bib`:

| Cita en main.tex | Estado en refs.bib |
|-----------------|-------------------|
| `araci2019finbert` | ✅ |
| `yang2023fingpt` | ✅ |
| `zhang2023fingpt` | ✅ |
| `xu2025finmultitime` | ✅ |
| `theFinAI2025` | ✅ |
| `kraskov2004` | ✅ |
| `finmem2024` | ✅ |
| `xiao2024tradingagents` | ✅ |
| `tian2025tradinggroup` | ✅ |
| `lu2025p1gpt` | ✅ |
| `hu2022lora` | ✅ |
| `schulman2017ppo` | ✅ |
| `xiao2025tradingr1` | ✅ |
| `xiong2025flagtrader` | ✅ |
| `shao2024deepseekmath` | ✅ |
| `deepseek2025deepseekr1` | ✅ |

**Acción requerida:** Ninguna. La bibliografía está completa.

---

## 2. Inconsistencias de Contenido: Dataset y Entrenamiento

### 2.1 Origen de datos de SOL (Prioridad: ALTA)

**main.tex (Sección 3.1):**
> "The first is historical market data retrieved from Yahoo Finance for three cryptocurrency assets: Bitcoin (BTC), Ethereum (ETH), and **Solana (SOL)**."

**WORKING_NOTES_MODELO_GANADOR_V2.md (Sección 5.1):**
> "Activos procesados: Training: BTC_YAHOO, ETH_YAHOO, **SOL_YAHOO**"

**Inconsistencia potencial:** El usuario cuestionó previamente si Yahoo Finance tiene noticias de SOL. Si los CSVs de SOL provienen de CLEF (no de Yahoo), entonces `main.tex` está mintiendo sobre la fuente. Si provienen realmente de Yahoo, todo está bien.

**Pregunta abierta para el usuario:** ¿De dónde provienen exactamente los datos de SOL? ¿`SOL_YAHOO` es un archivo CSV generado por el equipo o descargado directamente de Yahoo Finance? ¿Contiene noticias reales o es un placeholder?

**Impacto:** Si los revisores de CEUR-WS verifican la fuente y descubren que SOL no tiene noticias en Yahoo, el paper perdería credibilidad. Es mejor documentar con precisión la proveniencia.

---

### 2.2 Selección Manual vs. Criterio Automático del Hall of Fame

**Estado: ✅ RESUELTO en main.tex actual**

**main.tex (línea 273):**
> "While the automated search identified candidates with superior fitness (for example, a trial with fitness=1.974 and MDD=17.48%), the final deployment decision favored a model selected manually based on absolute Cumulative Return (90.67%) together with operational considerations."

**main.tex (línea 387):**
Compara explícitamente el modelo 111504 (fitness 1.974, MDD 17.48%) con el 085649 (CR 90.67%, MDD 48.05%), explicando la tensión riesgo/retorno.

**Conclusión:** El paper ya documenta honestamente que la selección fue manual y consciente, priorizando retorno absoluto sobre ajuste al riesgo. No requiere acción adicional.

---

### 2.3 Discrepancia Fast Mode vs. Validación Completa

**Estado: ✅ RESUELTO en main.tex actual**

**main.tex (línea 391):**
> "The fast-mode MDD of 0.11\% was catastrophically optimistic compared to the full-validation MDD of 48.05\%, confirming that evaluation over a single truncated fold produces an illusion of robustness."

**Conclusión:** El paper ya cuantifica explícitamente la discrepancia (0.11% vs 48.05%). No requiere acción adicional.

---

### 2.4 Número de Épocas de GRPO

**Estado: ✅ RESUELTO en main.tex actual**

**main.tex (línea 260):**
> "Epochs executed & 1 (SFT) + 1 (GRPO)"

**main.tex (línea 237):**
> "GRPO training commences for one epoch (approximately 6 minutes)"

**Conclusión:** El paper ya especifica que se ejecutó 1 época de GRPO. No requiere acción adicional.

---

### 2.5 Ground Truth Thresholds Exactos

**Estado: ✅ RESUELTO en main.tex actual**

**main.tex (línea 267):**
> "The classification thresholds, optimized via Optuna TPE with 3 trials, partition the distribution into Buy (top 35.82\%), Sell (bottom 19.18\%), and Hold (remaining 45\%)."

**Conclusión:** Los thresholds exactos ya están documentados. No requiere acción adicional.

---

### 2.6 Augmentations: Flags sin Implementación (Prioridad: MEDIA)

**main.tex:**
No menciona augmentations (correcto, ya que no se aplicaron).

**WORKING_NOTES_MODELO_GANADOR_V2.md (Sección 7):**
Documenta que `metadata.json` tiene flags `aug_block_shuffle=true`, `aug_subset_sampling=true`, etc., pero que **nunca fueron implementados** en el código.

**Consistencia:** `main.tex` es correcto al omitirlos. No hay acción requerida.

---

### 2.7 LoRA Dropout (Prioridad: BAJA)

**main.tex:**
> "Notably, the effective LoRA dropout is hardcoded to 0.0 in the training wrapper."

**WORKING_NOTES_MODELO_GANADOR_V2.md (Sección 8.1):**
> "LoRA dropout: 0 (hardcoded)... El ModelConfig dice 0.05 pero se ignora"

**Consistencia:** Coinciden. No hay acción requerida.

---

## 3. Inconsistencias de Estilo y Tono

### 3.1 Uso de "We" vs. "I"

**main.tex:**
Usa "we" en todo el paper ("we describe", "our work", "we disclose").

**Contexto real:**
El modelo documentado fue entrenado por Jona. Las arquitecturas aspiracionales (tripartite memory, dual-LLM, FIFO, 43 features) fueron propuestas por otro compañero. El paper actual presenta todo como trabajo colectivo sin distinguir contribuciones individuales.

**Decisión requerida:** ¿Es esto aceptable para el equipo? CEUR-WS no requiere sección de contribuciones individuales, pero en un Working Note es común ser transparente sobre quién hizo qué. Por ahora, mantener "we" es aceptable si todos los autores están de acuerdo.

---

### 3.2 Referencias al "Compañero"

En conversaciones previas, se mencionó que ciertas arquitecturas fueron propuestas por "el compañero". `main.tex` actualmente no distingue entre arquitecturas implementadas y no implementadas en el cuerpo del texto — todo se presenta como "our pipeline". Las arquitecturas no entrenadas ya fueron movidas a Future Work, lo cual es correcto.

**Consistencia:** Resuelto en ediciones previas. No hay acción requerida.

---

## 4. Resumen de Acciones Requeridas

| # | Inconsistencia | Estado en main.tex | ¿Requiere decisión del usuario? |
|---|----------------|-------------------|--------------------------------|
| 1 | Origen de datos SOL | ⚠️ Pendiente de verificación | ✅ Sí |
| 2 | Selección manual del modelo | ✅ Ya documentado (líneas 273, 387) | ❌ No |
| 3 | Discrepancia fast/full MDD | ✅ Ya documentado (línea 391) | ❌ No |
| 4 | Épocas GRPO = 1 (no 2) | ✅ Ya documentado (líneas 237, 260) | ❌ No |
| 5 | GT thresholds exactos | ✅ Ya documentado (línea 267) | ❌ No |
| 6 | Augmentations omitidos | ✅ Correcto así | — |
| 7 | LoRA dropout = 0 | ✅ Correcto así | — |

**Conclusión:** El `main.tex` actual ya es coherente con `WORKING_NOTES_MODELO_GANADOR_V2.md` en casi todos los puntos críticos. La única inconsistencia real pendiente es el origen exacto de los datos de SOL (punto 1).

---

## 5. Preguntas Abiertas para el Usuario

1. **Origen de SOL (único punto pendiente):** ¿`SOL_YAHOO` viene de Yahoo Finance directamente o fue generado/etiquetado por el equipo? ¿Contiene noticias reales o es un placeholder? Esto es importante porque el paper afirma que los datos de entrenamiento provienen de Yahoo Finance, y si SOL no tiene noticias en Yahoo, habría que corregir esa afirmación.

2. **Autores:** ¿Los nombres completos de autores y correos electrónicos ya están definidos? (Actualmente en main.tex: Rodrigo, Jona, Baru, David — solo nombres cortos)

3. **Diagramas:** ¿Quieren que priorice la creación de alguno de los diagramas propuestos en `propuestas_diagramas.md`? El más crítico es el "Dataset Split y Orígenes" (P0), pero también podría ayudar el diagrama de arquitectura del sistema.
