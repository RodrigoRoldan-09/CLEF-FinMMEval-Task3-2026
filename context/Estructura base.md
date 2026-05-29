Te redacto lo que creo que podría ser, dime qué te parece. Será un borrador muy borrador, con muchas faltas a la gramática, pero para que me digas si la estructura te parece correcta.

# Introducción
Se participó en el task 3 del CLEF, que pide [resumen de lo que pide la página]. Por tanto se optó por un enfoque incremental. En primer lugar se preparó el dataset, descargando el del año anterior y el que se tenía para este año. Así mismo se añadió información de Yahoo Finance para fechas anteriores y se le adjuntó resúmenes de noticias externas, esto con el fin de aumentar el dataset de entrenamiento y por tanto mejorar la calidad del modelo. Sin embargo, como se verá más adelante, dada las restricciones de tiempo y el consumo de tiempo de entrenamiento en el hardware disponible no fue posible utilizarlo al completo.
Una vez obtenido el dataset de entrenamiento, se procedió a buscar la solución del laboratorio. Quedó claro luego de una investigación que se podía usar dos enfoques para datos multimodales. Uno podría ser utilizar un espacio multidimensional donde codificarlos todos y combinarlos en ese espacio latente y tener una cabeza que se entrenara con RL o aprendizaje supervisado. En segundo lugar, usar un LLM como cabeza de decisión que ya posee en su espacio latente la capacidad de procesar los datos multimodales (noticias y números en distintos órdenes de magnitud), resultando en un enfoque de RL para mejorar su capacidad en la tarea planteada. Dado que el laboratorio tiene un enfoque con LLM, se optó por lo segundo. 
[Colocar los antecedentes]
[Agregar resumen de lo que sigue]
# Metodología
Entonces se procedió al proceso de transfer learning por medio de aprendizaje reforzado. Sin embargo, dada la restricción del hardware, se optó por optimizar y combinar los avances de trabajos anteriores, con un objetivo de minimizar el consumo de VRAM hasta que fuera capaz de caber en los 12 GB de VRAM de una RTX 3080ti.
Para que las iteraciones en busca de los mejores hiperparámetros no fueran eternas, se optó por un enfoque dual, primero se haría una exploración con una versión reducida del dataset, para después probar los mismos en un entrenamiento largo. No obstante, dado que la búsqueda de hiperparámetros sufrió retrasos debido al desvanecimiento del gradiente en el proceso de programarlo y a underfitting causado por encontrar la solución estadísticamente más fácil. Ambos problemas requirieron un análisis que consumió tiempo y el modelo entregado es un modelo obtenido en esa búsqueda, con un dataset reducido y una época de entrenamiento, sorprendiendo en rendimiento en la validación aunque no siendo muy exitoso en el rendimiento preliminar observado en la arena, con toda razón, sin embargo es una base capaz de mejorar en posteriores iteraciones en contextos donde se use hardware de consumo.

## Enfoques utilizados y el progreso más allá del estado del arte
El proceso para la creación del sistema fue incremental, luego en el proceso de realizar los experimentos fue iterativo. Una vez que se realizó la investigación del estado del arte y habiendo elegido el enfoque con un LLM al que se le realiza transfer learning, se procedió a programar e iterar. 

Nuestro progreso más remarcable sobre el estado del arte es tomar las diversas optimizaciones realizadas por ellos y adaptarlos para un hardware con menor cantidad de memoria RAM. En el estado del arte, existen sistemas como [hablar de la investigación donde se ocupa hardware más poderoso]. 
[Hablar sobre lo que sucedió con el colapso del modelo y qué se aprendió]
```
3. El colapso de tu modelo: El Equilibrio de Nash

Cuando implementaste por primera vez tu matriz (inspirada en la original), te encontraste con que tu modelo no aprendía nada y siempre predecía **HOLD**. En tu autopsia de los datos descubriste que el "Valor Esperado" de predecir HOLD era -0.0945, el de BUY era -0.3856 y el de SELL era -1.1040.

Aquí entramos a un concepto fundamental de la Teoría de Juegos: el **Equilibrio de Nash**.

- _Definición mínima:_ Es una situación en la que un jugador (o el modelo) se da cuenta de que cambiar su estrategia actual solo empeorará las cosas, por lo que decide no moverse.
- _Explicación intuitiva:_ Imagina un campo minado. El modelo sabe que si da un paso a la izquierda (BUY) o a la derecha (SELL) y se equivoca, la matriz original le explotará en la cara con un -2.0 o -2.25. Pero si se queda quieto (HOLD) y el mercado se mueve, la matriz original solo le daba una "bofetada" de **-1.0**. El modelo, siendo un genio estadístico, calculó el _Valor Esperado_ y concluyó: _"No entiendo este mercado, pero si me quedo quieto (HOLD), pierdo mucha menos sangre que si intento adivinar"_. Así que colapsó en la inacción.
```
Las decisiones de diseño se rigieron por la siguiente pregunta: ¿Cómo obtener la mayor calidad con la menor cantidad de VRAM consumida? Entonces, en consecuencia se eligió GRPO sobre PPO, dado que este es más eficiente en memoria, puesto que a diferencia de PPO, no se debe aumentar el tamaño en memoria añadiendo los pesos del modelo supervisor, sino que sólo con las respuestas del mismo modelo se puede mejorar. Así mismo, se redujeron los tokens de salida del modelo, esto se logró de dos maneras, realizando un preentrenamiento que consiste en [hablar del preentrenamiendo de etiquetas y de reducción de salida, incluye el prompt builder específico para apoyarlo, hablando que incluye 10k y 10q si hay, así como las estrategias para poder recuperar etiqueta verdadera].
[Colocar la descripción estrategia del entrenamiento del Lora, puesto que es fundamental.]
El tiempo de entrenamiento de modelos es conocido por tardar días. Luego, también es sabido que la búsqueda los hiperparámetros es una parte fundamental del entrenamiento de cualquier sistema de ML. Con esto en mente, realizar iteraciones rápidas de búsqueda con una estrategia de búsqueda también es una solución usual. La estrategia elegida fue Optuna, la cual [explicar optuna de manera resumida]. Además, Optuna requiere una configuración para realizar su labor. [describir la configuración de optuna].  La cual se une a un proceso de evaluación que guarda los mejores modelos con una heurística [explicar que utilizamos nuestro score propio para combinar CR y SR, que eran las métricas principales al comienzo del task 3, así como el guardado automático de los que pasaran el umbral, mencionar todas las métricas y explicar que no usamos volatilidad por estar inmersa en el SR, queriendo priorizarlo omitimos Daily Volatility (DV) y Annualized Volatility (AV) en la selección del modelo].
Es fundamental en este punto describir el espacio de búsqueda de hiperparámetros. [Colocar los datos del espacio].
Además de lo anterior, para aumentar la velocidad y mantener la equivalencia de rendimiento en entrenamientos largos, es preciso evitar [colocar las decisiones de diseño que se tomaron para aumentar la velocidad pero que mantienen la equivalencia de rendimiento en los entrenamientos largos.].
Finalmente cuando se evalúa el modelo en el contexto de Optuna, para aumentar aún más la velocidad de cada ciclo, se divide el dataset de validación en tres partes y sólo se usa uno para validar, la división no es aleatoria para garantizar que todos los trials tengan el mismo conjunto de validación [colocar la información de cómo está distribuido las seeds, que son las fijas para el experimento y dinámicas para el TPESampler del Optuna, la tabla de seed ayudaría mucho]. 
```

|Trial|Optuna Seed|Global Seed|Resultado|
|:-:|:-:|:-:|---|
|Trial 0|2|42|HP diferentes, datos idénticos, modelo idéntico|
|Trial 1|3|42|HP diferentes, datos idénticos, modelo idéntico|
|Trial 2|5|42|HP diferentes, datos idénticos, modelo idéntico|

```

Además, cabe recalcar que aunque la estrategia antes descrita nos puede llevar a regiones prometedores que por estadística son más robustas, con tan pocos trials realizados en cada *Study* que Optuna no nos ofrece grandes beneficios en términos de conocer el espacio de los hiperparámetros, puesto que TPE posee un hiperparámetro interno llamado `n_startup_trials` (cuyo valor por defecto suele ser 10), Durante estos primeros ensayos, el modelo bayesiano **no calcula absolutamente nada**; se limita a realizar un muestreo puramente aleatorio (_Random Search_) para poblar su base de datos inicial. Por lo que en realidad, aunque se usó Optuna, se debe hablar con honestidad: por haber realizado 3 trials en el test del sistema, realmente fue Random Search el responsable de encontrar el modelo ganador. Se deja el realizar más trials al trabajo a futuro.
El modelo fue obtenido en una de las iteraciones de búsqueda, que como se ha dicho, esta diseñado para ser rápido. Puesto que se realiza una pequeña evaluación al final del paso de búsqueda, el modelo fue guardado como un lora que se puede cargar. Como no fue posible entrenar un modelo con todo el dataset, no se tuvo más opción que colocar el mejor modelo en una validación completa para colocarlo en el endpoint.


## Recursos empleados
Para la realización del entrenamiento se empleó el dataset del concurso, se unieron los datos de este año con los del año pasado, además se aumentó con datos históricos [colocar la información de los datos extras, así como las noticias añadidas a ellos y las fuentes de las que provienen].
Los modelos empleados fueron [listar modelos], sin embargo, el modelo ganador fue Phi-4 en su versión [listar los datos precisos]. Estos fueron escogidos especialmente porque en el cálculo del tamaño en memoria, se podía desplegar en los 12 GB de VRAM disponibles. [Describir el cálculo de cómo un modelo cuando se entrena es más pesado]
En el entrenamiento se usó el siguiente hardware [describir el hardware con precisión]. En donde destaca la GPU Nvidia, puesto que sin ella, los tiempos de entrenamiento serían aún más tardados.
Además se utilizaron las siguientes librerías [describir las librerías y su uso en el proyecto, si es demasiado colocarlo en el anexo]
```
 neofetch
██████████████████  ████████   takeishi@kadragon-PC 
██████████████████  ████████   -------------------- 
██████████████████  ████████   OS: Manjaro Linux x86_64 
██████████████████  ████████   Kernel: 5.15.202-1-MANJARO 
████████            ████████   Uptime: 15 days, 13 hours, 37 mins 
████████  ████████  ████████   Packages: 2377 (pacman), 45 (flatpak), 19 (snap) 
████████  ████████  ████████   Shell: bash 5.3.9 
████████  ████████  ████████   Terminal: /dev/pts/2 
████████  ████████  ████████   CPU: AMD Ryzen 9 5900X (24) @ 3.700GHz 
████████  ████████  ████████   GPU: NVIDIA GeForce RTX 3080 Ti 
████████  ████████  ████████   Memory: 5507MiB / 32006MiB 
████████  ████████  ████████
████████  ████████  ████████                           
████████  ████████  ████████                           
```


## System Card

### Arquitectura
El flujo de información en el entrenamiento sigue los pasos descritos en [[figura_pipeline_entrenamiento]], los cuales son, para mayor claridad aquellos componentes que finalmente dieron por resultado el modelo obtenido. Se podría decir en resumidas cuentas que  sería una arquitectura de model stacking en donde el LLM actúa como un nodo de fusión.
En donde primero se obtienen los datasets de manera cruda. Estos son de naturalezas distintas, los del task 3 ya están curados, sin embargo al hacer el aumento de datos, estos necesitan por un lado descargarse y unirse con los resúmenes de noticias de aquel día obtenidos de [mencionar las fuentes]. Es necesario unirlos y colocarlos en un dataset unificado que proporcione la base del entrenamiento. Una vez unidos, se filtra aquellos que poseen noticias y aquellos que no, esto con el objetivo de entrenar sólo con datos que son lo más parecidos a los que el modelo verá en su evaluación. Una vez logrado esto se divide el dataset en Tain, Val y Test, siguiendo el estándar. Train se utilizará para entrenar las matrices A y B del Lora para la tarea actual, Val se usa para la validación en el entrenamiento y con el fin de no tener data leak, se usa Test en la validación final. En el momento del entrenamiento se utilizan los datos del dataset, pero no es posible mandarlos directamente al modelo, se debe construir los prompts que serán las entradas que el modelo pueda leer. Se deben construir en el momento para tener la flexibilidad de cambiar el formato cuando se desee, sin embargo, si se tiene un tipo o si sólo se usa uno se podrían generar y almacenar para reducir los cálculos fuera del tiempo de ejecución. Ahora que el modelo consume los prompts genera salidas y por medio del Loss [colocar la fórmula usada] se actualizan los pesos del LoRa. Cuando se ha terminado una época, se valida en un sólo Fold para mayor velocidad y Optuna continúa la búsqueda según esos resultados. Se finaliza cuando Optuna ha utilizado las N semillas en con M trials [debe reflejarse en el diagrama]. 

El endpoint toma la información mandada por el json y [colocar la descripción del pipeline del endpoint.]. [[grafico_endpoint]]. Como se puede observar, en este caso se añade la información que produce el modelo Chronos junto a todos los demás datos. Además, en retrospectiva, representa un riesgo por la discrepancia entre lo que el modelo vio en el entrenamiento y lo que realiza en el mundo real.

## Uso de datos
El esquema de creación del dataset se encuentra gráficamente en [[figura_dataset]]
```
PIPELINE DE CONSTRUCCIÓN:

  1. CSVs crudos en data/ (noticias incluidas)
     ├─ {asset}_texto.csv       (date, news, has_news, filing_10k/q, momentum, target)
     ├─ {asset}_escalares.csv   (date, log_return_t1, volatilidad_diaria, future_price_diff)
     └─ {asset}_temporal.csv    (para historial de precios — no usado en build actual)
        ↓
        [Filtro: has_news == 1] → solo filas con noticias
        ↓

  2. Enriquecimiento FinBERT (sentimiento)
     ├─ Lee caché: models/{asset}_sentiment.csv
     ├─ Si no existe: ejecuta SentimentExtractor (FinBERT) y guarda caché
     └─ Añade: sentiment_label, sentiment_score, prob_positive/negative/neutral
        ↓

  3. Merge con escalares + reconstructión de precio
     ├─ Fusiona _texto + sentiment + _escalares por fecha
     ├─ Reconstruye price_close desde log_return_t1 acumulativo
     ├─ Genera price_history (últimos N precios)
     └─ Añade: price_close, price_history, log_return_t1, volatilidad_diaria
        ↓

  4. Cálculo de Ground Truth (BUY/HOLD/SELL)
     ├─ Formula: signal = log_return_t1 / rolling_std(volatility_window)
     ├─ Top percentiles (40%) → BUY
     ├─ Bottom percentiles (15%) → SELL
     ├─ Resto (45%) → HOLD
     └─ Añade: ground_truth (etiqueta de entrenamiento)
        ↓

  5. Asignación de split temporal (train/val/test)
     ├─ Si fecha en OOS (task_3_end) o source=="clef" → "test"
     ├─ Si fecha >= val_start_approx → "val"
     └─ Resto → "train"
        Añade: split
        ↓

  6. Concatenación por activo → Parquet + metadata
     ├─ Concatena todos los activos (TRAINING_ASSET_KEYS)
     ├─ Guarda: data/parquet/lab_dataset.parquet
     └─ Guarda: data/parquet/dataset_card.json (metadatos y estadísticas)



```
Como ya se mencionó, lo datasets fueron descargados de [colocar aquí las fuentes, queda pendiente saber cuál fue el origen de todos y cada uno de los datos puesto que no hice esa parte.]. De los cuales se usaron [colocar una tabla con los datasets usados y qué características tienen]. Se filtran por medio de la bandera has_news, para ocupar sólo los datos que tengan noticias. Entonces, las noticias es lo que se agrega primero.

Luego de eso se coloca el conocimiento de FinBert. Dado que este modelo pueden proporcionar análisis especializados de los datos que puedan servirle al modelo principal de guía. FinBert se encarga de otorgar una [colocar lo que da]. En el caso de prob_positive/negative/neutral, estos se almacenan por si se requieren en el futuro pero no se usan en el prompt builder final. Cabe aclarar que el prompt builder está preparado para recibir más datos datos, por ejemplo Chronos [colocar lo que da]. En este caso se añadió posteriormente en el procesamiento del endpoint.

A continuación se añade la información: price_close, price_history, log_return_t1, volatilidad_diaria. El precio de cierre es fundamental puesto que es la base, el histórico se construye para que cada prompt tenga el contexto de los días anteriores, el log_return_t1 es [colocar la teoría y la fórmula: log_return_t+1 = ln(Price_t+1 / Price_t) el rendimiento logarítmico que **ocurrirá mañana**, calculado hoy.] que se utiliza en el cálculo posterior del Ground Truth (signal = log_return_t1 / volatilidad_diaria  # Normalizar por riesgo). Por último volatilidad_diaria se coloca pero finalmente no se usa.

Ahora estamos listos para añadir las etiquetas que después nos servirán en el GRPO, el ground_truth, se calcula [añadir los pormenores del cálculo y su razón de ser en el GRPO] . En resumen, cuando GRPO compara la acción del modelo contra el _Ground Truth_ en tu matriz asimétrica, le está enseñando directamente a buscar retornos robustos y seguros, ignorando el ruido.

Ahora ques se tienen todos los datos se divide por medio de la etiqueta split. [Explicación somera del por qué se realiza esto así.]

Finalmente, sólo queda empaquetar el dataset y agregar los metadatos con el fin de que al importarlos sea mucho más fácil de obtener la información del mismo. Se puede observar un resumen en la tabla siguiente.

```
Columnas del Parquet resultante:
    date, asset, symbol, source,                  # Identificadores
    price_close, price_history, momentum,         # Precio y dirección de mercado
    news,                                          # Texto de noticias (ya resumido)
    filing_10k, filing_10q,                        # Filings SEC (opcional)
    sentiment_label, sentiment_score,              # FinBERT
    prob_positive, prob_negative, prob_neutral,
    chronos_direction, chronos_confidence,         # Chronos (opcional — no integrado aún)
    log_return_t1,                                 # Retorno siguiente día (para GT)
    volatilidad_diaria,                            # Volatilidad diaria
    future_price_diff,                             # Diferencia de precio futura
    ground_truth,                                  # BUY / HOLD / SELL (precomputado)
    asset_class,                                   # "crypto" | "equity"
    annualization_factor,                          # 365 o 252
    split                                          # "train" | "val" | "test"

Regla de inclusión:
    Solo se incluyen filas donde has_news == 1.
    Las filas sin noticias se descartan silenciosamente.
    Activos training_eligible=False se incluyen en el card pero NO en el Parquet principal.

Idempotencia:
    Si el Parquet existe y los CSVs fuente no cambiaron (verificado por hash),
    no re-procesa. Usa force=True para reconstruir.
"""
```
### FinBert

[Colocar toda la información técnica de Finbert]
#### Chronos

[Colocar toda la información técnica de Chronos]


## Riesgos
Debido a una discrepancia en la implementación del código final, Chronos opera en el _endpoint_ sin haber estado en el entrenamiento del _pipeline_ previo. Esto introduce un riesgo de desalineación entre el rendimiento observado durante el _backtesting_ interno y las métricas oficiales en vivo, ya que el modelo toma decisiones financieras basándose en inferencia directa que no es equivalente al entrenamiento al 100%. 

Otra cosa a destacar es que en el conjunto de entrenamiento rápido, no se incluye ningún activo del tipo equity, lo cual hace que el modelo tenga un menor rendimiento en en este tipo de activos y que además no sea un modelo más general. Esto sumado al hecho de haber entrenado en una sola época, se puede decir que el modelo es aún muy inmaduro y no es lo que podría llegar a ser. 

Así mismo, dado que Optuna no ejecutó Trials más grandes provoca que los hiperparámetros no sean robustos sino producto de un Grid Search. Esto se puede manejar en futuras iteraciones.

También se debe pensar en que se buscaba optimizar un fitness que era un cálculo compuesto de métricas que en principio eran las  principales. Sin embargo, la elección del modelo fue guiada finalmente por la métrica principal del Task 3. Por tanto el modelo fue optimizado para un mundo y luego seleccionado para cumplir su ron en otro. 

## Reproducibilidad

[Colocar todas las métricas que se ocuparon, semillas ]


# Resultados y Análisis 

[Colocar los resultados]
[Eliminar las filas sin resultados]
[Agregar y contrastar los datos.]
```
                                        Checkpoints — métricas de validation                                        
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━┳━━━━━━━━━━━━━━┓
┃    # ┃ ID                         ┃      Fit ┃    SR ┃   MDD ┃    CR ┃   Acc ┃    WR ┃   PF ┃ Ext ┃ Modelo       ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━╇━━━━━━━━━━━━━━┩
│    0 │ exp_20260506_085649_254046 │  0.107 ✓ │  2.91 │ 48.0% │  0.91 │ 38.3% │ 54.8% │ 1.51 │ ✅  │ phi-4-mini-i │
│    1 │ exp_20260508_115709_051aed │ -0.443 ✓ │  1.40 │ 38.5% │  0.23 │ 39.0% │ 51.9% │ 1.25 │ ✅  │ qwen2.5-3b-i │
│    2 │ exp_20260506_004508_27f172 │ -0.076 ✓ │  3.04 │ 51.2% │  0.90 │ 38.9% │ 55.8% │ 1.57 │ ✅  │ phi-4-mini-i │
│    3 │ exp_20260508_014550_f9ad37 │ -0.876 ✓ │  1.41 │ 42.9% │  0.28 │ 40.2% │ 50.9% │ 1.22 │ ✅  │ qwen2.5-3b-i │
│    4 │ exp_20260509_113530_8bc775 │  1.213 ✓ │  1.21 │ 14.1% │  0.09 │ 41.0% │ 52.9% │ 1.70 │ ✅  │ phi-4-mini-i │
│    5 │ exp_20260508_062606_1cd3e0 │  0.000 ✓ │  0.00 │  0.0% │  0.00 │ 43.3% │  0.0% │    ∞ │ ✅  │ phi-4-mini-i │
│    6 │ exp_20260507_221904_9bb326 │ -2.517 ✓ │ -0.21 │ 43.1% │ -0.03 │ 43.6% │ 50.9% │ 0.93 │ ✅  │ phi-4-mini-i │
│    7 │ exp_20260509_113530_8bc775 │ -0.608 ✓ │ -0.11 │ 25.0% │ -0.01 │ 42.1% │ 51.2% │ 0.97 │ ✅  │ phi-4-mini-i │
│    8 │ exp_20260507_111504_33de19 │  1.974 ✓ │  1.97 │ 17.5% │  0.19 │ 41.8% │ 58.1% │ 1.69 │ ✅  │ phi-4-mini-i │
│    9 │ exp_20260508_100749_12f83a │  0.000 ✓ │  0.00 │  0.0% │  0.00 │ 43.3% │  0.0% │    ∞ │ ✅  │ qwen2.5-3b-i │
│   10 │ exp_20260505_204602_88701e │ -1.446 ✓ │ -0.20 │ 32.5% │ -0.02 │ 43.7% │ 48.9% │ 0.91 │ ✅  │ phi-4-mini-i │
└──────┴────────────────────────────┴──────────┴───────┴───────┴───────┴───────┴───────┴──────┴─────┴──────────────┘
```
Dados los resultados y la metodología para obtenerlos podemos decir que gran parte del trabajo fue gracias a la inteligencia de base que posee Phi 4. Puesto que con una sola época no se puede esperar mucho cambio y sin embargo logró realizar un notable avance sobre los otros modelos ocupando muchos de los lugares a pesar de haber sido Grid Search en realidad. Entonces, para futuras iteraciones se puede dejar como único modelo para hacer más pequeño el espacio de los hiperparámetros. Entonces, a través de las iteraciones y mejoras con optuna se pudo llegar a una que era capaz de entrenar modelos capaces en cierta medida en un hardware de consumo, por lo que nuestro principal aporte aún está en pie, sin embargo hay mucho camino que mejorar.



# Conclusión

[Colocar resumen]

## Trabajo a futuro

Probar más prompt builders
Y almacenarlos es eficiente así que deberíamos almacenar los prompts si no vamos a cambiar.
Randomizar los fold de evaluación y buscar la manera que sean rápidos y reproducibles pero a la vez más robustos.
Aumentar los trials de Optuna.
Data augmentation.
Probar si quitando o añadiendo modelos de preprocesamiento ayuda o no.
Utilizar Chronos en el entrenamiento.
Hacer entrenamientos largos.
Investigar y aplicar conceptos de Reward Shaping, de modo que los parámetros de la matriz sean mucho más robustos.
Utilizar Phi 4 como base.
