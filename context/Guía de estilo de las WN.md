Aquí tienes una guía de estilo directa y estructurada para que puedas trabajar en tu borrador y luego copiar y pegar al documento final de CEUR-WS (basado en la plantilla `ceurart` a una columna) sin romper nada:

**Estructura y contenido obligatorio del texto** El cuerpo de tu artículo debe tocar sí o sí estos temas:

- Tareas realizadas.
- Objetivos principales de los experimentos.
- Enfoques utilizados y progreso más allá del estado del arte.
- Recursos empleados.
- Resultados obtenidos.
- Análisis de los resultados.
- Perspectivas para trabajos futuros.

Nos dicen lo del laboratorio:

```
You may prepare the system description, methodology, prompts/training setup, and experimental details now, and add final leaderboard results once they are released.
```

**Títulos, Autores y Metadatos**

- **El título:** No debe contener saltos de línea. Además, todos los títulos deben mantener un estilo coherente (por ejemplo, o todos usan uso de mayúsculas enfatizadas o todos mantienen un estilo regular); no mezcles estilos.
- **Los autores:** Deben declararse de forma separada para extraer bien los metadatos. No abrevies los nombres (usa el nombre completo, e.g., "Donald E. Knuth", no "D. E. Knuth") y añade siempre el correo electrónico.
- **Abstract y Keywords:** Se deben colocar en entornos específicos (`\begin{abstract}` y `\begin{keywords}`). Las palabras clave adentro deben ir separadas siempre por el comando `\sep`.

**Secciones y numeración**

- **Debe llevar numeración:** Usa estrictamente los comandos de LATEX para las secciones (`\section`, `\subsection`, `\subsubsection` y `\paragraph`) y **no les quites la numeración**.
- **No inventes títulos:** Queda totalmente prohibido simular un título de sección usando simplemente texto en negrita o cursiva en un nuevo párrafo.

**Tablas y Figuras**

- **Las tablas:** Han de usar los estilos del paquete `booktabs`. El título (caption) de la tabla va obligatoriamente **arriba** del contenido. Si la tabla es muy ancha y ocupa toda la página, debes usar el entorno `table*` en lugar de `table`.
- **Las figuras:** El título (caption) va obligatoriamente **debajo** de la imagen. Debes incluir una descripción textual pensada para lectores de pantalla (por accesibilidad) y si la imagen es de un tercero, hay que dejarlo claro en el texto.

**Matemáticas y Ecuaciones**

- Usa el formato tradicional `$` o entorno `math` para fórmulas en línea. Para ecuaciones destacadas y numeradas usa el entorno `equation`, y si no quieres que estén numeradas, usa `displaymath`.

**Citas y Bibliografía**

- Usa **BibTeX** obligatoriamente para tus referencias.
- De nuevo, asegúrate de que en el archivo `.bib` los nombres de los autores estén completos y no uses únicamente iniciales.

**Agradecimientos (Acknowledgments)**

- Se debe colocar la sección de agradecimientos **justo antes** de las referencias.
- **Importante:** No debes usar el comando `\section` (ni numerado ni sin numerar) para esto. Tienes que usar el entorno cerrado `\begin{acknowledgments} ... \end{acknowledgments}`. Aquí debes nombrar cualquier fuente de financiación o apoyo.

**Declaración sobre Inteligencia Artificial (IA)**

- Se debe nombrar qué IA usaste y detallarlo mediante una declaración formal obligatoria.
- **Si usaste IA:** Debes especificar el nombre de la herramienta (e.g., "X-GPT-4", "X-AI-IMG"), para qué se utilizó (e.g., "revisión gramatical", "generar la figura 3"), e incluir una cláusula asegurando que revisaste y editaste el contenido y te haces responsable del mismo.
- **Si no usaste IA:** Debes escribir explícitamente la frase: "The author(s) have not employed any Generative AI tools".

**Apéndices**

- Si los tienes, deben ir justo al final, antes del comando `\end{document}`, y deben iniciar con el comando `\appendix` para que a partir de ese punto las secciones se nombren con letras (A, B...) en lugar de números.

**ESTRICTAMENTE PROHIBIDO (Para evitar que se rompa el estilo del original)**

- Se debe cuidar de **no cambiar el estilo** base de la clase `ceurart`: está rotundamente prohibido ajustar los márgenes, cambiar el tamaño de letra, modificar el interlineado, cambiar definiciones de párrafos o listas, y **no puedes usar el comando `\vspace`** para forzar o ajustar espacios verticales de forma manual.
<<<<<<< HEAD
- No incluyas los parámetros `twocolumn` ni `hf` al inicializar el documento. Tampoco hay un límite máximo de páginas establecido, pero se recomienda ser conciso.
=======
- No incluyas los parámetros `twocolumn` ni `hf` al inicializar el documento. Tampoco hay un límite máximo de páginas establecido, pero se recomienda ser conciso.


 Las instrucciones oficiales del FinMMEval Lab recomiendan estrictamente que, para referirte al laboratorio y a tu tarea, debes citar los artículos de resumen oficiales (Overview papers) una vez que estén disponibles, y no solo la página web.


 # TASK 3


 Task 3 - Financial Decision Making
Task 3 is a daily, news-driven trading workflow. We collect market news each day, call each submitted endpoint once, and execute positions from the returned action: BUY, HOLD, or SELL.

Data & Submission
Historical data for backtesting, validation, and model training: MBZUAI/finmmeval-lab-clef2026.

To participate, submit your endpoint via the Agent Market Arena page (Google Form): Agent Market Arena.

At present, we do not enforce a hard submission cap for Task 3. Teams may update their submitted endpoint as needed before the endpoint submission deadline, but should avoid unnecessary rapid resubmission.

The endpoint submission deadline is the deadline for submitting or updating your endpoint. It is not the end of the Task 3 evaluation period.

Submitting the Google Form registers or updates your endpoint, but it may not appear on the leaderboard immediately. Organizers first verify submitted endpoints and confirm the final endpoint list.

Daily Workflow
Each endpoint receives one request per day and must return one action: BUY, HOLD, or SELL.
Position mapping: BUY -> long, HOLD -> flat, SELL -> short.
Execution rule: each new action fully replaces the previous day’s position.
Price convention: daily close price.
Scheduling & Request Policy
Daily process starts at 00:00 UTC.
Requests are sent progressively after start time.
Official Task 3 performance is computed over a common evaluation window for all accepted endpoints, rather than starting separately from each team’s individual form submission date.
Submitted endpoints will continue to be called daily after the endpoint submission deadline for the official Task 3 evaluation window. We expect this window to run through late June or early July, aligned with the final lab reporting schedule.
Teams do not need to keep endpoints online for the full day, but should start them shortly before 00:00 UTC and keep them available for several hours to allow for queued requests, retries, and temporary network delays.
Per-request timeout is 3 minutes.
If a request fails (timeout, server error, or invalid response), the action defaults to HOLD.
Request / Response Format
Input is a JSON object containing date, price, news, symbol, momentum, history_price, and optional 10k/10q (object or null). The symbol varies by asset (e.g., TSLA, BTC).

{
  "recommended_action": "BUY"
}
Valid actions: BUY, HOLD, SELL.

Input Samples
Sample A (TSLA):

Note: 10k and 10q can each be an object or null.

{
  "date": "2025-01-15",
  "price": {"TSLA": 250.50},
  "news": {"TSLA": ["Tesla announces new production milestone"]},
  "symbol": ["TSLA"],
  "momentum": {"TSLA": "bullish"},
  "10k": {"TSLA": ["[SEC 10-K Filing - 2025-01-15]\nSummary..."]},
  "10q": {"TSLA": ["[SEC 10-Q Filing - 2025-01-15]\nSummary..."]},
  "history_price": {"TSLA": [
    {"date": "2025-01-12", "price": 249.80},
    ...,
    {"date": "2025-01-13", "price": 250.50},
    {"date": "2025-01-14", "price": 250.30}
  ]}
}
Sample B (BTC):

{
  "date": "2025-01-15",
  "price": {"BTC": 67890.50},
  "news": {"BTC": ["Bitcoin ETF inflows remain strong"]},
  "symbol": ["BTC"],
  "momentum": {"BTC": "neutral"},
  "10k": null,
  "10q": null,
  "history_price": {"BTC": [
    {"date": "2025-01-12", "price": 67580.00},
    ...,
    {"date": "2025-01-13", "price": 67720.00},
    {"date": "2025-01-14", "price": 67810.00}
  ]}
}
cURL Sample
curl -X POST "<your_endpoint>" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2025-01-15",
    "price": {"TSLA": 250.50},
    "news": {"TSLA": ["Tesla announces new production milestone"]},
    "symbol": ["TSLA"],
    "momentum": {"TSLA": "bullish"},
    "10k": {"TSLA": ["[SEC 10-K Filing - 2025-01-15]\nSummary..."]},
    "10q": {"TSLA": ["[SEC 10-Q Filing - 2025-01-15]\nSummary..."]},
    "history_price": {"TSLA": [
      {"date": "2025-01-12", "price": 249.80},
      ...,
      {"date": "2025-01-13", "price": 250.50},
      {"date": "2025-01-14", "price": 250.30}
    ]}
  }'
Data & Submission
Historical data for backtesting, validation, and model training: MBZUAI/finmmeval-lab-clef2026.

To participate, submit your endpoint via the Agent Market Arena page (Google Form): https://huggingface.co/spaces/TheFinAI/Agent-Market-Arena.

Reference endpoint example (FastAPI): examples/simple_trading_api.py.

Evaluation
Primary: Cumulative Return (CR)

Secondary: Sharpe Ratio (SR), Maximum Drawdown (MD), Daily Volatility (DV), and Annualized Volatility (AV)
>>>>>>> origin/Joni_rama
