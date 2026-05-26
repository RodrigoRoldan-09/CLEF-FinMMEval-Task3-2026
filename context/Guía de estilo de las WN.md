Aquí tienes una guía de estilo directa y estructurada para que puedas trabajar en tu borrador y luego copiar y pegar al documento final de CEUR-WS (basado en la plantilla `ceurart` a una columna) sin romper nada:

**Estructura y contenido obligatorio del texto** El cuerpo de tu artículo debe tocar sí o sí estos temas:

- Tareas realizadas.
- Objetivos principales de los experimentos.
- Enfoques utilizados y progreso más allá del estado del arte.
- Recursos empleados.
- Resultados obtenidos.
- Análisis de los resultados.
- Perspectivas para trabajos futuros.

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
- No incluyas los parámetros `twocolumn` ni `hf` al inicializar el documento. Tampoco hay un límite máximo de páginas establecido, pero se recomienda ser conciso.