# 001.- Curso de IA Generativa. El futuro de la programación.

Del curso de Jesus Conde :

https://www.youtube.com/playlist?list=PLEtcGQaT56cgyp-8fOIu_JE3D_4aClmDL

Generative
Pre-trained
Transformer

Llya Sutskever - Entrenamiento de Redes Neuronales Recurrentes

# 005.- Curso de IA Generativa. Auto-Atención con Cabezas múltiples.

La arquitectura Transformer a través de la denominada auto-atención permite que cada elemento de una secuencia de entrada calcule su relevanci respecto a todos los demás elementos. Esta auto-atención se aplica a través de múltiples cabezas que se centran en aspectos relevantes, como pueden ser reglas gramaticales concretas en los PLN. Vemos como hacerlo a través de un ejemplo práctico que usa El modelo BERT, Pytorch, y la librería bertviz que permite visualizar de modo interactivo lo que está haciendo cada cabeza y cada capa de las 144 con que cuenta este modelo pre-entrenado.

https://www-nlp.stanford.edu/pubs/clark2019what.pdf

# 006.- Curso de IA Generativa. Transferencia de Aprendizaje y PyTorch.

Antes de empezar con proyectos más prácticos, vemos los dos elementos clave que nos quedan por ver. La transferencia de aprendizaje y el modo de comunicarse con el modelo.  La IA generativa entrena modelos para que lleven a cabo tareas. En la mayoría de los casos esos modelos no empiezan de cero, sino que lo hacen utilizando modelos pre-entrenados ya existentes y llevando a cabo lo que se conoce como transferencia de aprendizaje,

https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbTQxZ3JjZ2ZLOWJtMkt6MDl2WnJ4OEFabV9Pd3xBQ3Jtc0tteXVFODhkY3B2M1huVjJiRGxnQUVIMnJCWDZBbkc2bnhTT2JTNzl1Y3lLMVQ5c1RxRWxjazFVN3o3OGJqaDJtWmk4V1BqZy1JYlpPa3h2SzBpc19UYjhsanFQdkUxSW12SDgyTkZRaDgyVzFOR19GOA&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1-zoFP9SuYzDkHogfi7qAcS9J4Zt33SX_%3Fusp%3Dsharing&v=FDfbHIP5p5g


# 007.- Curso de IA Generativa. Funcionamiento del modelo BERT

Vemos en detalle el modelo BERT, uno de los modelos basados en la Arquitectura Transfomer, especializado para la Comprensión del Lenguaje Natural.  Vemos lo que significa que sea un Modelo Bidireccional, que usa solo el codificador del Transformer y como crea las representaciones o incrustaciones vectoriales de sus tokens.

> **2017 Attention is all you need**

Bidirectional Encoder Representation from Transformer

# 008.- Curso de IA Generativa. Uso práctico de BERT

Importamos y usamos un modelo BERT al que entrenamos con una nueva secuencia y vemos como se realiza todo el proceso. Usamos la librería scikit-learn para poder aplicar la función similitud coseno; Vemos los parámetros con nombre de BERT y como acceder a ellos; El pooling o capa densa que se aplica al token [CLS] en la capa de salida. Vemos también porque los modelos de redes neuronales suelen trabajar con matrices de tres dimensiones, ¿Por qué? y de que modo se relaciona con el uso de la función unsqueeze( ) de Pytorch.

# 009.- Curso de IA Generativa. BETO, el BERT en español.

Vemos en este video la tokenización y BETO, un modelo BERT entrenado desde cero con un corpus de datos exclusivamente en español. Vemos las distintas técnicas de tokenización que se pueden aplicar a los Modelos de Lenguaje Grandes y las usadas por BERT y GPT. Vemos ejemplos prácticos sobre Colab de como BETO, el BERT en español se entrena a través de técnica de Enmascaramiento de Palabras Completas y diferencias entre la tokenización en el modelo en inglés y en español.

https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased

# 010.- Curso de IA Generativa. Los múltiples embeddings de BERT.

Vemos como BERT crea la matriz de incrustación para pasar a la primera capa oculta. Como combina tres embeddings: word_embeddings; position_embeddings y token_type_embeddings. Vemos la finalidad de cada uno y un ejemplo práctico directamente desde Colab.

# 011.- Curso de IA Generativa. La Arquitectura de BERT.

Vemos los componentes y tareas fundamentales en el LLM BERT. Desde la secuencia de entrada hasta la formación de la matriz de salida por codificador a través de la capa feed-forward, analizando las proyecciones lineales y operaciones de activación no lineales que tienen lugar en ella.

# 012.- Curso de IA Generativa. Modelado del Lenguaje Enmascarado.

MLM - Masked Language Modeling

La librería Transformers incluye, además de los modelos base BERT, una serie de cabezas adicionales que permiten realizar distintas tareas adicionales. Vemos hoy como realizar una de ellas: la tarea MLM de Masked Language Modeling, la tarea de Modelado del lenguaje enmascarado, que permite introducir una secuencia de entrada con el uso de un tipo de token especial denominado "[MASK]". Vemos el uso de la Clase "BertForMasKedLM" que permite añadir esa cabeza sobre las 12 capas de BERT y el uso de la función "pipeline" que nos permite automatizar secuencias de procesamiento.

ejemplos/La_tarea_de_modelado_de_lenguaje_enmascarado.ipynb

# 013.- Curso de IA Generativa. Predicción de la Siguiente Sentencia (NSP).

La Next Sentence Prediction(NSP) es otra de las cabezas adicionales de BERT que permite determinar si una sentencia sigue lógicamente a la anterior o no. Es junto con la tarea de Modelado del lenguaje enmascarado las dos que se utilizan en los modelos BERT pre-entrenados. Vemos su funcionamiento y un Colab práctico con ejemplos de su uso a los que puedes añadir tus propias oraciones para comprobar las predicciones del modelo de modo práctico.

ejemplos/Tarea_de_Predicción_de_la_Próxima_Sentencia.ipynb

# 014.- Curso de IA Generativa. Ajustes finos de BERT.

Video en el que vemos como llevar a cabo ajustes finos a partir de los modelos Base de BERT. Lo vemos con tres de las tareas más comunes. Análisis de sentencias, análisis de tokens y preguntas y respuestas. Lo vemos en este videotutorial con modelos subidos por desarrolladores al repositorio de Huggingface y a partir del próximo video empezaremos a crear los nuestros con datos propios.

ejemplos/014_Ajuste_Fino.ipynb  

# 015.- Curso de IA Generativa. Entrenar un modelo BERT con datos propios.

Video en el que empezamos el primer entrenamiento personalizado de un modelo BERT, alimentándolo con nuestros propios datos. Empezamos analizando todo el proceso, creando el entorno, que seguirá usando BERT, pero con TPU para la aceleración de Hardware y creando un disco virtual en google drive al que acceder para leer y escribir datos desde nuestro modelo. Vemos las librerías que tenemos que instalar y las principales clases y funciones que tenemos que importar para llevar a cabo el entrenamiento. Analizamos el Dataset SNIPS, cuyos datos usaremos y como convertirlo con Python en un formato adecuado para poder alimentar con estos datos al modelo. 

# 016.- Curso de IA Generativa. Entrenamiento completo y evaluación.

Terminamos el video de entrenamiento del modelo para la clasificación de secuencias por categorías. Vemos como convertir esos datos a un formato dataset utilizable por el modelo. Como dividirlos entre una parte para entrenamiento y otra para pruebas; Creamos los argumentos de entrenamiento con training_arguments y entrenamos el modelo con dos versiones. Uno que entrena los nuevos datos a través de todos los codificadores del modelo y otro que solo utiliza la nueva capa de ajuste fino añadida.

# 017.- Curso de IA Generativa. Entrenamiento de Reconocimiento de Entidades.

Segunda tarea de ajuste fino a partir de modelos pre-entrenados BERT, en el que entrenamos al model para que pueda reconocer entidades a partir de una serie de etiquetas de tokens. Usamos el mismo dataset del video anterior y llevamos a cabo las modificaciones necesarias para un tipo de clasificación diferente pero manteniendo parte de la estructura.

# 018.- Curso de IA Generativa. Entrenamiento de Preguntas y Respuestas.

Tercera tarea con BERT. De preguntas y respuestas. Diferencias entre tareas extractivas y abstractivas. Ejemplo práctico tarea preguntas y respuestas. Limitaciones de las tareas extractivas. 

BERT Extrae.  Los codificadores son buenos ordenando, clasificando, o extrayendo información de la que recibe como input
GTP Abstrae. Los decodifcadores transforman, crean a traves de los input, outputs diferentes pero con sentido de la entrada

# 019.- Curso de IA Generativa. Semejanzas y diferencias entre GPT y BERT.

Iniciamos un nuevo bloque del curso en el que nos vamos a centrar en GPT, el modelo Auto-Regresivo que ha popularizado el Procesamiento del Lenguaje Natural pero usando tecnologías que llevan consolidándose desde hace años. Nos centramos en ver las semajanzas y diferencias que presentan BERT y GPT, los dos basados en Transfomer, pero el primero es un modelo Auto-Codificado, mientras que GPT es un modelo Auto-regresivo.

# 020.- Curso de IA Generativa. La Arquitectura Interna de GPT.

Video en el que empezamos a analizar en profundidad GPT. Vemos la evolución de la familia GPT desde sus orígenes en 2018; Las diferencias entre la versión Open Source de GPT-2 y la situación de GPT-3 y GPT-4. Vemos como instalar, importar y utilizar GPT-2 desde Hugging Face y distintos ejemplos de uso. Analizamos la arquitectura interna de su modelo a partir de la información generada por el propio modelo sobre su contenido.

# 021.- Curso de IA Generativa. Atención Multi-cabeza enmascarada en GPT

Video en el que vemos el modo en el que GPT aplica la atención multi-cabeza al tratarse de un modelo auto-regresivo y usar solo los decodificadores de la arquitectura transformer. Vemos ejemplos de su uso y los visualizamos gráficamente con el uso de Bertviz, comparando su uso de las cabezas de atención con el modelo auto-codificado de BERT.


## Notas

```bash
nvidia-smi -q -g 0 -d UTILIZATION -l

nvidia-smi -l 1

pip install gpustat

gpustat -cp
watch -n 0.5 -c gpustat -cp --color

``````
