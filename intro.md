El final de ajedrez Rey + Torre vs. Rey + Peón en a7 (KR-vs-KP) es un escenario clásico donde, con el turno de blancas, el desenlace puede ser victoria o no victoria según la disposición precisa de las piezas. En este proyecto abordamos el problema como clasificación binaria supervisada usando el conjunto kr-vs-kp (OpenML ID 3): 3196 instancias, 36 atributos categóricos que describen la posición y 0 valores faltantes; la variable objetivo indica si las blancas ganan (“won”) o no ganan (“nowin”), con una distribución cercana a 52%/48%. El interés práctico es construir un oráculo rápido que, a partir de la codificación de la posición, anticipe el resultado y sirva como caso de estudio para tabulares totalmente discretos.

Objetivo general. Diseñar, entrenar y evaluar modelos de aprendizaje automático que predigan el resultado del final KR-vs-KP, comparando enfoques paramétricos y no paramétricos y analizando el efecto de la reducción de dimensionalidad sobre desempeño y costo computacional.

2.1 Contexto y utilidad

    El problema estudia el final de ajedrez Rey + Torre (blancas) vs. Rey + Peón en a7 (negras), con turno de las blancas. Se busca predecir automáticamente si, dada la configuración exacta del tablero, las blancas ganan (“won”) o no ganan (“nowin”). Una solución basada en ML permite construir un oráculo rápido para apoyar análisis/entrenamiento de finales y, a la vez, sirve como caso de estudio en tabulares 100% categóricos. Esta sección cumple con el requerimiento de presentar el contexto y la utilidad de una solución ML según la guía.

2.2 Conjunto de datos (composición y EDA inicial)

    Fuente: OpenML/UCI, dataset kr-vs-kp (ID=3).
    Tamaño: 3196 instancias.
    Variables: 36 predictoras categóricas (nominales) que codifican relaciones/posiciones; 1 variable objetivo (class ∈ {won, nowin}).

    Faltantes: 0.
    Distribución de clases: aproximada 52% won / 48% nowin (balance leve).
    Significado: cada atributo ocupa una posición fija en la lista de 36; no existe orden natural entre categorías (evita codificación ordinal).

    Análisis exploratorio previsto
    Conteo de niveles por variable, detección de cardinalidades altas; 2) distribución de la clase; 3) verificación de faltantes (no aplica) y definición de estrategia de codificación; 4) revisión rápida de co-ocurrencias entre categorías para motivar la selección de modelos y regularización. Se deja explícito el detalle exigido por la guía: número de muestras, variables, significado, faltantes y codificación.

    Codificación
    Dado que todas las variables son nominales, se usará One-Hot Encoding con handle_unknown="ignore" para evitar imponer orden artificial y para lidiar con categorías no vistas en validación/test. La selección y documentación de la codificación responde al lineamiento de la guía.

2.3 Paradigma de aprendizaje y justificación

    El problema se modela como clasificación binaria supervisada: se dispone de ejemplos etiquetados (won/nowin) y se desea aprender una función f:X→{0,1} que asigne la etiqueta correcta a nuevas posiciones. Esta configuración permite comparar enfoques paramétricos y no paramétricos en tabulares categóricos, y evaluar su comportamiento en términos de precisión, equilibrio entre clases y costo computacional.

    Validación y métricas (enlace con Sección 4):

    Validación: Stratified k-Fold (k=5 o 10) para estimar desempeño y varianza con preservación de la proporción de clases.

    Métricas primarias: Accuracy y F1-macro (ponderación equitativa de clases).

    Métricas de apoyo: ROC-AUC, matriz de confusión e intervalos de confianza sobre los puntajes de CV.

    Prevención de fuga de información: toda la codificación (y cualquier selección/reducción de características) se ajusta dentro de cada fold mediante pipelines; el conjunto de test permanece aislado hasta la evaluación final.