# Tabla Consolidada de Experimentación — 6 Desafíos + Bonus

| Desafío | Variante / Configuración | Train / Val Acc | Test Acc | F1-Score | ROC-AUC | Observación Técnica |
| --- | --- | --- | --- | --- | --- | --- |
| Base | RNA Base [32, 16] (split 80/20) | 0.9858 / 0.9683 | 0.9615 (96.15%) | 0.9683 | 0.9988 | Convergencia estable (mejor época 98/100) con Dropout regulando el sobreajuste. |
| Base | GBT Base (n=100, lr=0.1, depth=5, split 80/20) | 1.0000 / N/A | 0.9904 (99.04%) | 0.9921 | 1.0000 | Mayor capacidad en datos tabulares binarios (281.4 ms); captura interacciones discretas exactas. |
| Desafío 1 (Capas ocultas) | Variante 1a: RNA [64, 32, 16] | 0.9915 / 0.9683 | 0.9712 (97.12%) | 0.9764 | 0.9988 | Parámetros=3,713; ΔAcc=+0.96% vs Base. Aumentar profundidad mejora ligeramente o mantiene capacidad sin mejorar el límite tabular. |
| Desafío 1 (Capas ocultas) | Variante 1b: RNA [128, 64, 32, 16] | 0.9858 / 0.9683 | 0.9808 (98.08%) | 0.9844 | 0.9992 | Parámetros=13,057; ΔAcc=+1.92% vs Base. Aumentar profundidad mejora ligeramente o mantiene capacidad sin mejorar el límite tabular. |
| Desafío 2 (Partición extrema) | Variante 2a: Split 50/50 (test_size=0.50) | 0.9321 / 0.9231 | 0.9154 (91.54%) | 0.9295 | 0.9729 | Train=260 / Test=260. Con menos datos de entrenamiento (260 filas) cae el rendimiento por menor cobertura muestral. |
| Desafío 2 (Partición extrema) | Variante 2b: Split 95/5 (test_size=0.05) | 0.9714 / 0.9733 | 1.0000 (100.00%) | 1.0000 | 1.0000 | Train=494 / Test=26. Aunque entrena con 494 filas, evaluar sobre solo 26 muestras tiene alta varianza estadística (±3.8% por muestra). |
| Desafío 3 (Árboles GBT) | GBT n_estimators=10 | 0.9808 / N/A | 0.9808 (98.08%) | 0.9841 | 0.9891 | Tiempo: 22.7 ms. Sub-ajuste (underfitting): 10 árboles no bastan para reducir el residuo. |
| Desafío 3 (Árboles GBT) | GBT n_estimators=50 | 1.0000 / N/A | 0.9904 (99.04%) | 0.9921 | 0.9992 | Tiempo: 116.6 ms. Buen compromiso velocidad/precisión; captura la mayoría de patrones. |
| Desafío 3 (Árboles GBT) | GBT n_estimators=100 | 1.0000 / N/A | 0.9904 (99.04%) | 0.9921 | 1.0000 | Tiempo: 231.5 ms. Punto óptimo base: exactitud máxima en train (1.00) y excelente generalización. |
| Desafío 3 (Árboles GBT) | GBT n_estimators=200 | 1.0000 / N/A | 0.9808 (98.08%) | 0.9841 | 1.0000 | Tiempo: 472.7 ms. Rendimientos decrecientes: el costo computacional crece linealmente sin ganancia adicional en test. |
| Desafío 3 (Árboles GBT) | GBT n_estimators=500 | 1.0000 / N/A | 0.9808 (98.08%) | 0.9841 | 1.0000 | Tiempo: 870.0 ms. Rendimientos decrecientes: el costo computacional crece linealmente sin ganancia adicional en test. |
| Desafío 4 (Learning Rate GBT) | GBT learning_rate=0.001 | 0.6154 / N/A | 0.6154 (61.54%) | 0.7619 | 0.9910 | Convergencia demasiado lenta para 100 árboles; predice cercano a la clase mayoritaria/umbrales básicos. |
| Desafío 4 (Learning Rate GBT) | GBT learning_rate=0.01 | 0.9808 / N/A | 0.9808 (98.08%) | 0.9841 | 0.9891 | Aprendizaje conservador; requiere más árboles (>300) para alcanzar el óptimo. |
| Desafío 4 (Learning Rate GBT) | GBT learning_rate=0.1 | 1.0000 / N/A | 0.9904 (99.04%) | 0.9921 | 1.0000 | Tasa de aprendizaje óptima estándar con regularización implícita adecuada. |
| Desafío 4 (Learning Rate GBT) | GBT learning_rate=0.3 | 1.0000 / N/A | 0.9808 (98.08%) | 0.9841 | 1.0000 | Paso agresivo pero aún estable en este dataset limpio. |
| Desafío 4 (Learning Rate GBT) | GBT learning_rate=1.0 | 1.0000 / N/A | 0.9808 (98.08%) | 0.9841 | 1.0000 | Sobre-corrección de residuos (overshooting); degrada la estabilidad del ensamble y el AUC. |
| Desafío 5 (Ablación 1 feature) | Sin Poliuria (Imp GBT=0.393) — RNA / GBT | RNA:0.969 / GBT:1.000 | 0.9712 (97.12%) | 0.9764 | 0.9918 | RNA ΔAcc=+0.96% | GBT Test Acc=97.12% (ΔAcc=-1.92%, F1=0.9760). Síntoma clave pero compensado parcialmente por colinealidad. |
| Desafío 5 (Ablación 1 feature) | Sin Polidipsia (Imp GBT=0.129) — RNA / GBT | RNA:0.972 / GBT:1.000 | 0.9712 (97.12%) | 0.9764 | 0.9922 | RNA ΔAcc=+0.96% | GBT Test Acc=98.08% (ΔAcc=-0.96%, F1=0.9841). Síntoma clave pero compensado parcialmente por colinealidad. |
| Desafío 5 (Ablación 1 feature) | Sin Genero (Imp GBT=0.119) — RNA / GBT | RNA:0.955 / GBT:0.998 | 0.9904 (99.04%) | 0.9922 | 0.9992 | RNA ΔAcc=+2.88% | GBT Test Acc=100.00% (ΔAcc=+0.96%, F1=1.0000). Síntoma clave pero compensado parcialmente por colinealidad. |
| Desafío 5 (Ablación 1 feature) | Sin Edad (Imp GBT=0.083) — RNA / GBT | RNA:0.969 / GBT:0.995 | 0.9615 (96.15%) | 0.9683 | 0.9984 | RNA ΔAcc=+0.00% | GBT Test Acc=99.04% (ΔAcc=+0.00%, F1=0.9922). Síntoma clave pero compensado parcialmente por colinealidad. |
| Desafío 5 (Ablación 1 feature) | Sin Alopecia (Imp GBT=0.055) — RNA / GBT | RNA:0.977 / GBT:1.000 | 0.9712 (97.12%) | 0.9764 | 0.9977 | RNA ΔAcc=+0.96% | GBT Test Acc=98.08% (ΔAcc=-0.96%, F1=0.9841). Síntoma clave pero compensado parcialmente por colinealidad. |
| Desafío 5 (Ablación conjunta) | Sin Poliuria NI Polidipsia — RNA / GBT | RNA:0.943 / GBT:1.000 | 0.9327 (93.27%) | 0.9440 | 0.9816 | Caída notable al remover ambos síntomas cardinales: RNA ΔAcc=-2.88%, GBT Test Acc=95.19% (ΔAcc=-3.85%). |
| Desafío 6 (Semillas) | Semilla random_state=0 (RNA / GBT) | RNA:0.980/0.984 | GBT:1.000 | 0.9423 (94.23%) | 0.9531 | 0.9961 | RNA Acc=94.23% (F1=0.9531, AUC=0.9961) vs GBT Acc=98.08% (F1=0.9844, AUC=0.9949). |
| Desafío 6 (Semillas) | Semilla random_state=7 (RNA / GBT) | RNA:0.952/0.905 | GBT:1.000 | 0.9808 (98.08%) | 0.9846 | 0.9914 | RNA Acc=98.08% (F1=0.9846, AUC=0.9914) vs GBT Acc=99.04% (F1=0.9921, AUC=0.9988). |
| Desafío 6 (Semillas) | Semilla random_state=21 (RNA / GBT) | RNA:0.969/0.984 | GBT:1.000 | 0.9712 (97.12%) | 0.9767 | 0.9934 | RNA Acc=97.12% (F1=0.9767, AUC=0.9934) vs GBT Acc=98.08% (F1=0.9844, AUC=0.9973). |
| Desafío 6 (Semillas) | Semilla random_state=42 (RNA / GBT) | RNA:0.986/0.968 | GBT:1.000 | 0.9615 (96.15%) | 0.9683 | 0.9988 | RNA Acc=96.15% (F1=0.9683, AUC=0.9988) vs GBT Acc=99.04% (F1=0.9921, AUC=1.0000). |
| Desafío 6 (Semillas) | Semilla random_state=99 (RNA / GBT) | RNA:0.963/0.984 | GBT:1.000 | 0.9615 (96.15%) | 0.9683 | 0.9965 | RNA Acc=96.15% (F1=0.9683, AUC=0.9965) vs GBT Acc=96.15% (F1=0.9683, AUC=0.9793). |
| Desafío 6 (Resumen RNA) | RNA Media ± Desv. Est. (5 semillas) | N/A | 0.9635 (96.35%) | 0.9702 | 0.9952 | Accuracy RNA = 96.35% ± 1.28% (rango: 94.23% – 98.08%). |
| Desafío 6 (Resumen GBT) | GBT Media ± Desv. Est. (5 semillas) | N/A | 0.9808 (98.08%) | 0.9843 | 0.9941 | Accuracy GBT = 98.08% ± 1.05% (rango: 96.15% – 99.04%). Menor varianza y mayor media que RNA. |

## Detalle Completo de Ablación por Columna (Desafío 5)

| Feature Eliminado | Importancia GBT | RNA Test Acc | RNA ΔAcc | GBT Test Acc | GBT ΔAcc |
| --- | --- | --- | --- | --- | --- |
| Poliuria | 0.3927 | 97.12% | +0.96% | 97.12% | -1.92% |
| Polidipsia | 0.1288 | 97.12% | +0.96% | 98.08% | -0.96% |
| Genero | 0.1193 | 99.04% | +2.88% | 100.00% | +0.96% |
| Edad | 0.0827 | 96.15% | +0.00% | 99.04% | +0.00% |
| Alopecia | 0.0546 | 97.12% | +0.96% | 98.08% | -0.96% |
| Rigidez_muscular | 0.0508 | 96.15% | +0.00% | 99.04% | +0.00% |
| Cicatrizacion_lenta | 0.0347 | 96.15% | +0.00% | 99.04% | +0.00% |
| Irritabilidad | 0.0301 | 98.08% | +1.92% | 98.08% | -0.96% |
| Candidiasis_genital | 0.0288 | 97.12% | +0.96% | 99.04% | +0.00% |
| Perdida_repentina_peso | 0.0267 | 98.08% | +1.92% | 99.04% | +0.00% |
| Obesidad | 0.0187 | 96.15% | +0.00% | 98.08% | -0.96% |
| Picazon | 0.0093 | 98.08% | +1.92% | 98.08% | -0.96% |
| Vision_borrosa | 0.0090 | 97.12% | +0.96% | 98.08% | -0.96% |
| Polifagia | 0.0053 | 97.12% | +0.96% | 98.08% | -0.96% |
| Debilidad | 0.0051 | 97.12% | +0.96% | 99.04% | +0.00% |
| Paresia_parcial | 0.0032 | 96.15% | +0.00% | 99.04% | +0.00% |
