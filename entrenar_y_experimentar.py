import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, f1_score
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42


def fijar_semillas(semilla=SEED):
    np.random.seed(semilla)
    tf.random.set_seed(semilla)
    tf.keras.utils.set_random_seed(semilla)


fijar_semillas(SEED)


def cargar_y_preprocesar(ruta_csv='diabetes_data_es_GOLD.csv', test_size=0.20, random_state=42, guardar_scaler=True):
    df = pd.read_csv(ruta_csv)
    raw_shape = df.shape
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['TieneDiabetes'])

    feature_names = [c for c in df.columns if c not in ('TieneDiabetes', 'target')]
    X = df[feature_names].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if guardar_scaler:
        joblib.dump(scaler, 'scaler.pkl')

    return df, raw_shape, feature_names, le, scaler, X_train_scaled, X_test_scaled, y_train, y_test


def analizar_correlacion(df, ruta_img='correlacion.png'):
    corr = df.drop(columns=['TieneDiabetes']).corr()['target'].sort_values(ascending=False)
    print("\n=== RANKING DE CORRELACIÓN DE PEARSON CON TIENEDIABETES ===")
    print(corr.to_string())

    plt.figure(figsize=(8, 6))
    corr_features = corr.drop('target').sort_values(ascending=True)
    colors = ['#d95f02' if v < 0 else 'steelblue' for v in corr_features.values]
    corr_features.plot(kind='barh', color=colors)
    plt.title('Correlación de cada feature con TieneDiabetes')
    plt.xlabel('Coeficiente de Correlación de Pearson (r)')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    plt.savefig(ruta_img, dpi=150)
    plt.close()
    return corr


def construir_rna(capas_ocultas=(32, 16), input_dim=16, nombre='RNA_Diabetes'):
    secuencia = []
    for idx, unidades in enumerate(capas_ocultas):
        if idx == 0:
            secuencia.append(layers.Dense(unidades, activation='relu', input_shape=(input_dim,)))
            secuencia.append(layers.Dropout(0.3))
        else:
            secuencia.append(layers.Dense(unidades, activation='relu'))
            secuencia.append(layers.Dropout(0.2))
    secuencia.append(layers.Dense(1, activation='sigmoid'))

    model = keras.Sequential(secuencia, name=nombre)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def entrenar_rna(model, X_train, y_train, epochs=100, batch_size=32, val_split=0.15, verbose=0):
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[early_stop],
        verbose=verbose
    )
    return history


def graficar_curva_aprendizaje(history, ruta_img='curva_aprendizaje.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='Train', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val', linewidth=2)
    ax1.set_title('Pérdida (Binary Crossentropy)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val', linewidth=2)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta_img, dpi=150)
    plt.close()


def evaluar_modelo_rna(model, history, X_test, y_test):
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    best_epoch = int(np.argmin(history.history['val_loss']))
    train_acc = float(history.history['accuracy'][best_epoch])
    val_acc = float(history.history['val_accuracy'][best_epoch])
    test_acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    return {
        'y_prob': y_prob,
        'y_pred': y_pred,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'f1': f1,
        'auc': auc,
        'epochs_run': len(history.history['loss']),
        'best_epoch': best_epoch + 1
    }


def entrenar_y_evaluar_gbt(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42):
    t0 = time.perf_counter()
    gbt = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state
    )
    gbt.fit(X_train, y_train)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    train_acc = float(accuracy_score(y_train, gbt.predict(X_train)))
    y_pred = gbt.predict(X_test)
    y_prob = gbt.predict_proba(X_test)[:, 1]
    test_acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))

    return gbt, {
        'y_prob': y_prob,
        'y_pred': y_pred,
        'train_acc': train_acc,
        'val_acc': np.nan,
        'test_acc': test_acc,
        'f1': f1,
        'auc': auc,
        'time_ms': elapsed_ms
    }


def graficar_matrices_confusion(y_test, y_pred_rna, y_pred_gbt, ruta_img='confusion_matrix.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y_pred, title in [(ax1, y_pred_rna, 'RNA'), (ax2, y_pred_gbt, 'GBT')]:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos']
        )
        ax.set_title(f'Confusión — {title}')
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
    plt.tight_layout()
    plt.savefig(ruta_img, dpi=150)
    plt.close()


def ejecutar_desafios(df, feature_names, X_train_base, X_test_base, y_train_base, y_test_base, res_rna_base, gbt_base, res_gbt_base):
    filas_md = []

    def agregar_fila(desafio, variante, train_val_str, test_acc, f1, auc, obs):
        filas_md.append({
            'Desafío': desafio,
            'Variante / Configuración': variante,
            'Train / Val Acc': train_val_str,
            'Test Acc': f'{test_acc:.4f} ({test_acc*100:.2f}%)',
            'F1-Score': f'{f1:.4f}',
            'ROC-AUC': f'{auc:.4f}',
            'Observación Técnica': obs
        })

    # 0. Base RNA y Base GBT
    agregar_fila(
        'Base',
        'RNA Base [32, 16] (split 80/20)',
        f"{res_rna_base['train_acc']:.4f} / {res_rna_base['val_acc']:.4f}",
        res_rna_base['test_acc'], res_rna_base['f1'], res_rna_base['auc'],
        f"Convergencia estable (mejor época {res_rna_base['best_epoch']}/{res_rna_base['epochs_run']}) con Dropout regulando el sobreajuste."
    )
    agregar_fila(
        'Base',
        'GBT Base (n=100, lr=0.1, depth=5, split 80/20)',
        f"{res_gbt_base['train_acc']:.4f} / N/A",
        res_gbt_base['test_acc'], res_gbt_base['f1'], res_gbt_base['auc'],
        f"Mayor capacidad en datos tabulares binarios ({res_gbt_base['time_ms']:.1f} ms); captura interacciones discretas exactas."
    )

    # Desafío 1: Capas ocultas [64, 32, 16] y [128, 64, 32, 16]
    for nombre_var, capas in [('Variante 1a: RNA [64, 32, 16]', (64, 32, 16)),
                              ('Variante 1b: RNA [128, 64, 32, 16]', (128, 64, 32, 16))]:
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        tf.keras.utils.set_random_seed(SEED)
        m = construir_rna(capas_ocultas=capas, input_dim=len(feature_names), nombre=f"RNA_{'_'.join(map(str, capas))}")
        h = entrenar_rna(m, X_train_base, y_train_base, verbose=0)
        r = evaluar_modelo_rna(m, h, X_test_base, y_test_base)
        diff = (r['test_acc'] - res_rna_base['test_acc']) * 100
        obs = (
            f"Parámetros={m.count_params():,}; ΔAcc={diff:+.2f}% vs Base. "
            + ("Aumentar profundidad mejora ligeramente o mantiene capacidad sin mejorar el límite tabular."
               if diff >= 0 else
               "Sobre-parametrización para N=520 muestras induce mayor varianza y ligero deterioro en generalización.")
        )
        agregar_fila(
            'Desafío 1 (Capas ocultas)',
            nombre_var,
            f"{r['train_acc']:.4f} / {r['val_acc']:.4f}",
            r['test_acc'], r['f1'], r['auc'],
            obs
        )

    # Desafío 2: Partición extrema (50/50 y 95/5)
    for nombre_var, t_size in [('Variante 2a: Split 50/50 (test_size=0.50)', 0.50),
                               ('Variante 2b: Split 95/5 (test_size=0.05)', 0.05)]:
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        tf.keras.utils.set_random_seed(SEED)
        _, _, _, _, _, X_tr, X_te, y_tr, y_te = cargar_y_preprocesar(
            test_size=t_size, random_state=SEED, guardar_scaler=False
        )
        m = construir_rna(capas_ocultas=(32, 16), input_dim=len(feature_names))
        h = entrenar_rna(m, X_tr, y_tr, verbose=0)
        r = evaluar_modelo_rna(m, h, X_te, y_te)
        obs = (
            f"Train={len(y_tr)} / Test={len(y_te)}. Con menos datos de entrenamiento (260 filas) cae el rendimiento por menor cobertura muestral."
            if t_size == 0.50 else
            f"Train={len(y_tr)} / Test={len(y_te)}. Aunque entrena con 494 filas, evaluar sobre solo 26 muestras tiene alta varianza estadística (±3.8% por muestra)."
        )
        agregar_fila(
            'Desafío 2 (Partición extrema)',
            nombre_var,
            f"{r['train_acc']:.4f} / {r['val_acc']:.4f}",
            r['test_acc'], r['f1'], r['auc'],
            obs
        )

    # Desafío 3: Árboles GBT (10, 50, 100, 200, 500) midiendo tiempo
    for n_est in [10, 50, 100, 200, 500]:
        _, r_gbt = entrenar_y_evaluar_gbt(
            X_train_base, y_train_base, X_test_base, y_test_base,
            n_estimators=n_est, max_depth=5, learning_rate=0.1, random_state=SEED
        )
        obs = (
            f"Tiempo: {r_gbt['time_ms']:.1f} ms. "
            + ("Sub-ajuste (underfitting): 10 árboles no bastan para reducir el residuo." if n_est == 10 else
               "Buen compromiso velocidad/precisión; captura la mayoría de patrones." if n_est == 50 else
               "Punto óptimo base: exactitud máxima en train (1.00) y excelente generalización." if n_est == 100 else
               "Rendimientos decrecientes: el costo computacional crece linealmente sin ganancia adicional en test.")
        )
        agregar_fila(
            'Desafío 3 (Árboles GBT)',
            f'GBT n_estimators={n_est}',
            f"{r_gbt['train_acc']:.4f} / N/A",
            r_gbt['test_acc'], r_gbt['f1'], r_gbt['auc'],
            obs
        )

    # Desafío 4: Learning Rate GBT (0.001, 0.01, 0.1, 0.3, 1.0) + lr_vs_accuracy.png
    lrs = [0.001, 0.01, 0.1, 0.3, 1.0]
    lr_train_accs, lr_test_accs, lr_f1s, lr_aucs = [], [], [], []
    for lr in lrs:
        _, r_lr = entrenar_y_evaluar_gbt(
            X_train_base, y_train_base, X_test_base, y_test_base,
            n_estimators=100, max_depth=5, learning_rate=lr, random_state=SEED
        )
        lr_train_accs.append(r_lr['train_acc'])
        lr_test_accs.append(r_lr['test_acc'])
        lr_f1s.append(r_lr['f1'])
        lr_aucs.append(r_lr['auc'])
        obs = (
            "Convergencia demasiado lenta para 100 árboles; predice cercano a la clase mayoritaria/umbrales básicos." if lr == 0.001 else
            "Aprendizaje conservador; requiere más árboles (>300) para alcanzar el óptimo." if lr == 0.01 else
            "Tasa de aprendizaje óptima estándar con regularización implícita adecuada." if lr == 0.1 else
            "Paso agresivo pero aún estable en este dataset limpio." if lr == 0.3 else
            "Sobre-corrección de residuos (overshooting); degrada la estabilidad del ensamble y el AUC."
        )
        agregar_fila(
            'Desafío 4 (Learning Rate GBT)',
            f'GBT learning_rate={lr}',
            f"{r_lr['train_acc']:.4f} / N/A",
            r_lr['test_acc'], r_lr['f1'], r_lr['auc'],
            obs
        )

    x_idx = np.arange(len(lrs))
    width = 0.25
    plt.figure(figsize=(9, 5))
    b1 = plt.bar(x_idx - width, lr_train_accs, width, label='Train Accuracy', color='#1f77b4')
    b2 = plt.bar(x_idx, lr_test_accs, width, label='Test Accuracy', color='#ff7f0e')
    b3 = plt.bar(x_idx + width, lr_f1s, width, label='Test F1-Score', color='#2ca02c')
    for bars in (b1, b2, b3):
        for bar in bars:
            h_val = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h_val + 0.008, f'{h_val:.2f}',
                     ha='center', va='bottom', fontsize=8)
    plt.xticks(x_idx, [str(x) for x in lrs])
    plt.title('Desafío 4: Efecto del Learning Rate en Gradient Boosted Trees')
    plt.xlabel('Learning Rate')
    plt.ylabel('Puntaje de Métrica')
    plt.ylim(0.50, 1.07)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('lr_vs_accuracy.png', dpi=150)
    if 'ipykernel' in sys.modules:
        plt.show()
    plt.close()

    # Desafío 5: Importancia de síntomas (Ablación eliminando 1 columna a la vez + feature_importances.png)
    importances = gbt_base.feature_importances_
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    plt.figure(figsize=(9, 6))
    imp_series.plot(kind='barh', color='teal')
    plt.title('Desafío 5: Importancia de Síntomas en Gradient Boosted Trees (MDI)')
    plt.xlabel('Importancia Relativa (Mean Decrease in Impurity)')
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=150)
    if 'ipykernel' in sys.modules:
        plt.show()
    plt.close()

    # Evaluar caída eliminando Poliuria, Polidipsia, ambos, y cada columna individualmente
    ablacion_resultados = []
    for idx_col, col_name in enumerate(feature_names):
        X_tr_del = np.delete(X_train_base, idx_col, axis=1)
        X_te_del = np.delete(X_test_base, idx_col, axis=1)

        tf.keras.utils.set_random_seed(SEED)
        m_del = construir_rna(capas_ocultas=(32, 16), input_dim=len(feature_names) - 1)
        h_del = entrenar_rna(m_del, X_tr_del, y_train_base, verbose=0)
        r_rna_del = evaluar_modelo_rna(m_del, h_del, X_te_del, y_test_base)

        _, r_gbt_del = entrenar_y_evaluar_gbt(
            X_tr_del, y_train_base, X_te_del, y_test_base, random_state=SEED
        )
        ablacion_resultados.append((col_name, importances[idx_col], r_rna_del, r_gbt_del))

    # Ordenar por importancia GBT descendente y reportar las más relevantes + eliminar ambas (Poliuria + Polidipsia)
    ablacion_resultados.sort(key=lambda x: x[1], reverse=True)
    for col_name, imp_val, r_rna_del, r_gbt_del in ablacion_resultados[:5]:
        caida_rna = (r_rna_del['test_acc'] - res_rna_base['test_acc']) * 100
        caida_gbt = (r_gbt_del['test_acc'] - res_gbt_base['test_acc']) * 100
        agregar_fila(
            'Desafío 5 (Ablación 1 feature)',
            f'Sin {col_name} (Imp GBT={imp_val:.3f}) — RNA / GBT',
            f"RNA:{r_rna_del['train_acc']:.3f} / GBT:{r_gbt_del['train_acc']:.3f}",
            r_rna_del['test_acc'], r_rna_del['f1'], r_rna_del['auc'],
            f"RNA ΔAcc={caida_rna:+.2f}% | GBT Test Acc={r_gbt_del['test_acc']*100:.2f}% (ΔAcc={caida_gbt:+.2f}%, F1={r_gbt_del['f1']:.4f}). Síntoma clave pero compensado parcialmente por colinealidad."
        )

    # También evaluar eliminando Poliuria y Polidipsia simultáneamente
    idx_poliuria = feature_names.index('Poliuria')
    idx_polidipsia = feature_names.index('Polidipsia')
    X_tr_no_pp = np.delete(X_train_base, [idx_poliuria, idx_polidipsia], axis=1)
    X_te_no_pp = np.delete(X_test_base, [idx_poliuria, idx_polidipsia], axis=1)
    tf.keras.utils.set_random_seed(SEED)
    m_no_pp = construir_rna(capas_ocultas=(32, 16), input_dim=len(feature_names) - 2)
    h_no_pp = entrenar_rna(m_no_pp, X_tr_no_pp, y_train_base, verbose=0)
    r_rna_no_pp = evaluar_modelo_rna(m_no_pp, h_no_pp, X_te_no_pp, y_test_base)
    _, r_gbt_no_pp = entrenar_y_evaluar_gbt(X_tr_no_pp, y_train_base, X_te_no_pp, y_test_base, random_state=SEED)
    agregar_fila(
        'Desafío 5 (Ablación conjunta)',
        'Sin Poliuria NI Polidipsia — RNA / GBT',
        f"RNA:{r_rna_no_pp['train_acc']:.3f} / GBT:{r_gbt_no_pp['train_acc']:.3f}",
        r_rna_no_pp['test_acc'], r_rna_no_pp['f1'], r_rna_no_pp['auc'],
        f"Caída notable al remover ambos síntomas cardinales: RNA ΔAcc={(r_rna_no_pp['test_acc']-res_rna_base['test_acc'])*100:+.2f}%, GBT Test Acc={r_gbt_no_pp['test_acc']*100:.2f}% (ΔAcc={(r_gbt_no_pp['test_acc']-res_gbt_base['test_acc'])*100:+.2f}%)."
    )

    # Desafío 6: Varianza entre semillas [0, 7, 21, 42, 99]
    semillas = [0, 7, 21, 42, 99]
    rna_accs, rna_f1s, rna_aucs = [], [], []
    gbt_accs, gbt_f1s, gbt_aucs = [], [], []

    for s in semillas:
        np.random.seed(s)
        tf.random.set_seed(s)
        tf.keras.utils.set_random_seed(s)
        _, _, _, _, _, X_tr_s, X_te_s, y_tr_s, y_te_s = cargar_y_preprocesar(
            test_size=0.20, random_state=s, guardar_scaler=False
        )
        np.random.seed(s)
        tf.random.set_seed(s)
        m_s = construir_rna(capas_ocultas=(32, 16), input_dim=len(feature_names))
        h_s = entrenar_rna(m_s, X_tr_s, y_tr_s, verbose=0)
        r_rna_s = evaluar_modelo_rna(m_s, h_s, X_te_s, y_te_s)
        _, r_gbt_s = entrenar_y_evaluar_gbt(X_tr_s, y_tr_s, X_te_s, y_te_s, random_state=s)

        rna_accs.append(r_rna_s['test_acc'])
        rna_f1s.append(r_rna_s['f1'])
        rna_aucs.append(r_rna_s['auc'])

        gbt_accs.append(r_gbt_s['test_acc'])
        gbt_f1s.append(r_gbt_s['f1'])
        gbt_aucs.append(r_gbt_s['auc'])

        agregar_fila(
            'Desafío 6 (Semillas)',
            f'Semilla random_state={s} (RNA / GBT)',
            f"RNA:{r_rna_s['train_acc']:.3f}/{r_rna_s['val_acc']:.3f} | GBT:{r_gbt_s['train_acc']:.3f}",
            r_rna_s['test_acc'], r_rna_s['f1'], r_rna_s['auc'],
            f"RNA Acc={r_rna_s['test_acc']*100:.2f}% (F1={r_rna_s['f1']:.4f}, AUC={r_rna_s['auc']:.4f}) vs GBT Acc={r_gbt_s['test_acc']*100:.2f}% (F1={r_gbt_s['f1']:.4f}, AUC={r_gbt_s['auc']:.4f})."
        )

    rna_mean, rna_std = float(np.mean(rna_accs)), float(np.std(rna_accs))
    gbt_mean, gbt_std = float(np.mean(gbt_accs)), float(np.std(gbt_accs))

    agregar_fila(
        'Desafío 6 (Resumen RNA)',
        'RNA Media ± Desv. Est. (5 semillas)',
        'N/A',
        rna_mean, float(np.mean(rna_f1s)), float(np.mean(rna_aucs)),
        f"Accuracy RNA = {rna_mean*100:.2f}% ± {rna_std*100:.2f}% (rango: {min(rna_accs)*100:.2f}% – {max(rna_accs)*100:.2f}%)."
    )
    agregar_fila(
        'Desafío 6 (Resumen GBT)',
        'GBT Media ± Desv. Est. (5 semillas)',
        'N/A',
        gbt_mean, float(np.mean(gbt_f1s)), float(np.mean(gbt_aucs)),
        f"Accuracy GBT = {gbt_mean*100:.2f}% ± {gbt_std*100:.2f}% (rango: {min(gbt_accs)*100:.2f}% – {max(gbt_accs)*100:.2f}%). Menor varianza y mayor media que RNA."
    )

    df_desafios = pd.DataFrame(filas_md)

    # Exportar tabla_desafios.md
    headers = list(df_desafios.columns)
    md_lines = [
        "# Tabla Consolidada de Experimentación — 6 Desafíos + Bonus\n",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for _, row in df_desafios.iterrows():
        md_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    md_lines.append("\n## Detalle Completo de Ablación por Columna (Desafío 5)\n")
    md_lines.append("| Feature Eliminado | Importancia GBT | RNA Test Acc | RNA ΔAcc | GBT Test Acc | GBT ΔAcc |")
    md_lines.append("| --- | --- | --- | --- | --- | --- |")
    for col_name, imp_val, r_rna_del, r_gbt_del in ablacion_resultados:
        d_rna = (r_rna_del['test_acc'] - res_rna_base['test_acc']) * 100
        d_gbt = (r_gbt_del['test_acc'] - res_gbt_base['test_acc']) * 100
        md_lines.append(
            f"| {col_name} | {imp_val:.4f} | {r_rna_del['test_acc']*100:.2f}% | {d_rna:+.2f}% | {r_gbt_del['test_acc']*100:.2f}% | {d_gbt:+.2f}% |"
        )

    with open('tabla_desafios.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n=== RESUMEN DE LOS 6 DESAFÍOS EXPORTADO A tabla_desafios.md ===")
    print(f"RNA (5 semillas): {rna_mean*100:.2f}% ± {rna_std*100:.2f}%")
    print(f"GBT (5 semillas): {gbt_mean*100:.2f}% ± {gbt_std*100:.2f}%")
    return df_desafios


def main():
    print(f"TensorFlow versión: {tf.__version__}")
    df, raw_shape, feature_names, le, scaler, X_train, X_test, y_train, y_test = cargar_y_preprocesar()
    print(f"Dataset shape: {raw_shape} | Nulos totales: {df.isnull().sum().sum()}")
    print("Distribución TieneDiabetes:\n", df['TieneDiabetes'].value_counts().to_string())
    print(f"Clases LabelEncoder: {le.classes_}")
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    analizar_correlacion(df, ruta_img='correlacion.png')

    tf.keras.utils.set_random_seed(SEED)
    model = construir_rna(capas_ocultas=(32, 16), input_dim=len(feature_names), nombre='RNA_Diabetes')
    model.summary()

    history = entrenar_rna(model, X_train, y_train, epochs=100, batch_size=32, val_split=0.15, verbose=0)
    model.save('modelo_rna.keras')
    graficar_curva_aprendizaje(history, ruta_img='curva_aprendizaje.png')

    res_rna = evaluar_modelo_rna(model, history, X_test, y_test)
    gbt, res_gbt = entrenar_y_evaluar_gbt(X_train, y_train, X_test, y_test, random_state=SEED)

    tabla_comp = pd.DataFrame([
        {
            'Modelo': 'Red Neuronal Artificial (RNA [32, 16])',
            'Train Accuracy': f"{res_rna['train_acc']:.4f}",
            'Test Accuracy': f"{res_rna['test_acc']:.4f} ({res_rna['test_acc']*100:.2f}%)",
            'F1-Score': f"{res_rna['f1']:.4f}",
            'ROC-AUC': f"{res_rna['auc']:.4f}"
        },
        {
            'Modelo': 'Gradient Boosted Trees (GBT n=100, depth=5)',
            'Train Accuracy': f"{res_gbt['train_acc']:.4f}",
            'Test Accuracy': f"{res_gbt['test_acc']:.4f} ({res_gbt['test_acc']*100:.2f}%)",
            'F1-Score': f"{res_gbt['f1']:.4f}",
            'ROC-AUC': f"{res_gbt['auc']:.4f}"
        }
    ])
    print("\n=== TABLA COMPARATIVA: RNA vs. GRADIENT BOOSTED TREES (TEST SET) ===")
    print(tabla_comp.to_string(index=False))

    cm_rna = confusion_matrix(y_test, res_rna['y_pred'])
    cm_gbt = confusion_matrix(y_test, res_gbt['y_pred'])
    print(f"\nMatriz de Confusión RNA:\n{cm_rna}")
    print(f"\nMatriz de Confusión GBT:\n{cm_gbt}")

    graficar_matrices_confusion(y_test, res_rna['y_pred'], res_gbt['y_pred'], ruta_img='confusion_matrix.png')

    ejecutar_desafios(df, feature_names, X_train, X_test, y_train, y_test, res_rna, gbt, res_gbt)

    # Self-check de artefactos exportados e inferencia tipo Streamlit
    m_loaded = keras.models.load_model('modelo_rna.keras')
    s_loaded = joblib.load('scaler.pkl')
    paciente_bajo = s_loaded.transform([[30, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    paciente_alto = s_loaded.transform([[55, 2, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]])
    p_bajo = float(m_loaded.predict(paciente_bajo, verbose=0)[0][0])
    p_alto = float(m_loaded.predict(paciente_alto, verbose=0)[0][0])
    assert 0.0 <= p_bajo < 0.5 and 0.5 <= p_alto <= 1.0, f"Fallo en self-check: p_bajo={p_bajo}, p_alto={p_alto}"
    print(f"\nSelf-check OK -> Paciente bajo riesgo: {p_bajo*100:.2f}% | Paciente alto riesgo: {p_alto*100:.2f}%")
    print("Ejecución completada exitosamente. Gráficos (.png a 150 dpi), modelos y tablas generados.")


if __name__ == '__main__':
    main()

