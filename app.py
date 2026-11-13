import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

st.set_page_config(page_title='Predictor Diabetes RNA', page_icon='🩺', layout='wide')


@st.cache_resource
def cargar():
    return keras.models.load_model('modelo_rna.keras'), joblib.load('scaler.pkl')


modelo, scaler = cargar()

st.title('🩺 Predictor de Diabetes — Red Neuronal Artificial')
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    edad = st.number_input('Edad', 16, 90, 45)
    genero = st.selectbox('Género', ['Masculino (1)', 'Femenino (2)'])
    poliuria = st.checkbox('Poliuria')
    polidipsia = st.checkbox('Polidipsia')
    perdida = st.checkbox('Pérdida de peso')
    debilidad = st.checkbox('Debilidad')
with col2:
    polifagia = st.checkbox('Polifagia')
    candidiasis = st.checkbox('Candidiasis genital')
    vision = st.checkbox('Visión borrosa')
    picazon = st.checkbox('Picazón')
    irritab = st.checkbox('Irritabilidad')
    cicatriz = st.checkbox('Cicatrización lenta')
with col3:
    paresia = st.checkbox('Paresia parcial')
    rigidez = st.checkbox('Rigidez muscular')
    alopecia = st.checkbox('Alopecia')
    obesidad = st.checkbox('Obesidad')

st.divider()

if st.button('🔍 Predecir Riesgo', use_container_width=True):
    g = 1 if 'Masculino' in genero else 2
    x = scaler.transform([[
        edad, g, int(poliuria), int(polidipsia), int(perdida),
        int(debilidad), int(polifagia), int(candidiasis), int(vision), int(picazon),
        int(irritab), int(cicatriz), int(paresia), int(rigidez), int(alopecia), int(obesidad)
    ]])
    prob = float(modelo.predict(x, verbose=0)[0][0])
    if prob >= 0.5:
        st.error(f'⚠️ RIESGO ALTO — {prob * 100:.1f}%')
    else:
        st.success(f'✅ RIESGO BAJO — {prob * 100:.1f}%')
    st.progress(prob)
    st.caption('⚠️ Resultado orientativo. Consulte siempre a un médico.')

