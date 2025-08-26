# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️", layout="centered")

@st.cache_resource
def load_pipeline(path="heart_pipeline.pkl"):
    return joblib.load(path)

pipe = load_pipeline()

st.title("❤️ Предсказание риска сердечного заболевания")
st.write("Введите параметры пациента и получите предсказание модели.")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=120, value=50)
        resting_bp = st.number_input("RestingBP", min_value=0, max_value=300, value=130)
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=800, value=200,
                                      help="0 будет трактоваться как пропуск и заменён средним")
        max_hr = st.number_input("MaxHR", min_value=0, max_value=250, value=150)
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-5.0, max_value=10.0, value=1.0, step=0.1)

    with col2:
        sex = st.selectbox("Sex", ["M","F"])
        chest_pain = st.selectbox("ChestPainType", ["ATA","NAP","ASY","TA"])
        resting_ecg = st.selectbox("RestingECG", ["Normal","ST","LVH"])
        fasting_bs = st.selectbox("FastingBS (глюкоза ≥ 120 mg/dl?)", [0,1])
        exercise_angina = st.selectbox("ExerciseAngina", ["Y","N"])
        st_slope = st.selectbox("ST_Slope", ["Up","Flat","Down"])

    submitted = st.form_submit_button("Предсказать")

if submitted:
    # Сформируем один объект как DataFrame в том же виде, что на обучении
    row = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": int(fasting_bs),
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }])

    try:
        y_pred = pipe.predict(row)[0]
        proba = None
        # если модель поддерживает predict_proba — покажем вероятность «болезни = 1»
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            proba = float(pipe.predict_proba(row)[0][1])

        st.subheader("Результат")
        if y_pred == 1:
            st.error("🧠 Модель предсказывает: **риск повышен (1)**")
        else:
            st.success("🧠 Модель предсказывает: **риск низкий (0)**")

        if proba is not None:
            st.write(f"Вероятность класса 1: **{proba:.3f}**")
            # индикатор
            st.progress(min(max(proba, 0.0), 1.0))

        # Отладочная информация (скрываем за экспандером)
        with st.expander("Показать отправленные данные"):
            st.write(row)

    except Exception as e:
        st.error(f"Ошибка инференса: {e}")
        st.stop()

st.caption("Модель: единый Pipeline (preprocessor + model), обученный оффлайн и загруженный из pickle.")
