import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실", page_icon="🧪", layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
)

FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
KOREAN = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = ["hypertension", "heart_disease"]  # 0 또는 1 값만 갖는 속성

LOGREG_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "다수 클래스 모델(입력을 보지 않고 많은 쪽으로만 답하는 모델)"

COLOR_NO = "#1f77b4"  # 뇌졸중 아님
COLOR_YES = "#d62728"  # 뇌졸중
COLOR_LINE = "#2ca02c"  # 로지스틱 회귀 0.5 경계선
REGION_NO = "rgba(31, 119, 180, 0.15)"
REGION_YES = "rgba(214, 39, 40, 0.15)"

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 2.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() ->
