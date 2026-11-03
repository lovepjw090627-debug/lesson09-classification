import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
st.set_page_config(
    page_title="성능 높이기 - 뇌졸중 예측 실습실", page_icon="🚀", layout="wide"
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

LABEL_BEFORE = "가중치 맞추기 전"
LABEL_AFTER = "가중치 맞춘 뒤"

COLOR_NO = "#1f77b4"  # 실제 뇌졸중 아님
COLOR_YES = "#d62728"  # 실제 뇌졸중
COLOR_BEFORE = "#9467bd"  # 맞추기 전 0.5 경계선(보라색)
COLOR_AFTER = "#2ca02c"  # 맞춘 뒤 0.5 경계선(초록색)

CANNOT = "계산할 수 없음"


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


# ------------------------------------------------------------------
# 맨 위: 입력 속성 고르기 ('분류 모델' 페이지와 같은 방식)
# ------------------------------------------------------------------
st.title("🚀 성능 높이기")

st.header("1. 입력으로 사용할 속성 고르기")

selected = st.multiselect(
    "입력 속성",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: KOREAN[c],
)

if len(selected) < 2:
    st.info(
        "입력 속성을 두 개 이상 골라 주세요. "
        "두 개보다 적으면 두 속성으로 그리는 그림을 만들 수 없어서 여기서 멈춥니다."
    )
    st.stop()

cols = [c for c in FEATURES if c in selected]  # 항상 같은 순서로 맞춘다

st.write(
    "뇌졸중(`stroke`=1)을 **양성**으로 둡니다. 뇌졸중인 사람은 아주 드물어서, "
    "모델이 '아님'으로만 답해도 정확도가 높게 나옵니다. "
    "그래서 양쪽의 가중치를 맞추기 전과 맞춘 뒤의 결과를 나란히 비교해 봅니다."
)

try:
    df = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# ------------------------------------------------------------------
# 데이터 나누기: 번호 순으로 열 명씩 묶어 각 묶음의 앞 세 명은 테스트용
# ------------------------------------------------------------------
data = df.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(data)) % 10) < 3

train_df = data[~is_test]
test_df = data[is_test]

X_train = train_df[cols].astype(float).copy()
X_test = test_df[cols].astype(float).copy()
y_train = train_df["stroke"].to_numpy()
y_test = test_df["stroke"].to_numpy()

# 체질량지수(bmi)는 고른 경우에만, 훈련용의 중앙값으로 빈 값을 채운다
if "bmi" in cols:
    bmi_median = X_train["bmi"].median()
    n_missing_train = int(X_train["bmi"].isna().sum())
    n_missing_test = int(X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    st.write(
        f"체질량지수의 빈 값(훈련용 {n_missing_train:,}명, 테스트용 {n_missing_test:,}명)은 "
        f"**훈련용의 중앙값 {bmi_median:.1f}**으로 채웠습니다."
    )

n_train = len(y_train)
n_pos = int(y_train.sum())
n_neg = n_train - n_pos
if n_pos == 0 or n_neg == 0:
    st.error("훈련용에 뇌졸중인 사람 또는 아닌 사람이 한 명도 없어서 비교할 수 없습니다.")
    st.stop()

# ------------------------------------------------------------------
# 모델 학습: 가중치 맞추기 전(None)과 맞춘 뒤("balanced")
# ------------------------------------------------------------------
def make_logreg(class_weight):
    return Pipeline(
        [
            ("scaler", StandardScaler()),  # 크기 맞추기: 훈련용으로만 계산
            (
                "logreg",
                LogisticRegression(class_weight=class_weight, random_state=42),
            ),
        ]
    )


def make_tree(class_weight):
    return DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=5,
        class_weight=class_weight,
        random_state=42,
    )


models = {
    LOGREG_NAME: (make_logreg(None), make_logreg("balanced")),
    TREE_NAME: (make_tree(None), make_tree("balanced")),
}
for before_model, after_model in models.values():
    before_model.fit(X_train, y_train)
    after_model.fit(X_train, y_train)

logreg_before, logreg_after = models[LOGREG_NAME]


# ------------------------------------------------------------------
# 네 칸 세기와 지표 계산 ('채점표' 페이지와 같은 방법)
# ------------------------------------------------------------------
def count_cells(y_true, y_pred):
    """TP, FP, FN, TN 순서로 사람 수를 센다. 뇌졸중(1)이 양성."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def divide(top, bottom):
    return top / bottom if bottom else None


def percent(value):
    return CANNOT if value is None else f"{value:.2%}"


def score_texts(tp, fp, fn, tn):
    """정확도, 재현율, 정밀도, F1을 글자로 돌려준다."""
    accuracy = divide(tp + tn, tp + fp + fn + tn)
    recall = divide(tp, tp + fn)  # 분모: 실제 뇌졸중인 사람 수
    precision = divide(tp, tp + fp)  # 분모: 뇌졸중이라 예측한 사람 수
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return percent(accuracy), percent(recall), percent(precision), percent(f1)


def before_after_table(before, after):
    """네 칸과 네 지표를 전·후로 나란히 놓은 표."""
    tp0, fp0, fn0, tn0 = before
    tp1, fp1, fn1, tn1 = after
    s0 = score_texts(*before)
    s1 = score_texts(*after)
    rows = [
        ("TP (찾아낸 환자)", f"{tp0:,}명", f"{tp1:,}명"),
        ("FN (놓친 환자)", f"{fn0:,}명", f"{fn1:,}명"),
        ("FP (헛짚은 사람)", f"{fp0:,}명", f"{fp1:,}명"),
        ("TN (아니라고 맞힌 사람)", f"{tn0:,}명", f"{tn1:,}명"),
        ("정확도", s0[0], s1[0]),
        ("재현율", s0[1], s1[1]),
        ("정밀도", s0[2], s1[2]),
        ("F1", s0[3], s1[3]),
    ]
    table = pd.DataFrame(rows, columns=["항목", LABEL_BEFORE, LABEL_AFTER])
    return table.set_index("항목")


# ------------------------------------------------------------------
# 2. 가중치를 맞추기 전과 후
# ------------------------------------------------------------------
st.header("2. 가중치를 맞추기 전과 맞춘 뒤")

weight_ratio = n_neg / n_pos
st.write(
    f"훈련용 {n_train:,}명 중 뇌졸중인 사람은 {n_pos:,}명({n_pos / n_train:.2%})뿐입니다. "
    f"그래서 맞춘 뒤에는 뇌졸중인 사람 한 명을 아님인 사람 한 명보다 약 {weight_ratio:.1f}배 "
    "무겁게 세어 학습합니다(`class_weight='balanced'`)."
)
st.write(
    f"테스트용은 **{len(test_df):,}명**이고, 그중 뇌졸중인 사람은 "
    f"**{int((y_test == 1).sum()):,}명**입니다."
)

left, right = st.columns(2)
for column, name in [(left, LOGREG_NAME), (right, TREE_NAME)]:
    before_model, after_model = models[name]
    before_cells = count_cells(y_test, before_model.predict(X_test))
    after_cells = count_cells(y_test, after_model.predict(X_test))
    tp0, fp0, fn0, tn0 = before_cells
    tp1, fp1, fn1, tn1 = after_cells
    with column:
        st.markdown(f"**{name}**")
        st.table(before_after_table(before_cells, after_cells))
        st.write(
            f"찾아낸 환자 {tp0:,}명에서 {tp1:,}명으로, "
            f"놓친 환자 {fn0:,}명에서 {fn1:,}명으로, "
            f"헛짚은 사람 {fp0:,}명에서 {fp1:,}명으로 바뀌었습니다."
        )

st.caption(
    "TP: 뇌졸중이라 예측했고 실제로 뇌졸중 / FN: 아니라고 예측했지만 실제로는 뇌졸중 / "
    "FP: 뇌졸중이라 예측했지만 실제로는 아님 / TN: 아니라고 예측했고 실제로 아님.
