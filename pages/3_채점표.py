import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
st.set_page_config(
    page_title="채점표 - 뇌졸중 예측 실습실", page_icon="📝", layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
)

# '분류 모델' 페이지에서 처음 골라져 있는 네 속성과 같다
FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LOGREG_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "다수 클래스 모델(훈련용에서 많은 쪽으로만 답하는 모델)"

CANNOT = "계산할 수 없음"
LIST_LIMIT = 20  # 목록에 보여 줄 최대 줄 수


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("📝 채점표")
st.write(
    "테스트 데이터로 두 모델의 답을 채점합니다. 뇌졸중(`stroke`=1)을 **양성**으로 둡니다. "
    "'분류 모델' 페이지의 처음 설정과 같은 네 속성(나이, 평균 혈당, 고혈압, 심장병)으로 학습했습니다."
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

X_train = train_df[FEATURES].astype(float).copy()
X_test = test_df[FEATURES].astype(float).copy()
y_train = train_df["stroke"].to_numpy()
y_test = test_df["stroke"].to_numpy()

# ------------------------------------------------------------------
# 모델 학습 (크기 맞추기와 학습은 훈련용으로만)
# ------------------------------------------------------------------
logreg = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(random_state=42)),
    ]
)
logreg.fit(X_train, y_train)

tree = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42)
tree.fit(X_train, y_train)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

predictions = {
    LOGREG_NAME: logreg.predict(X_test),
    TREE_NAME: tree.predict(X_test),
    BASE_NAME: baseline.predict(X_test),
}


# ------------------------------------------------------------------
# 네 칸 세기와 지표 계산
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


def confusion_table(tp, fp, fn, tn):
    """행은 실제, 열은 예측인 2x2 표(합계 포함)."""
    table = pd.DataFrame(
        {
            "예측: 뇌졸중": [f"TP · {tp:,}명", f"FP · {fp:,}명", f"{tp + fp:,}명"],
            "예측: 아님": [f"FN · {fn:,}명", f"TN · {tn:,}명", f"{fn + tn:,}명"],
            "합계": [
                f"{tp + fn:,}명",
                f"{fp + tn:,}명",
                f"{tp + fp + fn + tn:,}명",
            ],
        },
        index=["실제: 뇌졸중", "실제: 아님", "합계"],
    )
    table.index.name = "행=실제 / 열=예측"
    return table


cells = {name: count_cells(y_test, pred) for name, pred in predictions.items()}

n_test = len(test_df)
n_test_stroke = int((y_test == 1).sum())
count_line = (
    f"테스트용은 **{n_test:,}명**이고, 그중 뇌졸중인 사람은 **{n_test_stroke:,}명**입니다."
)

# ------------------------------------------------------------------
# 1. 혼동행렬
# ------------------------------------------------------------------
st.header("1. 혼동행렬")
st.write(count_line)

left, right = st.columns(2)
for column, name in [(left, LOGREG_NAME), (right, TREE_NAME)]:
    tp, fp, fn, tn = cells[name]
    with column:
        st.markdown(f"**{name}**")
        st.table(confusion_table(tp, fp, fn, tn))

st.caption(
    "TP: 뇌졸중이라 예측했고 실제로 뇌졸중 / FP: 뇌졸중이라 예측했지만 실제로는 아님 / "
    "FN: 아니라고 예측했지만 실제로는 뇌졸중 / TN: 아니라고 예측했고 실제로 아님"
)

# ------------------------------------------------------------------
# 2. 네 가지 지표 비교
# ------------------------------------------------------------------
st.header("2. 정확도·재현율·정밀도·F1")
st.write(count_line)

rows = []
for name in [LOGREG_NAME, TREE_NAME, BASE_NAME]:
    accuracy, recall, precision, f1 = score_texts(*cells[name])
    rows.append(
        {
            "모델": name,
            "정확도": accuracy,
            "재현율": recall,
            "정밀도": precision,
            "F1": f1,
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True)

st.caption(
    "정확도 = (TP+TN) ÷ 전체 사람 수 / "
    "재현율 = TP ÷ (TP+FN), 분모는 실제 뇌졸중인 사람 수 / "
    "정밀도 = TP ÷ (TP+FP), 분모는 뇌졸중이라 예측한 사람 수 / "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율)"
)
st.caption(
    "뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1을, 식의 분모가 0이 되는 경우에는 "
    f"'{CANNOT}'이라고 적습니다."
)

# ------------------------------------------------------------------
# 3. 놓친 환자와 헛짚은 사람
# ------------------------------------------------------------------
st.header("3. 놓친 환자와 헛짚은 사람")

chosen = st.radio(
    "어느 모델의 목록을 볼까요?",
    [LOGREG_NAME, TREE_NAME],
    horizontal=True,
)
chosen_pred = predictions[chosen]
chosen_tp, chosen_fp, chosen_fn, chosen_tn = cells[chosen]

missed_mask = (y_test == 1) & (chosen_pred == 0)  # 실제 뇌졸중인데 아니라고 예측
wrong_mask = (y_test == 0) & (chosen_pred == 1)  # 아닌데 뇌졸중이라고 예측


def person_list(mask):
    """나이가 많은 순으로 정렬한 사람 목록과 전체 인원수."""
    part = test_df[mask].sort_values(["age", "id"], ascending=[False, True])
    table = pd.DataFrame(
        {
            "번호": part["id"].to_numpy(),
            "나이": part["age"].to_numpy(),
            "평균 혈당": part["avg_glucose_level"].to_numpy(),
            "고혈압 여부": np.where(part["hypertension"].to_numpy() == 1, "있음", "없음"),
            "심장병 여부": np.where(part["heart_disease"].to_numpy() == 1, "있음", "없음"),
        }
    )
    return len(table), table.head(LIST_LIMIT)


def show_list(title, note, mask):
    total, shown = person_list(mask)
    heading = f"{title} (전체 {total:,}명)"
    if total > LIST_LIMIT:
        heading += f" - 앞 {LIST_LIMIT}명만 표시"
    st.subheader(heading)
    st.caption(note)
    if total == 0:
        st.write("해당하는 사람이 없습니다.")
    else:
        st.dataframe(shown, hide_index=True)
    return total


st.write(f"**{chosen}**의 목록입니다. 나이가 많은 순서로 정렬했습니다.")

list_left, list_right = st.columns(2)
with list_left:
    n_missed = show_list(
        "놓친 환자", "실제 뇌졸중인데 아니라고 예측한 사람", missed_mask
    )
with list_right:
    n_wrong = show_list(
        "헛짚은 사람", "아닌데 뇌졸중이라고 예측한 사람", wrong_mask
    )

if n_missed == chosen_fn and n_wrong == chosen_fp:
    st.write(
        f"놓친 환자 {n_missed:,}명은 혼동행렬의 FN {chosen_fn:,}명과, "
        f"헛짚은 사람 {n_wrong:,}명은 FP {chosen_fp:,}명과 같습니다."
    )
else:
    st.error(
        f"목록 인원수(놓친 환자 {n_missed:,}명, 헛짚은 사람 {n_wrong:,}명)가 "
        f"혼동행렬(FN {chosen_fn:,}명, FP {chosen_fp:,}명)과 다릅니다."
    )
