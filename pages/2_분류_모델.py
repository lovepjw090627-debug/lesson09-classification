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
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🧪 분류 모델")
st.write(
    "여러 정보를 입력으로 넣어 뇌졸중인지 아닌지(`stroke`)를 맞히는 모델을 만들어 봅니다. "
    "여기서는 뇌졸중(`stroke`=1)을 **양성**, 아님(0)을 음성으로 둡니다."
)

try:
    df = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# ------------------------------------------------------------------
# 1. 입력 속성 고르기
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 2. 데이터 나누기 (번호 순으로 열 명씩 묶어 앞 세 명은 테스트용)
# ------------------------------------------------------------------
st.header("2. 훈련용과 테스트용으로 나누기")

data = df.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(data)) % 10) < 3  # 각 묶음(10명)의 앞 3명

train_df = data[~is_test]
test_df = data[is_test]

X_train = train_df[cols].astype(float).copy()
X_test = test_df[cols].astype(float).copy()
y_train = train_df["stroke"].to_numpy()
y_test = test_df["stroke"].to_numpy()

st.write(
    f"전체 {len(data):,}명을 번호(`id`) 순으로 세우고 열 명씩 묶었습니다. "
    f"각 묶음의 앞 세 명은 **테스트용 {len(test_df):,}명**으로 고정하고, "
    f"나머지 일곱 명은 **훈련용 {len(train_df):,}명**으로 씁니다. "
    f"(뇌졸중 비율: 훈련용 {y_train.mean():.2%}, 테스트용 {y_test.mean():.2%})"
)

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

# ------------------------------------------------------------------
# 모델 만들기 (학습과 크기 맞추기는 훈련용으로만)
# ------------------------------------------------------------------
logreg = Pipeline(
    [
        ("scaler", StandardScaler()),  # 크기 맞추기: 훈련용으로만 계산
        ("logreg", LogisticRegression(random_state=42)),
    ]
)
logreg.fit(X_train, y_train)

# 질문은 세 번까지, 사람이 5명 미만인 마디는 더 나누지 않음, 난수 고정
tree = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42)
tree.fit(X_train, y_train)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

# ------------------------------------------------------------------
# 3. 정확도 카드
# ------------------------------------------------------------------
st.header("3. 모델의 정확도")


def accuracy_card(name, model):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    with st.container(border=True):
        st.markdown(
            f'<div style="min-height:3.4em; font-weight:600;">{name}</div>',
            unsafe_allow_html=True,
        )
        st.metric("테스트 데이터 정확도", f"{test_acc:.2%}")
        st.markdown(
            '<div style="display:flex; justify-content:space-between; '
            'font-size:0.85rem; opacity:0.75;">'
            f"<span>훈련 데이터 {train_acc:.2%}</span>"
            f"<span>테스트 데이터 {test_acc:.2%}</span>"
            "</div>",
            unsafe_allow_html=True,
        )


card1, card2, card3 = st.columns(3)
with card1:
    accuracy_card(LOGREG_NAME, logreg)
with card2:
    accuracy_card(TREE_NAME, tree)
with card3:
    accuracy_card(BASE_NAME, baseline)

majority = int(baseline.predict(X_test.iloc[:1])[0])
st.caption(
    "다수 클래스 모델은 훈련용에서 더 많은 쪽인 "
    f"'{'뇌졸중' if majority == 1 else '뇌졸중 아님'}'으로만 답합니다. "
    "두 모델이 이 기준보다 나은지 비교해 보세요."
)

# ------------------------------------------------------------------
# 4. 두 속성으로 그린 그림
# ------------------------------------------------------------------
st.header("4. 두 속성으로 보는 모델의 경계")

axis_left, axis_right = st.columns(2)
with axis_left:
    x_col = st.selectbox("가로축", cols, format_func=lambda c: KOREAN[c])
with axis_right:
    y_col = st.selectbox(
        "세로축",
        [c for c in cols if c != x_col],
        format_func=lambda c: KOREAN[c],
    )

others = [c for c in cols if c not in (x_col, y_col)]
test_median = X_test[cols].median()  # 모델에 들어가는 테스트 데이터의 중앙값


def axis_range(col: str):
    if col in BINARY:
        return -0.3, 1.3
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    pad = (hi - lo) * 0.05
    if pad == 0:
        pad = 1.0
    return lo - pad, hi + pad


x_lo, x_hi = axis_range(x_col)
y_lo, y_hi = axis_range(y_col)

xs = np.linspace(x_lo, x_hi, 200)
ys = np.linspace(y_lo, y_hi, 200)
XX, YY = np.meshgrid(xs, ys)

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산
grid = pd.DataFrame({c: np.full(XX.size, test_median[c]) for c in cols})
grid[x_col] = XX.ravel()
grid[y_col] = YY.ravel()
grid = grid[cols]

prob = logreg.predict_proba(grid)[:, 1].reshape(XX.shape)  # 뇌졸중(1)일 확률
tree_answer = tree.predict(grid).reshape(XX.shape)

has_line = bool((prob >= 0.5).any() and (prob < 0.5).any())

fig = go.Figure()

# (1) 의사결정트리가 나눈 칸을 옅은 색으로
fig.add_trace(
    go.Heatmap(
        x=xs,
        y=ys,
        z=tree_answer,
        zmin=0,
        zmax=1,
        colorscale=[[0, REGION_NO], [1, REGION_YES]],
        showscale=False,
        hoverinfo="skip",
    )
)

# (2) 로지스틱 회귀가 확률 0.5로 가르는 선
if has_line:
    fig.add_trace(
        go.Contour(
            x=xs,
            y=ys,
            z=prob,
            contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
            colorscale=[[0, COLOR_LINE], [1, COLOR_LINE]],
            line=dict(width=3),
            showscale=False,
            showlegend=False,
            hoverinfo="skip",
        )
    )

# (3) 범례용 안내 표시
fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color=COLOR_LINE, width=3),
        name="로지스틱 회귀의 확률 0.5 경계선",
        legendrank=3,
    )
)
fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(symbol="square", size=14, color=REGION_NO.replace("0.15", "0.4")),
        name="의사결정트리 칸: 아님이라고 답함",
        legendrank=4,
    )
)
fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(symbol="square", size=14, color=REGION_YES.replace("0.15", "0.4")),
        name="의사결정트리 칸: 뇌졸중이라고 답함",
        legendrank=5,
    )
)

# (4) 테스트 데이터 점 (0/1 속성은 겹치지 않게 살짝 흩뜨려 그림)
rng = np.random.default_rng(0)
x_plot = X_test[x_col].to_numpy() + (
    rng.uniform(-0.08, 0.08, len(X_test)) if x_col in BINARY else 0
)
y_plot = X_test[y_col].to_numpy() + (
    rng.uniform(-0.08, 0.08, len(X_test)) if y_col in BINARY else 0
)

for label, value, color, size, rank in [
    ("실제 뇌졸중 아님", 0, COLOR_NO, 6, 1),
    ("실제 뇌졸중", 1, COLOR_YES, 8, 2),
]:
    mask = y_test == value
    fig.add_trace(
        go.Scatter(
            x=x_plot[mask],
            y=y_plot[mask],
            mode="markers",
            marker=dict(color=color, size=size, opacity=0.75),
            name=label,
            legendrank=rank,
        )
    )

fig.update_xaxes(range=[x_lo, x_hi], title=KOREAN[x_col])
fig.update_yaxes(range=[y_lo, y_hi], title=KOREAN[y_col])
for col, axis_update in [(x_col, fig.update_xaxes), (y_col, fig.update_yaxes)]:
    if col in BINARY:
        axis_update(tickmode="array", tickvals=[0, 1], ticktext=["없음", "있음"])
fig.update_layout(
    height=560,
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
)
st.plotly_chart(fig)

# 그림 아래 설명 글
if others:
    fixed_text = ", ".join(
        f"{KOREAN[c]} {test_median[c]:g}" if c in BINARY else f"{KOREAN[c]} {test_median[c]:.1f}"
        for c in others
    )
    st.write(
        f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: **{fixed_text}**"
        + (" (고혈압·심장병은 0=없음, 1=있음)" if any(c in BINARY for c in others) else "")
    )
else:
    st.write("고른 속성이 가로축과 세로축뿐이라 따로 세워 둔 값은 없습니다.")

if has_line:
    st.write("초록색 선은 로지스틱 회귀가 뇌졸중일 확률을 0.5로 보는 자리입니다.")
elif prob.max() < 0.5:
    st.write(
        "로지스틱 회귀의 0.5 경계선은 **그림 밖**에 있습니다. "
        f"이 그림 안에서는 뇌졸중일 확률이 가장 높은 곳도 {prob.max():.1%}라서 0.5에 닿지 않습니다."
    )
else:
    st.write(
        "로지스틱 회귀의 0.5 경계선은 **그림 밖**에 있습니다. "
        f"이 그림 안에서는 뇌졸중일 확률이 가장 낮은 곳도 {prob.min():.1%}라서 모두 0.5 이상입니다."
    )

notes = []
if x_col in BINARY or y_col in BINARY:
    notes.append("없음/있음 축의 점은 겹치지 않도록 조금 흩뜨려 그렸고, 계산은 0과 1 그대로 했습니다.")
if "bmi" in (x_col, y_col):
    notes.append("체질량지수가 비어 있던 사람은 훈련용 중앙값 자리에 찍혀 있습니다.")
if "bmi" in others:
    notes.append("체질량지수의 중앙값은 빈 값을 채운 뒤의 테스트 데이터로 구했습니다.")
for note in notes:
    st.caption(note)

# ------------------------------------------------------------------
# 5. 의사결정트리가 던진 질문 (가지 그림)
# ------------------------------------------------------------------
st.header("5. 의사결정트리가 던진 질문")

ARROW = "-" + ">"  # 가지 그림(DOT)에서 마디를 잇는 화살표 기호

QUESTION = {
    "age": "나이가 {t} 이하인가요?",
    "avg_glucose_level": "평균 혈당이 {t} 이하인가요?",
    "bmi": "체질량지수가 {t} 이하인가요?",
    "hypertension": "고혈압이 없나요?",
    "heart_disease": "심장병이 없나요?",
}


def build_tree_dot(model, X, y, feature_cols):
    """의사결정트리를 DOT 문자열로 바꾼다. 마디의 사람 수는 훈련용 기준."""
    y = np.asarray(y)
    t = model.tree_
    path = model.decision_path(X)  # 훈련용 사람이 지나간 마디 표시
    n_all = np.asarray(path.sum(axis=0)).ravel()
    n_pos = np.asarray(path[y == 1].sum(axis=0)).ravel()

    lines = [
        "digraph Tree {",
        'graph [bgcolor="white", ranksep=0.5, nodesep=0.3];',
        'node [shape=box, style="rounded,filled", fontname="sans-serif", '
        'fontsize=12, color="#555555", margin="0.15,0.08", width=2.4];',
        'edge [fontname="sans-serif", fontsize=12, color="#555555", fontcolor="#222222"];',
    ]

    leaf_answers = []
    asked = set()
    for node in range(t.node_count):
        n = int(n_all[node])
        k = int(n_pos[node])
        rate = k / n * 100 if n else 0.0
        stats = f"훈련 {n:,}명\\n뇌졸중 {k:,}명 ({rate:.1f}%)"

        if t.children_left[node] == -1:  # 더 묻지 않고 답을 내는 마디
            answer = int(model.classes_[np.argmax(t.value[node][0])])
            leaf_answers.append(answer)
            text = "뇌졸중" if answer == 1 else "아님"
            fill = "#f7c6c6" if answer == 1 else "#cfe3f5"
            label = f"답: {text}\\n{stats}"
        else:
            feature = feature_cols[t.feature[node]]
            asked.add(feature)
            question = QUESTION[feature].format(t=f"{t.threshold[node]:.1f}")
            fill = "#f2f2f2"
            label = f"{question}\\n{stats}"
        lines.append(f'n{node} [label="{label}", fillcolor="{fill}"];')

    for node in range(t.node_count):
        if t.children_left[node] != -1:
            lines.append(f'n{node} {ARROW} n{t.children_left[node]} [label="예"];')
            lines.append(f'n{node} {ARROW} n{t.children_right[node]} [label="아니요"];')

    lines.append("}")
    return "\n".join(lines), leaf_answers, asked


dot, leaf_answers, asked_features = build_tree_dot(tree, X_train, y_train, cols)
st.graphviz_chart(dot)

n_leaves = len(leaf_answers)
n_no = sum(1 for a in leaf_answers if a == 0)
asked_names = [KOREAN[c] for c in cols if c in asked_features]
not_asked_names = [KOREAN[c] for c in cols if c not in asked_features]

asked_line = (
    f"고른 속성 {len(cols)}개 가운데 이 나무가 실제로 물은 것은 "
    f"**{', '.join(asked_names) if asked_names else '없음(한 번도 묻지 않았어요)'}**입니다."
)
if asked_names and not_asked_names:
    asked_line += f" (묻지 않은 것: {', '.join(not_asked_names)})"

st.markdown(
    f"- 답을 내는 마디는 모두 **{n_leaves}칸**이고, 그중 **{n_no}칸**이 '아님'이라고 답합니다.\n"
    f"- {asked_line}"
)
