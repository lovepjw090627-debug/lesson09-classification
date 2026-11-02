import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
st.set_page_config(
    page_title="탐색 - 뇌졸중 예측 실습실", page_icon="🔍", layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
)


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔍 데이터 탐색")

try:
    df = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# ------------------------------------------------------------------
# 1. 나이와 평균 혈당의 분포
# ------------------------------------------------------------------
st.header("1. 나이와 평균 혈당의 분포")

col_age, col_glucose = st.columns(2)

with col_age:
    fig_age = px.histogram(
        df, x="age", nbins=30, title="나이 분포", labels={"age": "나이"}
    )
    fig_age.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig_age)

with col_glucose:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig_glucose.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig_glucose)

# ------------------------------------------------------------------
# 2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교
# ------------------------------------------------------------------
st.header("2. 뇌졸중을 겪은 사람 vs 겪지 않은 사람")

GROUP_ORDER = ["겪지 않은 사람", "겪은 사람"]
grouped = df.assign(그룹=df["stroke"].map({0: GROUP_ORDER[0], 1: GROUP_ORDER[1]}))

box_age, box_glucose = st.columns(2)

with box_age:
    fig_box_age = px.box(
        grouped,
        x="그룹",
        y="age",
        color="그룹",
        category_orders={"그룹": GROUP_ORDER},
        title="나이 비교",
        labels={"age": "나이"},
    )
    fig_box_age.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig_box_age)

with box_glucose:
    fig_box_glucose = px.box(
        grouped,
        x="그룹",
        y="avg_glucose_level",
        color="그룹",
        category_orders={"그룹": GROUP_ORDER},
        title="평균 혈당 비교",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig_box_glucose.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig_box_glucose)

st.subheader("두 그룹의 평균값")
mean_table = (
    grouped.groupby("그룹")
    .agg(
        **{
            "사람 수": ("stroke", "size"),
            "나이 평균": ("age", "mean"),
            "평균 혈당의 평균": ("avg_glucose_level", "mean"),
        }
    )
    .reindex(GROUP_ORDER)
    .round(2)
    .reset_index()
)
st.dataframe(mean_table, hide_index=True)

# ------------------------------------------------------------------
# 3. 고혈압·심장병 여부에 따른 뇌졸중 비율
# ------------------------------------------------------------------
st.header("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")


def stroke_rate_chart(column: str, title: str):
    """column 값(0/1)별로 뇌졸중 비율(%)을 막대그래프로 만든다."""
    rate = (
        df.groupby(column)["stroke"]
        .agg(비율="mean", 사람_수="size")
        .reset_index()
    )
    rate["뇌졸중 비율(%)"] = (rate["비율"] * 100).round(2)
    rate["구분"] = rate[column].map({0: "없음", 1: "있음"})
    rate = rate.rename(columns={"사람_수": "사람 수"})

    fig = px.bar(
        rate,
        x="구분",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        hover_data=["사람 수"],
        category_orders={"구분": ["없음", "있음"]},
        title=title,
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(
        xaxis_title=None,
        yaxis_range=[0, max(rate["뇌졸중 비율(%)"].max() * 1.3, 1)],
    )
    return fig


rate_hyper, rate_heart = st.columns(2)

with rate_hyper:
    st.plotly_chart(stroke_rate_chart("hypertension", "고혈압 여부별 뇌졸중 비율"))

with rate_heart:
    st.plotly_chart(
        stroke_rate_chart("heart_disease", "심장병 여부별 뇌졸중 비율")
    )

# ------------------------------------------------------------------
# 4. bmi가 비어 있는 사람들
# ------------------------------------------------------------------
st.header("4. bmi가 비어 있는 사람들")

bmi_missing = df[df["bmi"].isna()]


def summarize(label: str, part: pd.DataFrame) -> dict:
    n = len(part)
    n_stroke = int((part["stroke"] == 1).sum())
    ratio = f"{n_stroke / n * 100:.2f}%" if n else "-"
    return {
        "구분": label,
        "사람 수": n,
        "뇌졸중을 겪은 사람 수": n_stroke,
        "뇌졸중 비율": ratio,
    }


st.write(f"bmi가 비어 있는 사람은 **{len(bmi_missing):,}명**입니다.")
st.dataframe(
    pd.DataFrame(
        [
            summarize("bmi가 빈 사람들", bmi_missing),
            summarize("전체", df),
        ]
    ),
    hide_index=True,
)

# ------------------------------------------------------------------
# 5. 흡연 상태별 사람 수
# ------------------------------------------------------------------
st.header("5. 흡연 상태별 사람 수")

smoking_counts = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
smoking_counts["흡연 상태"] = smoking_counts["흡연 상태"].fillna("(빈 값)")
st.dataframe(smoking_counts, hide_index=True)
