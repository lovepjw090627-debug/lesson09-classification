import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# 기본 설정 (브라우저 탭 제목과 아이콘)
# ------------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
)
SAVE_DIR = Path(__file__).parent / "내_기록"
SAVE_FILE = SAVE_DIR / "데이터_소개_기록.json"

# 숫자 카드를 크게 보이게 하는 스타일
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 2.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


# ------------------------------------------------------------------
# 내 기록(우리말 뜻, 출처) 저장·불러오기
# ------------------------------------------------------------------
def load_notes() -> dict:
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_notes(meanings: dict, source: str) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"우리말_뜻": meanings, "출처": source},
            f,
            ensure_ascii=False,
            indent=2,
        )


# ------------------------------------------------------------------
# 열 설명 표 만들기
# ------------------------------------------------------------------
def describe_values(series: pd.Series) -> str:
    """열의 '값의 종류'를 짧은 글로 요약한다."""
    values = series.dropna()
    n_unique = values.nunique()

    if n_unique <= 10:
        return ", ".join(str(v) for v in sorted(values.unique(), key=str))

    if pd.api.types.is_numeric_dtype(series):
        return f"숫자 {values.min():g} ~ {values.max():g} (서로 다른 값 {n_unique}개)"

    return f"글자 (서로 다른 값 {n_unique}개)"


def build_column_table(df: pd.DataFrame, meanings: dict) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "열 이름": col,
                "우리말 뜻": meanings.get(col, ""),
                "값의 종류": describe_values(df[col]),
                "빈 값 개수": int(df[col].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 화면 시작
# ------------------------------------------------------------------
st.title(f"{APP_ICON} {APP_TITLE}")

try:
    df = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# 처음 한 번만 저장된 기록을 불러온다 (새로고침하면 파일에서 다시 불러옴)
if "notes_loaded" not in st.session_state:
    notes = load_notes()
    st.session_state["saved_meanings"] = notes.get("우리말_뜻", {})
    st.session_state["saved_source"] = notes.get("출처", "")
    st.session_state["base_table"] = build_column_table(
        df, st.session_state["saved_meanings"]
    )
    st.session_state["source_text"] = st.session_state["saved_source"]
    st.session_state["notes_loaded"] = True

# ------------------------------------------------------------------
# 1. 데이터 소개
# ------------------------------------------------------------------
st.header("1. 이 데이터는 무엇일까요?")
st.write(
    "사람마다 한 줄씩, 성별·나이·혈압·심장병·생활 습관 같은 정보와 "
    "뇌졸중 여부(`stroke`)가 기록된 데이터입니다. "
    "`stroke`가 1이면 뇌졸중을 겪은 사람, 0이면 아닌 사람입니다."
)

total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    with st.container(border=True):
        st.metric("전체 사람 수", f"{total:,}명")
with c2:
    with st.container(border=True):
        st.metric("열 개수", f"{n_cols}개")
with c3:
    with st.container(border=True):
        st.metric("stroke가 1인 사람 수", f"{n_stroke:,}명")
with c4:
    with st.container(border=True):
        st.metric("stroke가 1인 비율", f"{ratio:.2f}%")

# ------------------------------------------------------------------
# 2. 열 설명 표 (우리말 뜻은 직접 입력)
# ------------------------------------------------------------------
st.header("2. 열 설명 표")
st.caption("‘우리말 뜻’ 칸을 두 번 눌러 교재를 보고 직접 적어 보세요. 적은 내용은 자동으로 저장됩니다.")

edited = st.data_editor(
    st.session_state["base_table"],
    key="column_table",
    hide_index=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻", help="교재를 보고 직접 적어 넣으세요.", width="medium"
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수"),
    },
)

# ------------------------------------------------------------------
# 3. 처음 다섯 줄
# ------------------------------------------------------------------
st.header("3. 데이터 처음 다섯 줄")
st.dataframe(df.head(5), hide_index=True)

# ------------------------------------------------------------------
# 4. 데이터 출처
# ------------------------------------------------------------------
st.header("4. 데이터 출처")
st.caption("교재에 나온 출처를 아래에 적어 주세요. 입력 칸 밖을 누르면 저장됩니다.")
source_text = st.text_area(
    "출처",
    key="source_text",
    height=120,
    placeholder="여기에 출처를 적어 주세요.",
    label_visibility="collapsed",
)

# ------------------------------------------------------------------
# 바뀐 내용이 있으면 json 파일로 저장
# ------------------------------------------------------------------
current_meanings = {
    row["열 이름"]: (row["우리말 뜻"] or "").strip()
    for _, row in edited.iterrows()
    if (row["우리말 뜻"] or "").strip()
}

if (
    current_meanings != st.session_state["saved_meanings"]
    or source_text != st.session_state["saved_source"]
):
    try:
        save_notes(current_meanings, source_text)
        st.session_state["saved_meanings"] = current_meanings
        st.session_state["saved_source"] = source_text
        st.toast("내_기록 폴더에 저장했어요.", icon="💾")
    except OSError as e:
        st.error(f"저장하지 못했습니다: {e}")
