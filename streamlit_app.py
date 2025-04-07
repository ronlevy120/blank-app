import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="דשבורד לדוגמה", layout="wide")

st.title("דשבורד מכירות - דוגמה")

# נתונים פיקטיביים
data = {
    "חודש": ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני"],
    "מכירות": [15000, 18000, 12000, 20000, 22000, 17000],
    "קטגוריה": ["אלקטרוניקה", "ביגוד", "אלקטרוניקה", "בית", "ביגוד", "בית"]
}
df = pd.DataFrame(data)

# פילטר לפי קטגוריה
קטגוריה = st.selectbox("בחר קטגוריה", options=["הכל"] + df["קטגוריה"].unique().tolist())

if קטגוריה != "הכל":
    df = df[df["קטגוריה"] == קטגוריה]

# הצגת טבלה
st.subheader("טבלת נתונים")
st.dataframe(df)

# גרף
st.subheader("גרף מכירות")
chart = alt.Chart(df).mark_bar().encode(
    x='חודש',
    y='מכירות',
    color='קטגוריה'
).properties(width=700)
st.altair_chart(chart)