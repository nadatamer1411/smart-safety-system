import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Safety Dashboard", layout="wide")

st.title("🚨 Smart Safety Monitoring Dashboard")

file_name = "data/logs.csv"

if not os.path.exists(file_name):
    st.warning("No data available yet")
else:
    df = pd.read_csv(file_name)

    # عدد المخالفات
    st.metric("Total Violations", len(df))

    st.divider()

    # جدول البيانات
    st.subheader("📋 Violations Log")
    st.dataframe(df)

    st.divider()

    # الصور
    st.subheader("📸 Violation Images")

    if "image" in df.columns:
        for i in range(len(df)):
            col1, col2 = st.columns([1, 2])

            # إصلاح المسار
            image_name = os.path.basename(df.loc[i, "image"])
            full_path = os.path.join("data", "images", image_name)
            full_path = os.path.abspath(full_path)

            with col1:
                if os.path.exists(full_path):
                    st.image(full_path, width=250)
                else:
                    st.warning("Image not found")

            with col2:
                st.write(f"**🕒 Time:** {df.loc[i, 'time']}")
                st.write(f"**⚠️ Violation:** {df.loc[i, 'violation']}")
                st.write("---")