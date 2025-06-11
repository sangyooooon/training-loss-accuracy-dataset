import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page title
st.title("📉 머신러닝 학습 손실 & 정확도 시각화")

# Upload CSV file
uploaded_file = st.file_uploader("CSV 파일 업로드 (예: 1_scenario_5kfold.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # 데이터 확인용 출력
        if st.checkbox("원본 데이터 미리보기"):
            st.write(df.head())

        # epoch, loss, accuracy 관련 열만 필터링
        expected_columns = [
            'epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
        ]

        # 열 존재 여부 확인
        if not all(col in df.columns for col in expected_columns):
            st.error("CSV 파일에 필요한 열이 없습니다: " + ", ".join(expected_columns))
        else:
            df = df[expected_columns].sort_values('epoch')

            # 시각화: 손실 그래프
            st.subheader("📉 손실(Loss) 곡선")
            fig1, ax1 = plt.subplots()
            ax1.plot(df['epoch'], df['train_loss'], label='Train Loss')
            ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)

            # 시각화: 정확도 그래프
            st.subheader("✅ 정확도(Accuracy) 곡선")
            fig2, ax2 = plt.subplots()
            ax2.plot(df['epoch'], df['train_accuracy'], label='Train Accuracy')
            ax2.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"파일을 읽는 중 오류 발생: {e}")
else:
    st.info("왼쪽 사이드바에서 CSV 파일을 업로드하세요.")

