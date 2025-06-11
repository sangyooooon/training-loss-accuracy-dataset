streamlit
pandas
matplotlibimport streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page title
st.title("📉 머신러닝 학습 손실 & 정확도 시각화")

# CSV 파일 경로 (저장소 루트에 위치한 파일)
CSV_PATH = "1_scenario_5kfold.csv"

# CSV 파일 존재 여부 확인 후 불러오기
try:
    df = pd.read_csv(CSV_PATH)

    if st.checkbox("원본 데이터 미리보기"):
        st.write(df.head())

    # 필요한 열만 추출
    expected_columns = ['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']
    if not all(col in df.columns for col in expected_columns):
        st.error("CSV 파일에 다음 열이 있어야 합니다: " + ", ".join(expected_columns))
    else:
        df = df[expected_columns].sort_values('epoch')

        # 손실 시각화
        st.subheader("📉 손실(Loss) 곡선")
        fig1, ax1 = plt.subplots()
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss')
        ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        # 정확도 시각화
        st.subheader("✅ 정확도(Accuracy) 곡선")
        fig2, ax2 = plt.subplots()
        ax2.plot(df['epoch'], df['train_accuracy'], label='Train Accuracy')
        ax2.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

except FileNotFoundError:
    st.warning(f"'{CSV_PATH}' 파일이 저장소 루트에 없습니다. GitHub 저장소의 최상위 경로에 파일을 넣어주세요.")
except Exception as e:
    st.error(f"CSV 파일 읽기 오류: {e}")
