import streamlit as st
import joblib
import numpy as np

# 'model.pkl' 파일은 앱이 배포되는 환경에 존재해야 합니다.
# 이 파일은 일반적으로 Scikit-learn 같은 라이브러리로 학습된 모델을 저장한 것입니다.
model = joblib.load('model.pkl')

st.title("꽃 분류기 (Iris Classifier)")
st.write("입력값을 기반으로 꽃의 종류를 예측합니다.")

# 사용자 입력 슬라이더
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# 입력 데이터를 numpy 배열로 변환
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# 모델 예측
pred = model.predict(input_data)
# 여기가 수정되었습니다: 'pred_calss' -> 'pred_class'
pred_class = pred[0]
class_names = ['Setosa', 'Versicolor', 'Virginica']

st.subheader("예측 결과:")
# 예측된 클래스 이름 출력
st.write(f"-> {class_names[pred_class]}")