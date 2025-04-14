import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from PIL import Image
import os
import json
from wordcloud import WordCloud
import matplotlib
import requests
from io import BytesIO
matplotlib.use('Agg')

# 현재 작업 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 스타일 설정
plt.style.use('seaborn')
plt.rcParams['font.family'] = 'NanumGothic'

# 한글 폰트 설정
def set_korean_font():
    try:
        # Source Han Sans KR 폰트 다운로드
        font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/Korean/SourceHanSansKR-Medium.otf"
        response = requests.get(font_url)
        font_path = BytesIO(response.content)
        
        # 폰트 등록
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        return True
    except Exception as e:
        st.error(f"폰트 설정 중 오류 발생: {str(e)}")
        return False

# 파일이 존재하는지 확인
def file_exists(filepath):
    """파일이 존재하는지 확인"""
    exists = os.path.exists(filepath)
    if not exists:
        st.warning(f"파일을 찾을 수 없습니다: {filepath}")
    return exists

# CSV 파일 로드
def load_csv(file_path):
    """CSV 파일 로드"""
    try:
        if file_exists(file_path):
            return pd.read_csv(file_path)
        return pd.DataFrame()  # 빈 DataFrame 반환
    except Exception as e:
        st.error(f"CSV 파일 로드 중 오류 발생: {str(e)}")
        return pd.DataFrame()  # 오류 발생 시 빈 DataFrame 반환

# 이미지 로드
def load_image(file_path):
    """이미지 파일 로드"""
    try:
        if file_exists(file_path):
            return Image.open(file_path)
        return None
    except Exception as e:
        st.error(f"이미지 로드 중 오류 발생: {str(e)}")
        return None

# 텍스트 파일 로드
def load_text(file_path):
    """텍스트 파일 로드"""
    try:
        if file_exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""  # 빈 문자열 반환
    except Exception as e:
        st.error(f"텍스트 파일 로드 중 오류 발생: {str(e)}")
        return ""  # 오류 발생 시 빈 문자열 반환

# JSON 파일 로드
def load_json(file_path):
    """JSON 파일 로드"""
    try:
        if file_exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}  # 빈 딕셔너리 반환
    except Exception as e:
        st.error(f"JSON 파일 로드 중 오류 발생: {str(e)}")
        return {}  # 오류 발생 시 빈 딕셔너리 반환

# 데모 데이터 생성 함수
def create_demo_data():
    np.random.seed(42)  # 재현성을 위한 시드 설정
    return pd.DataFrame({
        'region': ['서울', '경기', '부산'] * 10,
        'age': np.random.randint(20, 60, 30),
        'gender': ['남성', '여성'] * 15,
        'rating': np.random.randint(1, 6, 30)
    })

# 메인 함수
def main():
    # 페이지 설정
    st.set_page_config(
        page_title="자전거 데이터 분석 대시보드",
        page_icon="🚲",
        layout="wide"
    )
    
    # 한글 폰트 설정
    if not set_korean_font():
        st.warning("한글 폰트 설정에 실패했습니다. 기본 폰트를 사용합니다.")
    
    # 사이드바 메뉴
    st.sidebar.title("자전거 데이터 분석 대시보드")
    st.sidebar.image("https://img.freepik.com/free-vector/flat-design-bicycle-silhouette_23-2149156381.jpg", width=200)
    
    menu = st.sidebar.radio(
        "메뉴 선택",
        ["홈", "데이터 분석"]
    )
    
    # 데모 데이터 생성
    data = create_demo_data()
    
    if menu == "홈":
        st.title("자전거 시장 데이터 분석 대시보드")
        st.markdown("""
        ## 환영합니다! 👋
        이 대시보드는 자전거 관련 데이터를 분석하고 시각화하여 보여줍니다.
        
        ### 주요 기능
        - 📊 데이터 개요
        - 📈 데이터 시각화
        
        왼쪽 사이드바에서 원하는 메뉴를 선택하세요.
        """)
        
        st.info("현재 데모 데이터를 사용하고 있습니다.")
        
    elif menu == "데이터 분석":
        st.title("데이터 분석")
        
        tab1, tab2, tab3 = st.tabs(["데이터 개요", "지역별 분석", "연령/성별 분석"])
        
        with tab1:
            st.header("데이터 개요")
            st.dataframe(data)
            st.write("### 기본 통계")
            st.write(data.describe())
        
        with tab2:
            st.header("지역별 분석")
            fig, ax = plt.subplots(figsize=(10, 6))
            region_counts = data['region'].value_counts()
            region_counts.plot(kind='bar')
            plt.title("지역별 분포")
            plt.xlabel("지역")
            plt.ylabel("빈도")
            st.pyplot(fig)
            
            st.write("### 지역별 평균 평점")
            region_ratings = data.groupby('region')['rating'].mean()
            st.write(region_ratings)
        
        with tab3:
            st.header("연령/성별 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 연령 분포")
                fig, ax = plt.subplots()
                sns.histplot(data=data, x='age', bins=20)
                plt.title("연령 분포")
                st.pyplot(fig)
            
            with col2:
                st.write("### 성별 분포")
                fig, ax = plt.subplots()
                data['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title("성별 분포")
                st.pyplot(fig)
            
            st.write("### 성별 평균 평점")
            gender_ratings = data.groupby('gender')['rating'].mean()
            st.write(gender_ratings)

if __name__ == "__main__":
    main() 