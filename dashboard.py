import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 페이지 설정
st.set_page_config(
    page_title="자전거 데이터 분석 대시보드",
    page_icon="🚲",
    layout="wide"
)

# 데모 데이터 생성
def create_demo_data():
    np.random.seed(42)
    return pd.DataFrame({
        'region': ['서울', '경기', '부산'] * 10,
        'age': np.random.randint(20, 60, 30),
        'gender': ['남성', '여성'] * 15,
        'rating': np.random.randint(1, 6, 30)
    })

def main():
    # 사이드바
    st.sidebar.title("자전거 데이터 분석 대시보드")
    st.sidebar.image("https://img.freepik.com/free-vector/flat-design-bicycle-silhouette_23-2149156381.jpg", width=200)
    
    menu = st.sidebar.radio(
        "메뉴",
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
            
            # 지역별 분포
            fig = px.bar(
                data_frame=data['region'].value_counts().reset_index(),
                x='index',
                y='region',
                title="지역별 분포",
                labels={'index': '지역', 'region': '빈도'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 지역별 평균 평점
            st.write("### 지역별 평균 평점")
            region_ratings = data.groupby('region')['rating'].mean().round(2)
            
            fig = px.bar(
                data_frame=region_ratings.reset_index(),
                x='region',
                y='rating',
                title="지역별 평균 평점",
                labels={'region': '지역', 'rating': '평균 평점'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("연령/성별 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 연령 분포")
                fig = px.histogram(
                    data_frame=data,
                    x='age',
                    nbins=20,
                    title="연령 분포",
                    labels={'age': '연령', 'count': '빈도'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### 성별 분포")
                fig = px.pie(
                    data_frame=data,
                    names='gender',
                    title="성별 분포"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.write("### 성별 평균 평점")
            gender_ratings = data.groupby('gender')['rating'].mean().round(2)
            
            fig = px.bar(
                data_frame=gender_ratings.reset_index(),
                x='gender',
                y='rating',
                title="성별 평균 평점",
                labels={'gender': '성별', 'rating': '평균 평점'}
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 