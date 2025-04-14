import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import base64

# 페이지 설정
st.set_page_config(
    page_title="자전거 데이터 분석 대시보드",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 메뉴
st.sidebar.title("자전거 데이터 분석 대시보드")
page = st.sidebar.radio(
    "페이지 선택",
    ["📊 개요", "📈 토픽 모델링", "🔍 키워드 네트워크", "👥 페르소나 분석", "💼 비즈니스 전략"]
)

# 이미지 로드 함수
def load_image(image_path):
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        st.error(f"이미지를 찾을 수 없습니다: {image_path}")
        return None

# PDF 다운로드 버튼 생성 함수
def create_download_link(pdf_path, filename="자전거_데이터_분석_보고서.pdf"):
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📥 PDF 보고서 다운로드</a>'
        return href
    else:
        return "PDF 파일을 찾을 수 없습니다."

# 개요 페이지
def overview_page():
    st.title("자전거 데이터 분석 대시보드")
    
    st.markdown("""
    ## 👋 환영합니다!
    
    이 대시보드는 자전거 리뷰 데이터를 분석하여 도출한 인사이트와 비즈니스 전략을 시각화한 도구입니다.
    다양한 분석 기법을 통해 소비자 행동과 선호도를 파악하고, 이를 바탕으로 실질적인 매출 증대 방안을 제시합니다.
    
    ### 📌 주요 분석 내용
    
    * **감성 분석**: 리뷰 데이터의 긍정/부정 감성 분석
    * **토픽 모델링**: LDA 알고리즘을 활용한 주요 토픽 도출
    * **키워드 네트워크**: 연관 키워드 분석 및 시각화
    * **페르소나 분석**: 고객 세그먼트 군집화 및 페르소나 도출
    * **비즈니스 전략**: 데이터 기반 제품 및 마케팅 전략 제안
    """)
    
    # 감성 분석 결과 표시
    try:
        sentiment_data = pd.read_csv('output/eda_results/sentiment_distribution.csv')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("감성 분석 결과")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=sentiment_data.columns, y=sentiment_data.iloc[0], ax=ax)
            ax.set_title("리뷰 감성 분포")
            ax.set_ylabel("리뷰 수")
            ax.set_xlabel("감성")
            st.pyplot(fig)
            
        with col2:
            st.subheader("지역별 선호도")
            try:
                regional_data = pd.read_csv('output/eda_results/regional_preference.csv')
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # 상위 5개 지역만 표시
                top_regions = regional_data.iloc[0].sort_values(ascending=False).head(5)
                sns.barplot(x=top_regions.index, y=top_regions.values, ax=ax)
                ax.set_title("상위 5개 지역 선호도")
                ax.set_ylabel("빈도")
                ax.set_xlabel("지역")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"지역별 선호도 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    
    except Exception as e:
        st.error(f"감성 분석 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    
    # 인구통계학적 특성
    st.subheader("인구통계학적 특성")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                age_data = pd.read_csv('output/eda_results/age_distribution.csv')
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x=age_data.columns, y=age_data.iloc[0], ax=ax)
                ax.set_title("연령대별 분포")
                ax.set_ylabel("빈도")
                ax.set_xlabel("연령대")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"연령대 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        
        with col2:
            try:
                gender_data = pd.read_csv('output/eda_results/gender_distribution.csv')
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x=gender_data.columns, y=gender_data.iloc[0], ax=ax)
                ax.set_title("성별 분포")
                ax.set_ylabel("빈도")
                ax.set_xlabel("성별")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"성별 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    
    except Exception as e:
        st.error(f"인구통계학적 데이터를 불러오는 중 오류가 발생했습니다: {e}")

# 토픽 모델링 페이지
def topic_modeling_page():
    st.title("토픽 모델링 분석 결과")
    
    st.markdown("""
    ## LDA 토픽 모델링
    
    잠재 디리클레 할당(LDA) 알고리즘을 활용하여 리뷰 텍스트에서 주요 토픽을 추출했습니다.
    이를 통해 소비자들이 자전거에 대해 어떤 주제로 이야기하는지 파악할 수 있습니다.
    """)
    
    # 토픽 키워드 표시
    try:
        with open('output/topic_modeling/topics_keywords.txt', 'r', encoding='utf-8') as file:
            topics = file.readlines()
        
        st.subheader("주요 토픽 키워드")
        
        for i, topic in enumerate(topics):
            if topic.strip():  # 빈 줄 건너뛰기
                st.markdown(f"**{topic.strip()}**")
    except Exception as e:
        st.error(f"토픽 키워드를 불러오는 중 오류가 발생했습니다: {e}")
    
    # 토픽 분포 시각화
    st.subheader("토픽 분포")
    topic_dist_img = load_image('output/topic_modeling/topic_distribution.png')
    if topic_dist_img:
        st.image(topic_dist_img, use_column_width=True)
    
    # 최적 토픽 수 결정 시각화
    st.subheader("최적 토픽 수 결정 (Perplexity)")
    perplexity_img = load_image('output/topic_modeling/perplexity_score.png')
    if perplexity_img:
        st.image(perplexity_img, use_column_width=True)
    
    # 워드클라우드 표시
    st.subheader("토픽별 워드클라우드")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["토픽 1", "토픽 2", "토픽 3", "토픽 4", "토픽 5"])
    
    with tab1:
        topic1_img = load_image('output/topic_modeling/topic_1_wordcloud.png')
        if topic1_img:
            st.image(topic1_img, use_column_width=True)
    
    with tab2:
        topic2_img = load_image('output/topic_modeling/topic_2_wordcloud.png')
        if topic2_img:
            st.image(topic2_img, use_column_width=True)
    
    with tab3:
        topic3_img = load_image('output/topic_modeling/topic_3_wordcloud.png')
        if topic3_img:
            st.image(topic3_img, use_column_width=True)
    
    with tab4:
        topic4_img = load_image('output/topic_modeling/topic_4_wordcloud.png')
        if topic4_img:
            st.image(topic4_img, use_column_width=True)
    
    with tab5:
        topic5_img = load_image('output/topic_modeling/topic_5_wordcloud.png')
        if topic5_img:
            st.image(topic5_img, use_column_width=True)
    
    # 대표 문서 표시
    st.subheader("토픽별 대표 문서")
    
    try:
        with open('output/topic_modeling/representative_documents.txt', 'r', encoding='utf-8') as file:
            rep_docs = file.read()
        
        st.text_area("대표 문서", rep_docs, height=300)
    except Exception as e:
        st.error(f"대표 문서를 불러오는 중 오류가 발생했습니다: {e}")

# 키워드 네트워크 페이지
def keyword_network_page():
    st.title("키워드 네트워크 분석 결과")
    
    st.markdown("""
    ## 키워드 연관성 분석
    
    리뷰 텍스트에서 추출한 키워드 간의 연관성을 분석하여 네트워크로 시각화했습니다.
    이를 통해 소비자들이 자전거의 어떤 특성을 함께 언급하는지 파악할 수 있습니다.
    """)
    
    # 키워드 유사도 히트맵
    st.subheader("키워드 유사도 히트맵")
    heatmap_img = load_image('output/keyword_network/keyword_similarity_heatmap.png')
    if heatmap_img:
        st.image(heatmap_img, use_column_width=True)
    
    # 키워드 네트워크 그래프
    st.subheader("키워드 네트워크 그래프")
    network_img = load_image('output/keyword_network/keyword_network.png')
    if network_img:
        st.image(network_img, use_column_width=True)
    
    # 특정 테마별 키워드 네트워크
    st.subheader("테마별 키워드 네트워크")
    
    tab1, tab2, tab3 = st.tabs(["안전", "어린이", "디자인"])
    
    with tab1:
        safety_img = load_image('output/keyword_network/theme_안전_network.png')
        if safety_img:
            st.image(safety_img, use_column_width=True)
    
    with tab2:
        children_img = load_image('output/keyword_network/theme_어린이_network.png')
        if children_img:
            st.image(children_img, use_column_width=True)
    
    with tab3:
        design_img = load_image('output/keyword_network/theme_디자인_network.png')
        if design_img:
            st.image(design_img, use_column_width=True)

# 페르소나 분석 페이지
def persona_page():
    st.title("페르소나 분석 결과")
    
    st.markdown("""
    ## 고객 세그먼트 및 페르소나
    
    리뷰 데이터, 인구통계학적 특성, 구매 패턴 등을 기반으로 K-means 군집화를 수행하여 고객 세그먼트를 도출하고,
    각 세그먼트를 대표하는 페르소나를 개발했습니다.
    """)
    
    # 페르소나 레이더 차트
    st.subheader("페르소나 레이더 차트")
    radar_img = load_image('output/persona/persona_radar_charts.png')
    if radar_img:
        st.image(radar_img, use_column_width=True)
    
    # 페르소나 설명
    st.subheader("페르소나 프로필")
    
    try:
        with open('output/persona/persona_descriptions.txt', 'r', encoding='utf-8') as file:
            personas = file.read()
        
        st.markdown(personas.replace("■", "### "))
    except Exception as e:
        st.error(f"페르소나 설명을 불러오는 중 오류가 발생했습니다: {e}")
    
    # 고객 여정 타임라인
    st.subheader("고객 여정 타임라인")
    journey_img = load_image('output/persona/customer_journey_timeline.png')
    if journey_img:
        st.image(journey_img, use_column_width=True)
    
    # 마케팅 채널 효과성
    st.subheader("마케팅 채널 효과성")
    
    col1, col2 = st.columns(2)
    
    with col1:
        channel_img = load_image('output/persona/marketing_channel_effectiveness.png')
        if channel_img:
            st.image(channel_img, use_column_width=True)
    
    with col2:
        try:
            channel_data = pd.read_csv('output/persona/marketing_channel_effectiveness.csv', index_col=0)
            st.dataframe(channel_data)
        except Exception as e:
            st.error(f"마케팅 채널 데이터를 불러오는 중 오류가 발생했습니다: {e}")
    
    # 전환 퍼널 및 마케팅 채널 맵
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("페르소나별 전환 퍼널")
        funnel_img = load_image('output/persona/conversion_funnel_by_persona.png')
        if funnel_img:
            st.image(funnel_img, use_column_width=True)
    
    with col2:
        st.subheader("마케팅 채널 맵")
        map_img = load_image('output/persona/marketing_channel_map.png')
        if map_img:
            st.image(map_img, use_column_width=True)

# 비즈니스 전략 페이지
def business_strategy_page():
    st.title("비즈니스 전략 제안")
    
    st.markdown("""
    ## 데이터 기반 매출 개선 전략
    
    토픽 모델링, 키워드 네트워크, 페르소나 분석 등의 결과를 종합하여 실질적인 매출 증대를 위한
    전략적 제안을 제시합니다.
    """)
    
    # 비즈니스 전략 문서 표시
    try:
        with open('business_strategy_recommendations.md', 'r', encoding='utf-8') as file:
            strategy = file.read()
        
        st.markdown(strategy)
    except Exception as e:
        st.error(f"비즈니스 전략 문서를 불러오는 중 오류가 발생했습니다: {e}")
    
    # 마케팅 전략 요약 표시
    st.subheader("마케팅 전략 실행 계획")
    
    try:
        with open('marketing_strategy_summary.md', 'r', encoding='utf-8') as file:
            marketing_strategy = file.read()
        
        tab1, tab2 = st.tabs(["타임라인", "ROI 예측"])
        
        with tab1:
            st.markdown(marketing_strategy.split("## 투자 대비 효과 예측")[0])
        
        with tab2:
            split_content = marketing_strategy.split("## 투자 대비 효과 예측")
            if len(split_content) > 1:
                roi_content = "## 투자 대비 효과 예측" + split_content[1].split("## 핵심 성공 요소")[0]
                st.markdown(roi_content)
    except Exception as e:
        st.error(f"마케팅 전략 요약을 불러오는 중 오류가 발생했습니다: {e}")
    
    # PDF 다운로드 버튼
    st.markdown("## 전체 보고서 다운로드")
    st.markdown("분석 결과와 전략 제안을 종합한 PDF 보고서를 다운로드할 수 있습니다.")
    
    pdf_path = "자전거_데이터_분석_보고서.pdf"
    st.markdown(create_download_link(pdf_path), unsafe_allow_html=True)

# 메인 실행부
if page == "📊 개요":
    overview_page()
elif page == "📈 토픽 모델링":
    topic_modeling_page()
elif page == "🔍 키워드 네트워크":
    keyword_network_page()
elif page == "👥 페르소나 분석":
    persona_page()
elif page == "💼 비즈니스 전략":
    business_strategy_page()

# 푸터
st.markdown("---")
st.markdown("© 2025 자전거 데이터 분석 프로젝트. All rights reserved.") 