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
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

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
        else:
            st.warning(f"파일을 찾을 수 없습니다: {file_path}")
            return None
    except Exception as e:
        st.error(f"CSV 파일 로드 중 오류 발생: {str(e)}")
        return None

# 이미지 로드
def load_image(file_path):
    """이미지 파일 로드"""
    try:
        if file_exists(file_path):
            return Image.open(file_path)
        else:
            st.warning(f"이미지를 찾을 수 없습니다: {file_path}")
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
        else:
            st.warning(f"텍스트 파일을 찾을 수 없습니다: {file_path}")
            return None
    except Exception as e:
        st.error(f"텍스트 파일 로드 중 오류 발생: {str(e)}")
        return None

# JSON 파일 로드
def load_json(file_path):
    """JSON 파일 로드"""
    try:
        if file_exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            st.warning(f"JSON 파일을 찾을 수 없습니다: {file_path}")
            return None
    except Exception as e:
        st.error(f"JSON 파일 로드 중 오류 발생: {str(e)}")
        return None

# 메인 함수
def main():
    # 페이지 설정
    st.set_page_config(
        page_title="자전거 데이터 분석 대시보드",
        page_icon="🚲",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 한글 폰트 설정
    if not set_korean_font():
        st.error("한글 폰트 설정에 실패했습니다. 기본 폰트를 사용합니다.")
    
    # 사이드바 메뉴
    st.sidebar.title("자전거 데이터 분석 대시보드")
    st.sidebar.image("https://img.freepik.com/free-vector/flat-design-bicycle-silhouette_23-2149156381.jpg", width=200)
    
    menu = st.sidebar.radio(
        "메뉴 선택",
        ["홈", "데이터 개요", "감성 분석", "토픽 모델링", "키워드 네트워크", "페르소나", "마케팅 채널"]
    )
    
    # 데이터 로드
    data_path = os.path.join(OUTPUT_DIR, "data", "processed_data.csv")
    data = load_csv(data_path)
    
    # 홈
    if menu == "홈":
        st.title("자전거 시장 데이터 분석 대시보드")
        st.markdown("""
        이 대시보드는 자전거 관련 데이터 분석 결과를 시각화하여 제공합니다.
        
        ## 주요 기능
        - **데이터 개요**: 데이터의 기본 통계 및 분포 확인
        - **감성 분석**: 리뷰 텍스트의 감성 분석 결과
        - **토픽 모델링**: LDA를 활용한 토픽 모델링 결과
        - **키워드 네트워크**: 키워드 간 관계 시각화
        - **페르소나**: 고객 페르소나 프로필
        - **마케팅 채널**: 마케팅 채널 효과성 분석
        
        왼쪽 사이드바에서 메뉴를 선택하여 각 분석 결과를 확인하세요.
        """)
        
        # 데이터 분석 흐름도
        st.header("데이터 분석 흐름도")
        flow_chart = """
        ```mermaid
        graph TD
            A[데이터 수집] --> B[데이터 전처리]
            B --> C[탐색적 데이터 분석]
            C --> D[감성 분석]
            C --> E[토픽 모델링]
            C --> F[키워드 네트워크 분석]
            D --> G[페르소나 도출]
            E --> G
            F --> G
            G --> H[마케팅 채널 분석]
            H --> I[최종 보고서]
        ```
        """
        st.markdown(flow_chart)
        
    # 데이터 개요
    elif menu == "데이터 개요":
        st.title("데이터 개요")
        
        if data is None:
            st.error("데이터를 불러올 수 없습니다. 파일 경로를 확인해주세요.")
            return
            
        # 데이터 개요 탭
        tabs = st.tabs(["지역별 분포", "연령 분포", "성별 분포", "기타 통계"])
        
        with tabs[0]:
            st.header("지역별 선호도")
            region_counts = data['region'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            region_counts.plot(kind='bar', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with tabs[1]:
            st.header("연령 분포")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=data, x='age', bins=20, ax=ax)
            st.pyplot(fig)
        
        with tabs[2]:
            st.header("성별 분포")
            gender_counts = data['gender'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            gender_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)
        
        with tabs[3]:
            st.header("기타 통계")
            eda_path = os.path.join(OUTPUT_DIR, "eda_results", "eda_report.json")
            eda_report = load_json(eda_path)
            if eda_report:
                st.json(eda_report)
    
    # 감성 분석
    elif menu == "감성 분석":
        st.title("감성 분석 결과")
        
        # 감성 분석 탭
        tabs = st.tabs(["감성 분포", "예측 결과"])
        
        with tabs[0]:
            st.header("감성 분포")
            sentiment_path = os.path.join(OUTPUT_DIR, "sentiment", "sentiment_distribution.csv")
            sentiment_data = load_csv(sentiment_path)
            if sentiment_data is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                sentiment_data['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
                st.pyplot(fig)
        
        with tabs[1]:
            st.header("감성 예측 결과")
            predictions_path = os.path.join(OUTPUT_DIR, "sentiment", "predictions.csv")
            predictions = load_csv(predictions_path)
            if predictions is not None:
                st.dataframe(predictions.head())
    
    # 토픽 모델링
    elif menu == "토픽 모델링":
        st.title("토픽 모델링 결과")
        
        # 토픽 모델링 탭
        tabs = st.tabs(["주제 키워드", "적정 토픽 수", "대표 문서", "토픽 분포", "워드클라우드"])
        
        with tabs[0]:
            st.header("주제별 핵심 키워드")
            topics_path = os.path.join(OUTPUT_DIR, "topic", "topics.csv")
            topics = load_csv(topics_path)
            if topics is not None:
                st.dataframe(topics)
        
        with tabs[1]:
            st.header("적정 토픽 수")
            coherence_path = os.path.join(OUTPUT_DIR, "topic", "coherence_scores.csv")
            coherence = load_csv(coherence_path)
            if coherence is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(coherence['num_topics'], coherence['coherence_score'])
                plt.xlabel('토픽 수')
                plt.ylabel('일관성 점수')
                st.pyplot(fig)
        
        with tabs[2]:
            st.header("토픽별 대표 문서")
            docs_path = os.path.join(OUTPUT_DIR, "topic", "representative_docs.csv")
            representative_docs = load_csv(docs_path)
            if representative_docs is not None:
                st.dataframe(representative_docs)
        
        with tabs[3]:
            st.header("토픽 분포")
            dist_path = os.path.join(OUTPUT_DIR, "topic", "topic_distribution.csv")
            topic_dist = load_csv(dist_path)
            if topic_dist is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                topic_dist.plot(kind='bar', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
        
        with tabs[4]:
            st.header("토픽별 워드클라우드")
            wordcloud_path = os.path.join(OUTPUT_DIR, "topic", "wordcloud.png")
            wordcloud_img = load_image(wordcloud_path)
            if wordcloud_img is not None:
                st.image(wordcloud_img)
    
    # 키워드 네트워크
    elif menu == "키워드 네트워크":
        st.title("키워드 네트워크 분석")
        
        # 키워드 네트워크 탭
        tabs = st.tabs(["키워드 유사도", "키워드 네트워크", "테마별 분석"])
        
        with tabs[0]:
            st.header("키워드 유사도")
            similarity_path = os.path.join(OUTPUT_DIR, "keyword", "similarity_matrix.csv")
            similarity = load_csv(similarity_path)
            if similarity is not None:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(similarity, annot=True, cmap='YlOrRd', ax=ax)
                st.pyplot(fig)
                
            relations_path = os.path.join(OUTPUT_DIR, "keyword", "top_keyword_pairs.csv")
            keyword_relations = load_csv(relations_path)
            if keyword_relations is not None:
                st.dataframe(keyword_relations)
        
        with tabs[1]:
            st.header("키워드 네트워크")
            network_path = os.path.join(OUTPUT_DIR, "keyword", "network.png")
            network_img = load_image(network_path)
            if network_img is not None:
                st.image(network_img)
        
        with tabs[2]:
            st.header("테마별 키워드 네트워크")
            # 테마별 네트워크
            theme_tabs = st.tabs(["어린이 관련 키워드", "안전 관련 키워드", "디자인 관련 키워드"])
            
            for i, theme in enumerate(["어린이", "안전", "디자인"]):
                with theme_tabs[i]:
                    caption = f"{theme} 관련 키워드 네트워크"
                    file_path = os.path.join(OUTPUT_DIR, "keyword", f"theme_{theme}_network.png")
                    theme_img = load_image(file_path)
                    if theme_img:
                        st.image(theme_img, caption=caption, use_container_width=True)
                    
                    # 관련 데이터 표시
                    st.subheader(f"{theme} 관련 키워드 상위 관계")
                    relation_path = os.path.join(OUTPUT_DIR, "keyword", f"theme_{theme}_relations.csv")
                    relations_df = load_csv(relation_path)
                    if relations_df is not None:
                        st.dataframe(relations_df)
    
    # 페르소나
    elif menu == "페르소나":
        st.title("고객 페르소나 분석")
        
        # 페르소나 탭
        tabs = st.tabs(["페르소나 프로필", "레이더 차트", "고객 여정"])
        
        with tabs[0]:
            st.header("페르소나 프로필")
            profiles_path = os.path.join(OUTPUT_DIR, "persona", "cluster_profiles.csv")
            profiles = load_csv(profiles_path)
            if profiles is not None:
                st.dataframe(profiles)
        
        with tabs[1]:
            st.header("페르소나 레이더 차트")
            radar_path = os.path.join(OUTPUT_DIR, "persona", "radar_chart.png")
            radar_img = load_image(radar_path)
            if radar_img is not None:
                st.image(radar_img)
                
                # 클러스터 프로필 데이터
                profiles_path = os.path.join(OUTPUT_DIR, "persona", "cluster_profiles.csv")
                profiles_df = load_csv(profiles_path)
                if profiles_df is not None:
                    st.dataframe(profiles_df)
        
        with tabs[2]:
            st.header("고객 여정")
            journey_path = os.path.join(OUTPUT_DIR, "persona", "customer_journey.png")
            journey_img = load_image(journey_path)
            if journey_img is not None:
                st.image(journey_img)
    
    # 마케팅 채널
    elif menu == "마케팅 채널":
        st.title("마케팅 채널 분석")
        
        # 마케팅 채널 탭
        tabs = st.tabs(["채널 효과성", "전환 퍼널", "채널 맵"])
        
        with tabs[0]:
            st.header("마케팅 채널 효과성")
            channel_path = os.path.join(OUTPUT_DIR, "marketing", "channel_effectiveness.csv")
            channel_data = load_csv(channel_path)
            if channel_data is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                channel_data.plot(kind='bar', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            channel_df_path = os.path.join(OUTPUT_DIR, "persona", "marketing_channel_effectiveness.csv")
            channel_df = load_csv(channel_df_path)
            if channel_df is not None:
                st.dataframe(channel_df)
        
        with tabs[1]:
            st.header("페르소나별 전환 퍼널")
            funnel_path = os.path.join(OUTPUT_DIR, "marketing", "conversion_funnel.png")
            funnel_img = load_image(funnel_path)
            if funnel_img is not None:
                st.image(funnel_img)
        
        with tabs[2]:
            st.header("마케팅 채널 맵")
            map_path = os.path.join(OUTPUT_DIR, "marketing", "channel_map.csv")
            channel_map = load_csv(map_path)
            if channel_map is not None:
                st.dataframe(channel_map)
    
    # 데이터 인사이트
    st.header("데이터 기반 인사이트")
    st.markdown("""
    ### 데이터 속 숨겨진 가치
    
    1. **커뮤니케이션의 변화**: 자전거 데이터에서 보이는 것은 단순한 구매 패턴이 아닌, 소비자들의 라이프스타일 변화와 소통 방식의 변화입니다. 30-40대 남성의 높은 관심도는 가족 중심 문화와 건강에 대한 새로운 인식을 반영합니다.
    
    2. **감성의 연결성**: 데이터에서 드러난 키워드 간 연결성(안장-편안함, 디자인-심플함)은 소비자들이 제품을 단순한 기능이 아닌 '감성적 경험'으로 소비하고 있음을 보여줍니다. 이는 '물건'을 넘어 '이야기'를 판매해야 하는 시대로의 전환을 의미합니다.
    
    3. **세분화된 공감**: 페르소나 분석을 통해 발견된 다양한 고객군은 획일적 마케팅이 아닌, 개인 경험에 기반한 세분화된 공감이 필요함을 시사합니다. 이는 빅데이터가 아닌 '스몰데이터'의 가치, 즉 개인의 미시적 경험이 중요해지는 현상을 보여줍니다.
    
    4. **경계의 융합**: 온/오프라인 채널의 효과성 차이는 점차 사라지고 있으며, 이는 디지털과 아날로그의 경계가 무너지는 현대 소비 패턴을 반영합니다. 향후 소비자 경험은 이러한 경계가 없는 '초경험(Hyper-experience)'으로 진화할 것입니다.
    
    5. **공유와 순환**: 토픽 분석에서 드러난 '대여', '공유' 관련 키워드는 소유보다 접근과 경험을 중시하는 새로운 소비 문화의 태동을 보여줍니다. 이는 지속가능성과 순환경제로의 패러다임 전환을 시사합니다.
    """)
    
    # 푸터
    st.markdown("""
    ---
    © 2025 자전거 데이터 분석 프로젝트
    """)

if __name__ == "__main__":
    main() 