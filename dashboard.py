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
import platform
matplotlib.use('Agg')

# 한글 폰트 설정
def set_korean_font():
    plt.rcParams['axes.unicode_minus'] = False
    
    # Streamlit Cloud에서는 NanumGothic 폰트 사용
    plt.rc('font', family='NanumGothic')
    
    print(f"폰트 설정 완료: {plt.rcParams['font.family']}")

# 파일이 존재하는지 확인
def file_exists(filepath):
    """파일이 존재하는지 확인"""
    return os.path.exists(filepath)

# CSV 파일 로드
def load_csv(filepath, encoding='utf-8'):
    """CSV 파일 로드"""
    try:
        if file_exists(filepath):
            return pd.read_csv(filepath, encoding=encoding)
        else:
            st.warning(f"파일을 찾을 수 없습니다: {filepath}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"파일 로드 중 오류 발생: {e}")
        return pd.DataFrame()

# 이미지 로드
def load_image(filepath):
    """이미지 파일 로드"""
    try:
        if file_exists(filepath):
            return Image.open(filepath)
        else:
            st.warning(f"이미지를 찾을 수 없습니다: {filepath}")
            return None
    except Exception as e:
        st.error(f"이미지 로드 중 오류 발생: {e}")
        return None

# 텍스트 파일 로드
def load_text(filepath, encoding='utf-8'):
    """텍스트 파일 로드"""
    try:
        if file_exists(filepath):
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        else:
            st.warning(f"텍스트 파일을 찾을 수 없습니다: {filepath}")
            return ""
    except Exception as e:
        st.error(f"텍스트 파일 로드 중 오류 발생: {e}")
        return ""

# JSON 파일 로드
def load_json(filepath, encoding='utf-8'):
    """JSON 파일 로드"""
    try:
        if file_exists(filepath):
            with open(filepath, 'r', encoding=encoding) as f:
                return json.load(f)
        else:
            st.warning(f"JSON 파일을 찾을 수 없습니다: {filepath}")
            return {}
    except Exception as e:
        st.error(f"JSON 파일 로드 중 오류 발생: {e}")
        return {}

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
    set_korean_font()
    
    # 사이드바 메뉴
    st.sidebar.title("자전거 데이터 분석 대시보드")
    st.sidebar.image("https://img.freepik.com/free-vector/flat-design-bicycle-silhouette_23-2149156381.jpg", width=200)
    
    menu = st.sidebar.radio(
        "메뉴 선택",
        ["홈", "데이터 개요", "감성 분석", "토픽 모델링", "키워드 네트워크", "페르소나", "마케팅 채널"]
    )
    
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
        
        # 데이터 개요 탭
        tabs = st.tabs(["지역별 분포", "연령 분포", "성별 분포", "기타 통계"])
        
        with tabs[0]:
            st.header("지역별 선호도")
            region_df = load_csv("output/eda_results/regional_preference.csv")
            if not region_df.empty:
                # 지역별 합계 계산
                region_summary = region_df.groupby('region')['count'].sum().reset_index()
                region_summary = region_summary.sort_values('count', ascending=False)
                
                # 카테고리별 시각화
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # 전체 지역 분포 바 차트
                sns.barplot(x='count', y='region', data=region_summary.head(10), ax=ax1)
                ax1.set_title('전체 지역별 선호도 (상위 10개)')
                ax1.set_xlabel('인원 수')
                ax1.set_ylabel('지역')
                
                # 카테고리별 지역 분포 시각화
                pivot_df = pd.pivot_table(region_df, values='count', index='region', 
                                         columns='category', aggfunc='sum').fillna(0)
                top_regions = region_summary.head(7)['region'].tolist()
                category_region = pivot_df.loc[top_regions].reset_index()
                
                # 가독성을 위해 데이터 정렬 및 멜트
                melted_df = pd.melt(category_region, id_vars='region', var_name='category', value_name='count')
                sns.barplot(x='region', y='count', hue='category', data=melted_df, ax=ax2)
                ax2.set_title('카테고리별 지역 분포 (상위 7개 지역)')
                ax2.set_xlabel('지역')
                ax2.set_ylabel('인원 수')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 데이터프레임 표시
                st.subheader("지역별 선호도 데이터")
                st.dataframe(region_df)
        
        with tabs[1]:
            st.header("연령 분포")
            age_df = load_csv("output/eda_results/age_distribution.csv")
            if not age_df.empty:
                # 연령대별 합계 계산
                age_summary = age_df.groupby('age_group')['count'].sum().reset_index()
                
                # 연령대 매핑 (숫자에서 텍스트로)
                age_mapping = {2: '20대', 3: '30대', 4: '40대', 5: '50대 이상'}
                age_summary['age_group'] = age_summary['age_group'].map(age_mapping)
                
                # 카테고리별 시각화
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # 연령대별 합계 시각화
                ax1.bar(age_summary['age_group'], age_summary['count'])
                ax1.set_xlabel('연령대')
                ax1.set_ylabel('인원 수')
                ax1.set_title('전체 연령별 분포')
                
                # 카테고리별 연령 분포 시각화
                category_age = age_df.groupby(['category', 'age_group'])['count'].sum().reset_index()
                sns.barplot(x='age_group', y='count', hue='category', data=category_age, ax=ax2)
                ax2.set_xlabel('연령대')
                ax2.set_ylabel('인원 수')
                ax2.set_title('카테고리별 연령 분포')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 데이터프레임 표시
                st.subheader("연령 분포 데이터")
                st.dataframe(age_df)
        
        with tabs[2]:
            st.header("성별 분포")
            gender_df = load_csv("output/eda_results/gender_distribution.csv")
            if not gender_df.empty:
                # 성별 합계 계산
                gender_summary = gender_df.groupby('gender')['count'].sum().reset_index()
                
                # 카테고리별 시각화
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # 전체 성별 분포 파이 차트
                ax1.pie(gender_summary['count'], labels=gender_summary['gender'], autopct='%1.1f%%')
                ax1.set_title('전체 성별 분포')
                
                # 카테고리별 성별 분포 시각화
                category_gender = gender_df.groupby(['category', 'gender'])['count'].sum().reset_index()
                sns.barplot(x='gender', y='count', hue='category', data=category_gender, ax=ax2)
                ax2.set_xlabel('성별')
                ax2.set_ylabel('인원 수')
                ax2.set_title('카테고리별 성별 분포')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 데이터프레임 표시
                st.subheader("성별 분포 데이터")
                st.dataframe(gender_df)
        
        with tabs[3]:
            st.header("기타 통계")
            if file_exists("output/eda_results/eda_report.json"):
                eda_report = load_json("output/eda_results/eda_report.json")
                if eda_report:
                    st.json(eda_report)
    
    # 감성 분석
    elif menu == "감성 분석":
        st.title("감성 분석 결과")
        
        # 감성 분석 탭
        tabs = st.tabs(["감성 분포", "예측 결과"])
        
        with tabs[0]:
            st.header("감성 분포")
            sentiment_df = load_csv("output/eda_results/sentiment_distribution.csv")
            if not sentiment_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['lightcoral', 'lightgreen']
                ax.bar(sentiment_df['sentiment'], sentiment_df['count'], color=colors)
                ax.set_xlabel('감성')
                ax.set_ylabel('리뷰 수')
                ax.set_title('감성 분포')
                st.pyplot(fig)
                st.dataframe(sentiment_df)
        
        with tabs[1]:
            st.header("감성 예측 결과")
            predictions_df = load_csv("output/predictions.csv")
            if not predictions_df.empty:
                st.dataframe(predictions_df.head(20))
    
    # 토픽 모델링
    elif menu == "토픽 모델링":
        st.title("토픽 모델링 결과")
        
        # 토픽 모델링 탭
        tabs = st.tabs(["주제 키워드", "적정 토픽 수", "대표 문서", "토픽 분포", "워드클라우드"])
        
        with tabs[0]:
            st.header("주제별 핵심 키워드")
            topics_text = load_text("output/topic_modeling/topics_keywords.txt")
            if topics_text:
                st.text(topics_text)
        
        with tabs[1]:
            st.header("적정 토픽 수 결정")
            perplexity_img = load_image("output/topic_modeling/perplexity_score.png")
            if perplexity_img:
                st.image(perplexity_img, use_container_width=True)
        
        with tabs[2]:
            st.header("토픽별 대표 문서")
            docs_text = load_text("output/topic_modeling/representative_documents.txt")
            if docs_text:
                st.text(docs_text)
        
        with tabs[3]:
            st.header("토픽 분포")
            topic_dist_img = load_image("output/topic_modeling/topic_distribution.png")
            if topic_dist_img:
                st.image(topic_dist_img, use_container_width=True)
        
        with tabs[4]:
            st.header("토픽별 워드클라우드")
            col1, col2, col3 = st.columns(3)
            
            wordcloud_files = [
                "output/topic_modeling/wordcloud_topic_0.png",
                "output/topic_modeling/wordcloud_topic_1.png",
                "output/topic_modeling/wordcloud_topic_2.png",
                "output/topic_modeling/wordcloud_topic_3.png",
                "output/topic_modeling/wordcloud_topic_4.png"
            ]
            
            with col1:
                if file_exists(wordcloud_files[0]):
                    st.image(load_image(wordcloud_files[0]), caption="토픽 1 워드클라우드")
                if file_exists(wordcloud_files[3]):
                    st.image(load_image(wordcloud_files[3]), caption="토픽 4 워드클라우드")
            
            with col2:
                if file_exists(wordcloud_files[1]):
                    st.image(load_image(wordcloud_files[1]), caption="토픽 2 워드클라우드")
                if file_exists(wordcloud_files[4]):
                    st.image(load_image(wordcloud_files[4]), caption="토픽 5 워드클라우드")
            
            with col3:
                if file_exists(wordcloud_files[2]):
                    st.image(load_image(wordcloud_files[2]), caption="토픽 3 워드클라우드")
    
    # 키워드 네트워크
    elif menu == "키워드 네트워크":
        st.title("키워드 네트워크 분석")
        
        # 키워드 네트워크 탭
        tabs = st.tabs(["키워드 유사도", "키워드 네트워크", "테마별 분석"])
        
        with tabs[0]:
            st.header("키워드 유사도 히트맵")
            heatmap_img = load_image("output/keyword_network/keyword_similarity_heatmap.png")
            if heatmap_img:
                st.image(heatmap_img, use_container_width=True)
                
            keyword_relations = load_csv("output/keyword_network/top_keyword_pairs.csv")
            if not keyword_relations.empty:
                st.dataframe(keyword_relations)
        
        with tabs[1]:
            st.header("키워드 네트워크 시각화")
            network_img = load_image("output/keyword_network/keyword_network.png")
            if network_img:
                st.image(network_img, use_container_width=True)
        
        with tabs[2]:
            st.header("테마별 키워드 네트워크")
            # 테마별 네트워크
            theme_tabs = st.tabs(["어린이 관련 키워드", "안전 관련 키워드", "디자인 관련 키워드"])
            
            for i, theme in enumerate(["어린이", "안전", "디자인"]):
                with theme_tabs[i]:
                    caption = f"{theme} 관련 키워드 네트워크"
                    file_path = f"output/keyword_network/theme_{theme}_network.png"
                    theme_img = load_image(file_path)
                    if theme_img:
                        st.image(theme_img, caption=caption, use_container_width=True)
                    
                    # 관련 데이터 표시
                    st.subheader(f"{theme} 관련 키워드 상위 관계")
                    relation_path = f"output/keyword_network/theme_{theme}_relations.csv"
                    relations_df = load_csv(relation_path)
                    if not relations_df.empty:
                        st.dataframe(relations_df)
                    else:
                        st.warning(f"파일을 찾을 수 없습니다: {relation_path}")
    
    # 페르소나
    elif menu == "페르소나":
        st.title("고객 페르소나 분석")
        
        # 페르소나 탭
        tabs = st.tabs(["페르소나 프로필", "레이더 차트", "고객 여정"])
        
        with tabs[0]:
            st.header("페르소나 프로필")
            persona_text = load_text("output/persona/persona_descriptions.txt")
            if persona_text:
                sections = persona_text.split("===")
                for section in sections:
                    if section.strip():
                        st.markdown(section)
                        st.markdown("---")
                
                # 클러스터 세부 정보 표시
                cluster_df = load_csv("output/persona/cluster_details.csv")
                if not cluster_df.empty:
                    st.dataframe(cluster_df)
        
        with tabs[1]:
            st.header("페르소나 레이더 차트")
            radar_img = load_image("output/persona/persona_radar_charts.png")
            if radar_img:
                st.image(radar_img, use_container_width=True)
                
                # 클러스터 프로필 데이터
                profiles_df = load_csv("output/persona/cluster_profiles.csv")
                if not profiles_df.empty:
                    st.dataframe(profiles_df)
        
        with tabs[2]:
            st.header("고객 여정 타임라인")
            journey_img = load_image("output/persona/customer_journey_timeline.png")
            if journey_img:
                st.image(journey_img, use_container_width=True)
    
    # 마케팅 채널
    elif menu == "마케팅 채널":
        st.title("마케팅 채널 분석")
        
        # 마케팅 채널 탭
        tabs = st.tabs(["채널 효과성", "전환 퍼널", "채널 맵"])
        
        with tabs[0]:
            st.header("마케팅 채널 효과성")
            effectiveness_img = load_image("output/persona/marketing_channel_effectiveness.png")
            if effectiveness_img:
                st.image(effectiveness_img, use_container_width=True)
                
            channel_df = load_csv("output/persona/marketing_channel_effectiveness.csv")
            if not channel_df.empty:
                st.dataframe(channel_df)
        
        with tabs[1]:
            st.header("페르소나별 전환 퍼널")
            funnel_img = load_image("output/persona/conversion_funnel_by_persona.png")
            if funnel_img:
                st.image(funnel_img, use_container_width=True)
        
        with tabs[2]:
            st.header("마케팅 채널 맵")
            map_img = load_image("output/persona/marketing_channel_map.png")
            if map_img:
                st.image(map_img, use_container_width=True)
    
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