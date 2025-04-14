#!/usr/bin/env python
# -*- coding: utf-8 -*-

# persona_analysis.py
# 자전거 관심 고객 페르소나 분석

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from matplotlib import font_manager, rc
import platform
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.cm as cm

# 경고 무시
warnings.filterwarnings('ignore')

# 한글 폰트 설정
if platform.system() == 'Windows':
    font_path = font_manager.findfont(font_manager.FontProperties(family='Malgun Gothic'))
else:
    font_path = "NanumGothic.ttf"
    
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

# 기본 경로 설정
DATA_DIR = "output/eda_results"
OUTPUT_DIR = "output/persona"

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_datasets():
    """기존 EDA 결과 데이터 로드"""
    print("기존 분석 데이터 로드 중...")
    
    # 지역 선호도
    region_df = pd.read_csv(f"{DATA_DIR}/regional_preference.csv")
    
    # 연령 분포
    age_df = pd.read_csv(f"{DATA_DIR}/age_distribution.csv")
    
    # 성별 분포
    gender_df = pd.read_csv(f"{DATA_DIR}/gender_distribution.csv")
    
    # 감성 분포
    sentiment_df = pd.read_csv(f"{DATA_DIR}/sentiment_distribution.csv")
    
    print("데이터 로드 완료")
    return region_df, age_df, gender_df, sentiment_df

def preprocess_data(region_df, age_df, gender_df):
    """데이터 전처리 및 통합"""
    
    print("데이터 전처리 및 통합 중...")
    
    # 연령 그룹 매핑
    age_mapping = {
        2: '20대',
        3: '30대',
        4: '40대',
        5: '50대 이상'
    }
    
    # 연령 데이터 피벗
    age_pivot = age_df.copy()
    age_pivot['age_group'] = age_pivot['age_group'].map(age_mapping)
    age_pivot = age_pivot.pivot_table(
        index=['category', 'file'], 
        columns='age_group', 
        values='count', 
        aggfunc='sum'
    ).reset_index()
    
    # 결측값 처리
    age_pivot = age_pivot.fillna(0)
    
    # 성별 데이터 피벗
    gender_pivot = gender_df.pivot_table(
        index=['category', 'file'], 
        columns='gender', 
        values='count', 
        aggfunc='sum'
    ).reset_index()
    
    # 지역 데이터 상위 5개 지역으로 집계
    top_regions = ['서울', '경기', '인천', '부산', '대구']
    
    def process_region(df):
        region_result = []
        for cat_file in df[['category', 'file']].drop_duplicates().values:
            cat, file = cat_file
            temp_df = df[(df['category'] == cat) & (df['file'] == file)]
            
            # 상위 5개 지역만 선택
            region_counts = {region: 0 for region in top_regions}
            other_count = 0
            
            for _, row in temp_df.iterrows():
                if row['region'] in top_regions:
                    region_counts[row['region']] = row['count']
                else:
                    other_count += row['count']
            
            # 결과 저장
            result_row = {'category': cat, 'file': file, '기타 지역': other_count}
            result_row.update(region_counts)
            region_result.append(result_row)
        
        return pd.DataFrame(region_result)
    
    region_pivot = process_region(region_df)
    
    # 데이터 통합 (category, file 기준으로 조인)
    merged_df = pd.merge(age_pivot, gender_pivot, on=['category', 'file'], how='outer')
    merged_df = pd.merge(merged_df, region_pivot, on=['category', 'file'], how='outer')
    
    # 결측값 0으로 채우기
    merged_df = merged_df.fillna(0)
    
    print("데이터 전처리 완료")
    return merged_df

def create_persona_clusters(data_df, n_clusters=4):
    """K-means 군집화로 페르소나 도출"""
    
    print(f"군집 분석을 통한 {n_clusters}개 페르소나 도출 중...")
    
    # 분석에 사용할 특성 선택
    features = ['20대', '30대', '40대', '50대 이상', 'M', 'F', 
                '서울', '경기', '인천', '부산', '대구', '기타 지역']
    
    # 데이터 준비
    X = data_df[features].values
    
    # 특성 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means 군집화
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 군집 결과 원본 데이터에 추가
    data_df['cluster'] = clusters
    
    # 군집 중심점 역변환
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # 각 군집의 특성 분석
    cluster_df = pd.DataFrame(cluster_centers, columns=features)
    cluster_df.index.name = 'cluster'
    cluster_df.index = [f'Cluster {i+1}' for i in range(n_clusters)]
    
    # 가장 특징적인 속성 찾기
    cluster_profiles = []
    
    for i in range(n_clusters):
        center = cluster_centers[i]
        profile = {
            'cluster': f'Cluster {i+1}',
            'size': np.sum(clusters == i),
            'dominant_age': features[np.argmax(center[:4])],
            'gender_ratio': f"{center[4]/(center[4]+center[5])*100:.1f}% 남성",
            'top_region': features[6:][np.argmax(center[6:])],
            'category': data_df[data_df['cluster'] == i]['category'].value_counts().index[0]
        }
        cluster_profiles.append(profile)
    
    print("군집 분석 완료")
    return data_df, pd.DataFrame(cluster_profiles), cluster_df

def save_cluster_info(cluster_profiles, cluster_details):
    """군집 정보 저장"""
    
    # 군집 프로필 저장
    cluster_profiles.to_csv(f"{OUTPUT_DIR}/cluster_profiles.csv", index=False, encoding='utf-8')
    
    # 군집 상세정보 저장
    cluster_details.to_csv(f"{OUTPUT_DIR}/cluster_details.csv", encoding='utf-8')
    
    print(f"군집 정보가 {OUTPUT_DIR} 폴더에 저장되었습니다.")

def visualize_personas(cluster_df):
    """페르소나 시각화 - 레이더 차트"""
    
    print("페르소나 시각화 (레이더 차트) 생성 중...")
    
    # 스케일링 함수
    def scale_features(cluster_df):
        result = cluster_df.copy()
        for col in result.columns:
            max_val = result[col].max()
            if max_val > 0:
                result[col] = result[col] / max_val
        return result
    
    # 데이터 스케일링
    scaled_df = scale_features(cluster_df)
    
    # 레이더 차트 생성
    categories = list(scaled_df.columns)
    n_categories = len(categories)
    
    # 각 군집별 차트 생성
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), subplot_kw=dict(polar=True))
    axs = axs.flatten()
    
    # 각도 계산
    angles = np.linspace(0, 2*np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]  # 닫힌 다각형을 위해 처음으로 돌아감
    
    # 색상 맵
    colors = plt.cm.tab10(np.linspace(0, 1, len(scaled_df)))
    
    for i, (idx, row) in enumerate(scaled_df.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1]  # 닫힌 다각형을 위해 처음으로 돌아감
        
        ax = axs[i]
        ax.plot(angles, values, color=colors[i], linewidth=2, label=idx)
        ax.fill(angles, values, color=colors[i], alpha=0.25)
        
        # 축 레이블 추가
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        
        # 제목 설정
        ax.set_title(f'페르소나 {i+1}', size=14, color=colors[i], y=1.1)
        
        # 그리드 설정
        ax.grid(True, linestyle='-', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/persona_radar_charts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"레이더 차트가 {OUTPUT_DIR}/persona_radar_charts.png에 저장되었습니다.")

def create_persona_descriptions(cluster_profiles):
    """페르소나 설명 생성"""
    
    print("페르소나 스토리 생성 중...")
    
    # 페르소나 템플릿
    persona_templates = [
        {
            "name": "김서준",
            "age": 35,
            "gender": "남성",
            "job": "IT 회사 직장인",
            "family": "아내와 5살 아들",
            "scenario": "출퇴근과 주말 가족 나들이를 위한 자전거 구매 고려 중",
            "goal": "일상 통근 + 가족 레저용 자전거",
            "pain_points": ["비싼 가격", "보관 공간 제약", "안전 문제"],
            "channels": ["온라인 리뷰", "유튜브", "자전거 매장 방문"]
        },
        {
            "name": "이지현",
            "age": 28,
            "gender": "여성",
            "job": "마케팅 전문가",
            "family": "1인 가구",
            "scenario": "건강 관리와 취미 활동을 위한 여성용 자전거 탐색 중",
            "goal": "운동 + 주말 라이딩",
            "pain_points": ["혼자 조립 어려움", "여성용 디자인 부족", "무게"],
            "channels": ["인스타그램", "블로그", "여성 커뮤니티"]
        },
        {
            "name": "박민우",
            "age": 45,
            "gender": "남성",
            "job": "중소기업 대표",
            "family": "아내와 중학생 자녀 2명",
            "scenario": "가족 모두가 함께 자전거 여행을 즐기길 원함",
            "goal": "가족 여행용 + 건강관리",
            "pain_points": ["품질 걱정", "가성비", "A/S"],
            "channels": ["지인 추천", "네이버 카페", "오프라인 매장"]
        },
        {
            "name": "최현우",
            "age": 24,
            "gender": "남성",
            "job": "대학생",
            "family": "자취생",
            "scenario": "교내 이동과 주변 탐색을 위한 저가형 자전거 필요",
            "goal": "통학 + 가성비",
            "pain_points": ["가격", "도난 위험", "보관"],
            "channels": ["유튜브", "대학생 커뮤니티", "중고거래 앱"]
        }
    ]
    
    # 군집 특성에 맞게 페르소나 매칭
    personas = []
    
    for i, profile in cluster_profiles.iterrows():
        persona = persona_templates[i].copy()
        
        # 군집 특성 기반 페르소나 정보 수정
        if '20대' in profile['dominant_age']:
            persona['age'] = 25
        elif '30대' in profile['dominant_age']:
            persona['age'] = 35
        elif '40대' in profile['dominant_age']:
            persona['age'] = 45
        else:
            persona['age'] = 55
            
        if '남성' not in profile['gender_ratio']:
            persona['gender'] = '여성'
            
        # 지역 정보 추가
        persona['region'] = profile['top_region']
        
        # 카테고리 기반 특성 추가
        if '구매' in profile['category']:
            persona['interest'] = '자전거 구매에 적극적'
        else:
            persona['interest'] = '자전거 여행/관광에 관심'
            
        # 군집 크기 정보 추가
        persona['segment_size'] = profile['size']
        
        personas.append(persona)
    
    # 페르소나 정보 저장
    with open(f"{OUTPUT_DIR}/persona_descriptions.txt", "w", encoding="utf-8") as f:
        for i, persona in enumerate(personas):
            f.write(f"■ 페르소나 {i+1}: {persona['name']} ({persona['age']}세, {persona['gender']})\n")
            f.write(f"- 직업: {persona['job']}\n")
            f.write(f"- 가족구성: {persona['family']}\n")
            f.write(f"- 주요 지역: {persona['region']}\n")
            f.write(f"- 시나리오: {persona['scenario']}\n")
            f.write(f"- 구매 목적: {persona['goal']}\n")
            f.write(f"- 주요 고민: {', '.join(persona['pain_points'])}\n")
            f.write(f"- 주요 접점: {', '.join(persona['channels'])}\n")
            f.write(f"- 특징: {persona['interest']}\n")
            f.write(f"- 세그먼트 크기: {persona['segment_size']}\n\n")
    
    print(f"페르소나 설명이 {OUTPUT_DIR}/persona_descriptions.txt에 저장되었습니다.")

def create_timeline_visualization():
    """페르소나별 고객 여정 타임라인 시각화"""
    
    print("고객 여정 타임라인 시각화 생성 중...")
    
    # 고객 여정 단계
    journey_stages = ['인지', '정보 탐색', '고려', '구매 결정', '사용', '추천']
    
    # 페르소나별 여정 특성
    persona_journeys = [
        # 페르소나 1 (30대 남성, 직장인)
        {
            '인지': {'channels': ['YouTube', '인스타그램'], 'pain_points': [], 'score': 4},
            '정보 탐색': {'channels': ['온라인 리뷰', '블로그'], 'pain_points': ['정보 과부하'], 'score': 3},
            '고려': {'channels': ['매장 방문', '가격 비교'], 'pain_points': ['가격 부담'], 'score': 3},
            '구매 결정': {'channels': ['공식 웹사이트', '오프라인 매장'], 'pain_points': ['배송 지연'], 'score': 4},
            '사용': {'channels': [], 'pain_points': ['수리 어려움'], 'score': 3},
            '추천': {'channels': ['SNS', '지인'], 'pain_points': [], 'score': 4}
        },
        # 페르소나 2 (20대 여성, 마케팅 전문가)
        {
            '인지': {'channels': ['인스타그램', '페이스북'], 'pain_points': [], 'score': 5},
            '정보 탐색': {'channels': ['인플루언서', '여성 커뮤니티'], 'pain_points': ['여성용 정보 부족'], 'score': 2},
            '고려': {'channels': ['시착', '리뷰'], 'pain_points': ['디자인 제한'], 'score': 2},
            '구매 결정': {'channels': ['온라인몰'], 'pain_points': ['조립 불편'], 'score': 3},
            '사용': {'channels': [], 'pain_points': ['무게'], 'score': 3},
            '추천': {'channels': ['인스타그램'], 'pain_points': [], 'score': 4}
        },
        # 페르소나 3 (40대 남성, 중소기업 대표)
        {
            '인지': {'channels': ['TV', '주변 추천'], 'pain_points': [], 'score': 3},
            '정보 탐색': {'channels': ['네이버 카페', '전문 매장'], 'pain_points': ['신뢰성 의문'], 'score': 3},
            '고려': {'channels': ['가족 의견', '전문가 상담'], 'pain_points': ['가족 적합성'], 'score': 4},
            '구매 결정': {'channels': ['대리점'], 'pain_points': ['A/S 걱정'], 'score': 4},
            '사용': {'channels': [], 'pain_points': ['공간 차지'], 'score': 4},
            '추천': {'channels': ['지인'], 'pain_points': [], 'score': 5}
        },
        # 페르소나 4 (20대 남성, 대학생)
        {
            '인지': {'channels': ['YouTube', '커뮤니티'], 'pain_points': [], 'score': 5},
            '정보 탐색': {'channels': ['중고거래 앱', '대학 커뮤니티'], 'pain_points': ['예산 제약'], 'score': 4},
            '고려': {'channels': ['가격 비교', '중고 매물'], 'pain_points': ['품질 우려'], 'score': 3},
            '구매 결정': {'channels': ['중고거래', '할인행사'], 'pain_points': ['배송/수령'], 'score': 3},
            '사용': {'channels': [], 'pain_points': ['도난 위험'], 'score': 2},
            '추천': {'channels': ['친구'], 'pain_points': [], 'score': 3}
        }
    ]
    
    # 타임라인 시각화
    fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    persona_names = ["김서준 (35세, 남성, 직장인)", 
                     "이지현 (28세, 여성, 마케팅 전문가)",
                     "박민우 (45세, 남성, 중소기업 대표)",
                     "최현우 (24세, 남성, 대학생)"]
    
    for i, (persona, journey) in enumerate(zip(persona_names, persona_journeys)):
        ax = axs[i]
        
        # 만족도 점수 플롯
        scores = [journey[stage]['score'] for stage in journey_stages]
        ax.plot(journey_stages, scores, marker='o', markersize=10, linewidth=2, color=f'C{i}')
        
        # 터치포인트와 페인포인트 표시
        for j, stage in enumerate(journey_stages):
            # 터치포인트
            channels = journey[stage]['channels']
            if channels:
                channel_text = '\n'.join(channels)
                ax.annotate(channel_text, (j, scores[j] + 0.2), 
                            ha='center', va='bottom', fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", fc='lightgreen', alpha=0.5))
            
            # 페인포인트
            pain_points = journey[stage]['pain_points']
            if pain_points:
                pain_text = '\n'.join(pain_points)
                ax.annotate(pain_text, (j, scores[j] - 0.2),
                            ha='center', va='top', fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", fc='lightpink', alpha=0.5))
        
        # 축 설정
        ax.set_ylim(0, 6)
        ax.set_yticks(range(1, 6))
        ax.set_ylabel('만족도')
        ax.set_title(f'페르소나 {i+1}: {persona}')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 배경색 설정
        for j in range(len(journey_stages)):
            ax.axvspan(j-0.5, j+0.5, alpha=0.1, color=f'C{j}')
    
    # x축 레이블 설정
    plt.xlabel('고객 여정 단계')
    plt.xticks(range(len(journey_stages)), journey_stages)
    
    # 범례 추가
    green_patch = mpatches.Patch(color='lightgreen', alpha=0.5, label='주요 접점')
    pink_patch = mpatches.Patch(color='lightpink', alpha=0.5, label='페인 포인트')
    plt.figlegend(handles=[green_patch, pink_patch], loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/customer_journey_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"고객 여정 타임라인이 {OUTPUT_DIR}/customer_journey_timeline.png에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    print("===== 자전거 관심 고객 페르소나 분석 시작 =====")
    
    # 데이터 로드
    region_df, age_df, gender_df, sentiment_df = load_datasets()
    
    # 데이터 전처리
    data_df = preprocess_data(region_df, age_df, gender_df)
    
    # 군집화로 페르소나 도출
    data_with_clusters, cluster_profiles, cluster_details = create_persona_clusters(data_df)
    
    # 군집 정보 저장
    save_cluster_info(cluster_profiles, cluster_details)
    
    # 페르소나 시각화
    visualize_personas(cluster_details)
    
    # 페르소나 스토리 생성
    create_persona_descriptions(cluster_profiles)
    
    # 고객 여정 타임라인 생성
    create_timeline_visualization()
    
    print("\n===== 페르소나 분석 완료 =====")
    print(f"모든 결과물은 {OUTPUT_DIR} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main() 