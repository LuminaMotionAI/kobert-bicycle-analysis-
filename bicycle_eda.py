# bicycle_eda.py
# 자전거 데이터에 대한 탐색적 데이터 분석(EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from matplotlib import font_manager, rc
import platform
import json
from collections import Counter

# 한글 폰트 설정
if platform.system() == 'Windows':
    font_path = font_manager.findfont(font_manager.FontProperties(family='Malgun Gothic'))
else:
    font_path = "NanumGothic.ttf"
    
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
warnings.filterwarnings('ignore')

# 파일 경로 설정
NUMERICAL_DIR = "output/numerical"
TEXT_DIR = "output"
RESULTS_DIR = "output/eda_results"

# 결과 디렉토리 생성
os.makedirs(RESULTS_DIR, exist_ok=True)

# 데이터 카테고리 정의
NUMERICAL_CATEGORIES = [
    "인구_통계", 
    "자전거_인프라", 
    "관심_구매", 
    "관광_여행"
]

# 데이터 로드 함수
def load_numerical_data():
    """전처리된 수치 데이터 로드"""
    data_dict = {}
    
    for category in NUMERICAL_CATEGORIES:
        category_dir = f"{NUMERICAL_DIR}/{category}"
        if not os.path.exists(category_dir):
            continue
            
        # 카테고리 요약 정보 로드
        summary_file = f"{category_dir}/category_summary.csv"
        if os.path.exists(summary_file):
            summary_df = pd.read_csv(summary_file)
            
            # 각 파일별 데이터 로드
            file_data = {}
            for _, row in summary_df.iterrows():
                file_name = row['file_name']
                base_name = os.path.splitext(file_name)[0]
                processed_file = f"{category_dir}/{base_name}_processed.csv"
                
                if os.path.exists(processed_file):
                    try:
                        df = pd.read_csv(processed_file)
                        file_data[file_name] = df
                    except Exception as e:
                        print(f"파일 '{processed_file}' 로드 중 오류: {str(e)}")
            
            data_dict[category] = file_data
    
    return data_dict

def load_text_data():
    """텍스트 분석 결과 로드"""
    text_data = {}
    
    # 감성 분석 결과 로드
    sentiment_file = f"{TEXT_DIR}/predictions.csv"
    if os.path.exists(sentiment_file):
        try:
            text_data["sentiment"] = pd.read_csv(sentiment_file)
        except Exception as e:
            print(f"파일 '{sentiment_file}' 로드 중 오류: {str(e)}")
    
    # NSMC 감성 분석 결과 로드
    nsmc_file = f"{TEXT_DIR}/predictions_nsmc.csv"
    if os.path.exists(nsmc_file):
        try:
            text_data["nsmc"] = pd.read_csv(nsmc_file)
        except Exception as e:
            print(f"파일 '{nsmc_file}' 로드 중 오류: {str(e)}")
    
    return text_data

# 데이터 분석 함수들
def analyze_age_distribution(data_dict):
    """연령대별 분석"""
    
    age_dfs = []
    
    # 관심_구매 및 관광_여행 카테고리에서 연령대 관련 데이터 추출
    for category in ["관심_구매", "관광_여행"]:
        if category not in data_dict:
            continue
            
        for file_name, df in data_dict[category].items():
            if 'age_group_cd' in df.columns:
                # 연령대별 카운트
                age_counts = df['age_group_cd'].value_counts().reset_index()
                age_counts.columns = ['age_group', 'count']
                age_counts['category'] = category
                age_counts['file'] = file_name
                age_dfs.append(age_counts)
    
    if not age_dfs:
        print("연령대 관련 데이터가 없습니다.")
        return
        
    # 데이터 결합
    age_df = pd.concat(age_dfs, ignore_index=True)
    
    # 시각화
    plt.figure(figsize=(14, 8))
    sns.barplot(x='age_group', y='count', hue='category', data=age_df)
    plt.title('카테고리별 연령대 분포')
    plt.xlabel('연령대')
    plt.ylabel('빈도')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/age_distribution.png")
    plt.close()
    
    # 결과 저장
    age_df.to_csv(f"{RESULTS_DIR}/age_distribution.csv", index=False)
    print("연령대별 분석 완료")

def analyze_gender_distribution(data_dict):
    """성별 분석"""
    
    gender_dfs = []
    
    # 관심_구매 및 관광_여행 카테고리에서 성별 관련 데이터 추출
    for category in ["관심_구매", "관광_여행"]:
        if category not in data_dict:
            continue
            
        for file_name, df in data_dict[category].items():
            if 'auser_sxdst_cd' in df.columns:
                # 성별 카운트
                gender_counts = df['auser_sxdst_cd'].value_counts().reset_index()
                gender_counts.columns = ['gender', 'count']
                gender_counts['category'] = category
                gender_counts['file'] = file_name
                gender_dfs.append(gender_counts)
    
    if not gender_dfs:
        print("성별 관련 데이터가 없습니다.")
        return
        
    # 데이터 결합
    gender_df = pd.concat(gender_dfs, ignore_index=True)
    
    # 시각화
    plt.figure(figsize=(12, 6))
    sns.barplot(x='gender', y='count', hue='category', data=gender_df)
    plt.title('카테고리별 성별 분포')
    plt.xlabel('성별')
    plt.ylabel('빈도')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/gender_distribution.png")
    plt.close()
    
    # 결과 저장
    gender_df.to_csv(f"{RESULTS_DIR}/gender_distribution.csv", index=False)
    print("성별 분석 완료")

def analyze_regional_preference(data_dict):
    """지역별 선호도 분석"""
    
    region_dfs = []
    
    # 관심_구매 및 관광_여행 카테고리에서 지역 관련 데이터 추출
    for category in ["관심_구매", "관광_여행"]:
        if category not in data_dict:
            continue
            
        for file_name, df in data_dict[category].items():
            if 'auser_ara_ctprv_nm' in df.columns:
                # 지역별 카운트
                region_counts = df['auser_ara_ctprv_nm'].value_counts().reset_index()
                region_counts.columns = ['region', 'count']
                region_counts['category'] = category
                region_counts['file'] = file_name
                region_dfs.append(region_counts)
    
    if not region_dfs:
        print("지역 관련 데이터가 없습니다.")
        return
        
    # 데이터 결합
    region_df = pd.concat(region_dfs, ignore_index=True)
    
    # 상위 10개 지역만 추출
    top_regions = region_df.groupby('region')['count'].sum().nlargest(10).index
    region_df_top = region_df[region_df['region'].isin(top_regions)]
    
    # 시각화
    plt.figure(figsize=(14, 8))
    sns.barplot(x='region', y='count', hue='category', data=region_df_top)
    plt.title('카테고리별 지역 분포 (상위 10개)')
    plt.xlabel('지역')
    plt.ylabel('빈도')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/regional_preference.png")
    plt.close()
    
    # 결과 저장
    region_df.to_csv(f"{RESULTS_DIR}/regional_preference.csv", index=False)
    print("지역별 선호도 분석 완료")

def analyze_bicycle_infrastructure(data_dict):
    """자전거 인프라 분석"""
    
    if "자전거_인프라" not in data_dict:
        print("자전거 인프라 데이터가 없습니다.")
        return
    
    infra_data = data_dict["자전거_인프라"]
    
    # 자전거 도로 데이터 분석
    if "Bicycle_자전거길_도로_데이터.csv" in infra_data:
        road_df = infra_data["Bicycle_자전거길_도로_데이터.csv"]
        
        # 시도별 자전거 도로 수 분석
        if '시도명' in road_df.columns:
            road_by_sido = road_df['시도명'].value_counts().reset_index()
            road_by_sido.columns = ['sido', 'count']
            
            # 시각화
            plt.figure(figsize=(12, 6))
            sns.barplot(x='sido', y='count', data=road_by_sido)
            plt.title('시도별 자전거 도로 수')
            plt.xlabel('시도')
            plt.ylabel('자전거 도로 수')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/bicycle_road_by_sido.png")
            plt.close()
            
            # 결과 저장
            road_by_sido.to_csv(f"{RESULTS_DIR}/bicycle_road_by_sido.csv", index=False)
        
        # 자전거 도로 종류별 분석
        if '자전거도로종류' in road_df.columns:
            road_by_type = road_df['자전거도로종류'].value_counts().reset_index()
            road_by_type.columns = ['type', 'count']
            
            # 시각화
            plt.figure(figsize=(10, 6))
            sns.barplot(x='type', y='count', data=road_by_type)
            plt.title('자전거 도로 종류별 분포')
            plt.xlabel('도로 종류')
            plt.ylabel('수')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/bicycle_road_by_type.png")
            plt.close()
            
            # 결과 저장
            road_by_type.to_csv(f"{RESULTS_DIR}/bicycle_road_by_type.csv", index=False)
    
    # 공공자전거 수요인구 분석
    if "Bicycle_공공자전거_수요인구_정보_데이터.csv" in infra_data:
        demand_df = infra_data["Bicycle_공공자전거_수요인구_정보_데이터.csv"]
        
        # 연령대별 공공자전거 수요 분석
        if 'agrde_cd' in demand_df.columns and 'age_use_rt' in demand_df.columns:
            # 연령대별 평균 이용률
            age_usage = demand_df.groupby('agrde_cd')['age_use_rt'].mean().reset_index()
            
            # 시각화
            plt.figure(figsize=(10, 6))
            sns.barplot(x='agrde_cd', y='age_use_rt', data=age_usage)
            plt.title('연령대별 공공자전거 평균 이용률')
            plt.xlabel('연령대')
            plt.ylabel('평균 이용률')
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/public_bicycle_usage_by_age.png")
            plt.close()
            
            # 결과 저장
            age_usage.to_csv(f"{RESULTS_DIR}/public_bicycle_usage_by_age.csv", index=False)
    
    print("자전거 인프라 분석 완료")

def analyze_sentiment_distribution(text_data):
    """감성 분석 결과 분석"""
    
    if "sentiment" not in text_data:
        print("감성 분석 데이터가 없습니다.")
        return
    
    sentiment_df = text_data["sentiment"]
    
    if '감정분석결과' in sentiment_df.columns:
        # 감성 분포 분석
        sentiment_counts = sentiment_df['감정분석결과'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        
        # 시각화
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='sentiment', y='count', data=sentiment_counts)
        plt.title('리뷰 감성 분포')
        plt.xlabel('감성')
        plt.ylabel('수')
        
        # 막대 위에 값 표시
        for i, v in enumerate(sentiment_counts['count']):
            ax.text(i, v + 5, str(v), ha='center')
        
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/sentiment_distribution.png")
        plt.close()
        
        # 원형 차트
        plt.figure(figsize=(8, 8))
        plt.pie(sentiment_counts['count'], labels=sentiment_counts['sentiment'], autopct='%1.1f%%')
        plt.title('리뷰 감성 분포')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/sentiment_distribution_pie.png")
        plt.close()
        
        # 결과 저장
        sentiment_counts.to_csv(f"{RESULTS_DIR}/sentiment_distribution.csv", index=False)
    
    print("감성 분석 결과 분석 완료")

def analyze_keyword_correlation():
    """키워드 분석 및 상관관계"""
    # konlpy_keywords.png 파일 분석 결과를 텍스트로 정리
    
    # 상위 키워드 목록 수동 입력 (출력 결과에서 확인한 상위 20개)
    top_keywords = [
        ("조립", 1937),
        ("아이", 1195),
        ("바퀴", 507),
        ("타고", 452),
        ("가격", 440),
        ("안장", 437),
        ("디자인", 405),
        ("생각", 393),
        ("조금", 349),
        ("접이식", 327),
        ("아주", 325),
        ("색상", 319),
        ("선물", 301),
        ("바구니", 283),
        ("가성", 263),
        ("핸들", 222),
        ("학년", 221),
        ("보조", 220),
        ("포장", 209),
        ("마음", 204)
    ]
    
    # 키워드 데이터프레임 생성
    keyword_df = pd.DataFrame(top_keywords, columns=['keyword', 'frequency'])
    
    # 시각화
    plt.figure(figsize=(12, 8))
    sns.barplot(x='keyword', y='frequency', data=keyword_df)
    plt.title('자전거 리뷰 상위 키워드 빈도')
    plt.xlabel('키워드')
    plt.ylabel('빈도')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/top_keywords.png")
    plt.close()
    
    # 결과 저장
    keyword_df.to_csv(f"{RESULTS_DIR}/top_keywords.csv", index=False)
    print("키워드 분석 완료")

def generate_eda_report(numerical_data, text_data):
    """EDA 분석 보고서 생성"""
    
    report = {
        "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "numerical_categories": {},
        "text_analysis": {}
    }
    
    # 수치 데이터 요약
    for category, data in numerical_data.items():
        category_summary = []
        for file_name, df in data.items():
            file_summary = {
                "file_name": file_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_list": df.columns.tolist(),
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": df.select_dtypes(exclude=['number']).columns.tolist()
            }
            category_summary.append(file_summary)
        report["numerical_categories"][category] = category_summary
    
    # 텍스트 데이터 요약
    for data_type, df in text_data.items():
        text_summary = {
            "rows": len(df),
            "columns": df.columns.tolist()
        }
        
        if data_type == "sentiment" and '감정분석결과' in df.columns:
            sentiment_counts = df['감정분석결과'].value_counts().to_dict()
            text_summary["sentiment_distribution"] = sentiment_counts
        
        report["text_analysis"][data_type] = text_summary
    
    # 보고서 저장
    with open(f"{RESULTS_DIR}/eda_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print("EDA 분석 보고서 생성 완료")

# 메인 실행 함수
def main():
    print("===== 자전거 데이터 탐색적 분석(EDA) 시작 =====")
    
    # 데이터 로드
    print("데이터 로드 중...")
    numerical_data = load_numerical_data()
    text_data = load_text_data()
    
    # 데이터 분석
    print("\n데이터 분석 중...")
    analyze_age_distribution(numerical_data)
    analyze_gender_distribution(numerical_data)
    analyze_regional_preference(numerical_data)
    analyze_bicycle_infrastructure(numerical_data)
    analyze_sentiment_distribution(text_data)
    analyze_keyword_correlation()
    
    # 보고서 생성
    print("\nEDA 보고서 생성 중...")
    generate_eda_report(numerical_data, text_data)
    
    print("\n===== 자전거 데이터 탐색적 분석(EDA) 완료 =====")
    print(f"분석 결과는 '{RESULTS_DIR}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main() 