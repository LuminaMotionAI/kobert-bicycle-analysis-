# bicycle_numerical_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from matplotlib import font_manager, rc
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    font_path = font_manager.findfont(font_manager.FontProperties(family='Malgun Gothic'))
else:
    font_path = "NanumGothic.ttf"
    
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 기본 경로 설정
INPUT_DIR = "Bicycle/수치 분석"
OUTPUT_DIR = "output/numerical"

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 데이터 카테고리 정의
DATA_CATEGORIES = {
    "인구_통계": ["Bicycle_이동수단_보유_현황_데이터.csv", 
                "Bicycle_국민생활체육조사_체육활동_이동수단_데이터.xlsx",
                "Bicycle_국민생활체육조사_주요참여_체육활동_데이터.xlsx",
                "Bicycle_국민생활체육조사_가입희망_체육동호회_데이터.xlsx"],
    "자전거_인프라": ["Bicycle_자전거길_도로_데이터.csv",
                    "Bicycle_공공자전거_수요인구_정보_데이터.csv"],
    "관심_구매": ["Bicycle_자전거_관심인구_용품정보_데이터.csv",
               "Bicycle_자전거_관심인구_구매정보_데이터.csv"],
    "관광_여행": ["Bicycle_자전거_관심인구_숙박정보_데이터.csv",
               "Bicycle_자전거_관심인구_선호관광지_데이터.csv",
               "Bicycle_자전거_관심인구_검색정보_데이터.csv"]
}

# 결측치 처리 함수
def handle_missing_values(df):
    """결측치 처리 함수"""
    # 결측치 비율 계산
    missing_ratio = df.isnull().mean().round(4) * 100
    print(f"결측치 비율(%): \n{missing_ratio[missing_ratio > 0]}")
    
    # 결측치가 50% 이상인 컬럼 제거
    cols_to_drop = missing_ratio[missing_ratio > 50].index.tolist()
    if cols_to_drop:
        print(f"제거할 컬럼: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # 수치형 컬럼 결측치 평균으로 대체
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    
    # 범주형 컬럼 결측치 최빈값으로 대체
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    return df

# 이상치 처리 함수
def handle_outliers(df):
    """이상치 처리 함수 (IQR 방법 사용)"""
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        # 이상치 탐지 (IQR 방법)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 이상치를 경계값으로 대체
        df[col] = df[col].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
    
    return df

# 데이터 요약 함수
def summarize_data(df, file_name):
    """데이터 요약 통계 계산 및 저장"""
    # 기본 요약 통계
    summary = df.describe(include='all').T
    summary['missing_count'] = df.isnull().sum()
    summary['missing_ratio'] = df.isnull().mean().round(4) * 100
    
    # 요약 통계 저장
    summary_file = f"{OUTPUT_DIR}/{os.path.splitext(file_name)[0]}_summary.csv"
    summary.to_csv(summary_file)
    print(f"요약 통계 저장 완료: {summary_file}")
    
    return summary

# 데이터 시각화 함수
def visualize_data(df, file_name):
    """기본적인 데이터 시각화"""
    file_base = os.path.splitext(file_name)[0]
    
    # 1. 숫자형 컬럼 분포 시각화
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        plt.figure(figsize=(15, 5 * ((len(numeric_cols) + 1) // 2)))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot((len(numeric_cols) + 1) // 2, 2, i)
            sns.histplot(df[col], kde=True)
            plt.title(f'{col} 분포')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{file_base}_numeric_dist.png")
        plt.close()
    
    # 2. 범주형 컬럼 분포 시각화
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    if len(categorical_cols) > 0 and len(categorical_cols) <= 10:  # 범주형 변수가 많으면 시각화 생략
        plt.figure(figsize=(15, 5 * ((len(categorical_cols) + 1) // 2)))
        for i, col in enumerate(categorical_cols, 1):
            if df[col].nunique() <= 20:  # 고유값이 너무 많으면 시각화 생략
                plt.subplot((len(categorical_cols) + 1) // 2, 2, i)
                df[col].value_counts().plot(kind='bar')
                plt.title(f'{col} 분포')
                plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{file_base}_categorical_dist.png")
        plt.close()
    
    # 3. 상관관계 분석 (숫자형 변수가 2개 이상인 경우)
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation = df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('상관관계 분석')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{file_base}_correlation.png")
        plt.close()

# 메인 실행 함수
def process_file(file_path, file_name):
    """파일 전처리 및 분석 실행 함수"""
    print(f"===== 처리 중: {file_name} =====")
    
    try:
        # 파일 확장자에 따라 적절한 방법으로 로딩
        if file_name.endswith('.csv'):
            # 다양한 인코딩 시도
            encodings = ['utf-8', 'cp949', 'euc-kr']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"인코딩 '{encoding}'으로 파일 로드 성공")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"오류: 파일 '{file_name}'을 로드할 수 없습니다. 인코딩 문제.")
                return None
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            print(f"오류: 지원되지 않는 파일 형식입니다: {file_name}")
            return None
        
        # 데이터 기본 정보 출력
        print(f"원본 데이터 크기: {df.shape}")
        print(f"컬럼: {df.columns.tolist()}")
        
        # 결측치 처리
        df = handle_missing_values(df)
        
        # 데이터 타입 최적화
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    print(f"컬럼 '{col}'을 날짜 타입으로 변환했습니다.")
                except:
                    pass
        
        # 이상치 처리
        df = handle_outliers(df)
        
        # 데이터 요약
        summary = summarize_data(df, file_name)
        
        # 데이터 시각화
        visualize_data(df, file_name)
        
        # 처리된 데이터 저장
        output_path = f"{OUTPUT_DIR}/{os.path.splitext(file_name)[0]}_processed.csv"
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"처리된 데이터 저장 완료: {output_path}")
        
        return df
    
    except Exception as e:
        print(f"파일 '{file_name}' 처리 중 오류 발생: {str(e)}")
        return None

# 모든 파일 처리 실행
def process_all_files():
    """모든 데이터 파일 처리"""
    processed_data = {}
    
    for category, files in DATA_CATEGORIES.items():
        print(f"\n***** 카테고리: {category} *****")
        category_dir = f"{OUTPUT_DIR}/{category}"
        os.makedirs(category_dir, exist_ok=True)
        
        category_data = {}
        for file_name in files:
            file_path = f"{INPUT_DIR}/{file_name}"
            if os.path.exists(file_path):
                df = process_file(file_path, file_name)
                if df is not None:
                    # 카테고리별 폴더에 복사본 저장
                    output_path = f"{category_dir}/{os.path.splitext(file_name)[0]}_processed.csv"
                    df.to_csv(output_path, index=False, encoding='utf-8')
                    category_data[file_name] = df
            else:
                print(f"경고: 파일이 존재하지 않습니다: {file_path}")
        
        processed_data[category] = category_data
    
    return processed_data

# 카테고리별 통합 분석
def category_analysis(processed_data):
    """카테고리별 통합 분석"""
    for category, data_dict in processed_data.items():
        if not data_dict:
            continue
            
        print(f"\n***** 카테고리 통합 분석: {category} *****")
        
        # 카테고리별 요약 정보 저장
        category_summary = pd.DataFrame({
            'file_name': [],
            'rows': [],
            'columns': [],
            'missing_ratio': [],
            'numeric_cols': [],
            'categorical_cols': []
        })
        
        for file_name, df in data_dict.items():
            summary_row = {
                'file_name': file_name,
                'rows': df.shape[0],
                'columns': df.shape[1],
                'missing_ratio': df.isnull().mean().mean() * 100,
                'numeric_cols': len(df.select_dtypes(include=['number']).columns),
                'categorical_cols': len(df.select_dtypes(exclude=['number']).columns)
            }
            category_summary = pd.concat([category_summary, pd.DataFrame([summary_row])], ignore_index=True)
        
        # 카테고리 요약 저장
        category_summary.to_csv(f"{OUTPUT_DIR}/{category}/category_summary.csv", index=False, encoding='utf-8')
        print(f"카테고리 요약 저장 완료: {OUTPUT_DIR}/{category}/category_summary.csv")

# 메인 실행
if __name__ == "__main__":
    print("===== 자전거 수치 데이터 분석 시작 =====")
    processed_data = process_all_files()
    category_analysis(processed_data)
    print("===== 자전거 수치 데이터 분석 완료 =====") 