# bicycle_text_mining.py
# 자전거 데이터에 대한 키워드 마이닝 및 감성 분석

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import warnings
from matplotlib import font_manager, rc
import platform
import json
from collections import Counter
from wordcloud import WordCloud
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer

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
TEXT_DIR = "output"
RESULTS_DIR = "output/text_mining_results"

# 결과 디렉토리 생성
os.makedirs(RESULTS_DIR, exist_ok=True)

# Okt 형태소 분석기 초기화
okt = Okt()

# 불용어 사전 정의
STOPWORDS = [
    '이', '그', '저', '것', '수', '등', '및', '더', '를', '에', '의', '을', '은', '는', 
    '이다', '있다', '하다', '이다', '그것', '저것', '어떤', '무슨', '어느', '같은', '또한',
    '그리고', '한', '일', '이런', '저런', '그런', '어떻게', '왜', '어찌'
]

# 데이터 로드 함수
def load_sentiment_data():
    """감성 분석 결과와 텍스트 데이터 로드"""
    
    sentiment_data = {}
    
    # 감성 분석 결과 로드
    sentiment_file = f"{TEXT_DIR}/predictions.csv"
    if os.path.exists(sentiment_file):
        try:
            sentiment_data = pd.read_csv(sentiment_file)
            print(f"감성 분석 결과 로드 완료: {sentiment_file}")
        except Exception as e:
            print(f"파일 '{sentiment_file}' 로드 중 오류: {str(e)}")
    else:
        print(f"감성 분석 결과 파일이 존재하지 않습니다: {sentiment_file}")
        
    # NSMC 결과 로드
    nsmc_file = f"{TEXT_DIR}/predictions_nsmc.csv"
    if os.path.exists(nsmc_file):
        try:
            nsmc_data = pd.read_csv(nsmc_file)
            print(f"NSMC 감성 분석 결과 로드 완료: {nsmc_file}")
            sentiment_data['nsmc'] = nsmc_data
        except Exception as e:
            print(f"파일 '{nsmc_file}' 로드 중 오류: {str(e)}")
    
    return sentiment_data

# 텍스트 전처리 함수
def preprocess_text(text):
    """텍스트 전처리 함수"""
    if not isinstance(text, str):
        return ""
    
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 숫자 제거
    text = re.sub(r'\d+', ' ', text)
    
    # 소문자 변환 및 공백 정리
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text

# 명사 추출 함수
def extract_nouns(text):
    """텍스트에서 명사 추출"""
    try:
        nouns = okt.nouns(text)
        # 불용어와 한 글자 단어 제거
        nouns = [noun for noun in nouns if noun not in STOPWORDS and len(noun) > 1]
        return nouns
    except Exception as e:
        print(f"명사 추출 중 오류: {str(e)}")
        return []

# 단어 빈도 계산 함수
def calculate_word_frequency(texts):
    """텍스트 목록에서 단어 빈도 계산"""
    all_nouns = []
    
    for text in texts:
        preprocessed_text = preprocess_text(text)
        nouns = extract_nouns(preprocessed_text)
        all_nouns.extend(nouns)
    
    # 단어 빈도 계산
    word_counts = Counter(all_nouns)
    
    return word_counts

# 워드클라우드 생성 함수
def generate_wordcloud(word_counts, title, filename):
    """워드클라우드 생성 및 저장"""
    
    # 워드클라우드 생성
    wordcloud = WordCloud(
        font_path=font_path,
        width=800, 
        height=600,
        background_color='white',
        max_words=100
    ).generate_from_frequencies(word_counts)
    
    # 시각화
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    
    # 저장
    plt.savefig(f"{RESULTS_DIR}/{filename}.png")
    plt.close()
    
    print(f"워드클라우드 생성 완료: {filename}.png")

# 바차트 생성 함수
def generate_barchart(word_counts, title, filename, top_n=20):
    """상위 단어의 빈도수 바차트 생성 및 저장"""
    
    # 상위 N개 단어 추출
    top_words = word_counts.most_common(top_n)
    df = pd.DataFrame(top_words, columns=['word', 'count'])
    
    # 바차트 생성
    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='word', data=df)
    plt.title(title)
    plt.xlabel('빈도수')
    plt.ylabel('단어')
    plt.tight_layout()
    
    # 저장
    plt.savefig(f"{RESULTS_DIR}/{filename}.png")
    plt.close()
    
    # CSV 파일로 저장
    df.to_csv(f"{RESULTS_DIR}/{filename}.csv", index=False, encoding='utf-8')
    
    print(f"바차트 생성 완료: {filename}.png")

# TF-IDF 분석 함수
def analyze_tfidf(texts):
    """TF-IDF 분석으로 중요 키워드 추출"""
    
    # 텍스트 전처리 및 명사 추출
    preprocessed_texts = []
    for text in texts:
        preprocessed_text = preprocess_text(text)
        nouns = extract_nouns(preprocessed_text)
        preprocessed_texts.append(' '.join(nouns))
    
    # TF-IDF 계산
    tfidf_vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts)
    
    # 특성 이름 추출
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # 평균 TF-IDF 값 계산
    tfidf_mean = tfidf_matrix.mean(axis=0).A1
    
    # 단어와 TF-IDF 값 매핑
    tfidf_scores = {feature_names[i]: tfidf_mean[i] for i in range(len(feature_names))}
    
    # TF-IDF 값으로 정렬
    sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 데이터프레임으로 변환
    tfidf_df = pd.DataFrame(sorted_tfidf, columns=['word', 'tfidf_score'])
    
    return tfidf_df

# 감성별 키워드 분석 함수
def analyze_keywords_by_sentiment(df):
    """감성별 주요 키워드 분석"""
    
    if '감정분석결과' not in df.columns or 'text' not in df.columns:
        print("감성 또는 텍스트 컬럼이 없습니다.")
        return
    
    # 긍정/부정 리뷰 분리
    positive_texts = df[df['감정분석결과'] == '긍정']['text'].tolist()
    negative_texts = df[df['감정분석결과'] == '부정']['text'].tolist()
    
    print(f"긍정 리뷰 수: {len(positive_texts)}")
    print(f"부정 리뷰 수: {len(negative_texts)}")
    
    # 긍정 리뷰 키워드 분석
    positive_word_counts = calculate_word_frequency(positive_texts)
    generate_wordcloud(positive_word_counts, '긍정 리뷰 워드클라우드', 'positive_wordcloud')
    generate_barchart(positive_word_counts, '긍정 리뷰 상위 키워드', 'positive_keywords', top_n=20)
    
    # 부정 리뷰 키워드 분석
    negative_word_counts = calculate_word_frequency(negative_texts)
    generate_wordcloud(negative_word_counts, '부정 리뷰 워드클라우드', 'negative_wordcloud')
    generate_barchart(negative_word_counts, '부정 리뷰 상위 키워드', 'negative_keywords', top_n=20)
    
    # 전체 리뷰 키워드 분석
    all_texts = df['text'].tolist()
    all_word_counts = calculate_word_frequency(all_texts)
    generate_wordcloud(all_word_counts, '전체 리뷰 워드클라우드', 'all_wordcloud')
    generate_barchart(all_word_counts, '전체 리뷰 상위 키워드', 'all_keywords', top_n=30)
    
    # TF-IDF 분석
    tfidf_df = analyze_tfidf(all_texts)
    tfidf_df_top = tfidf_df.head(30)
    
    # TF-IDF 결과 시각화
    plt.figure(figsize=(12, 8))
    sns.barplot(x='tfidf_score', y='word', data=tfidf_df_top)
    plt.title('TF-IDF 상위 키워드')
    plt.xlabel('TF-IDF 점수')
    plt.ylabel('단어')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/tfidf_keywords.png")
    plt.close()
    
    # TF-IDF 결과 저장
    tfidf_df.to_csv(f"{RESULTS_DIR}/tfidf_scores.csv", index=False, encoding='utf-8')
    
    print("감성별 키워드 분석 완료")

# 시간별 감성 트렌드 분석 함수
def analyze_sentiment_trends(df):
    """시간별 감성 트렌드 분석"""
    
    if '감정분석결과' not in df.columns or 'date' not in df.columns:
        # date 컬럼이 없는 경우 확인
        print("date 컬럼이 없습니다. 날짜 관련 컬럼이 있는지 확인합니다.")
        date_columns = [col for col in df.columns if 'date' in col.lower() or '날짜' in col]
        
        if not date_columns:
            print("날짜 관련 컬럼이 없어 시간별 트렌드 분석을 건너뜁니다.")
            return
        
        date_column = date_columns[0]
        print(f"'{date_column}' 컬럼을 사용하여 시간별 트렌드를 분석합니다.")
    else:
        date_column = 'date'
    
    try:
        # 날짜 컬럼을 datetime 타입으로 변환
        df['date_parsed'] = pd.to_datetime(df[date_column], errors='coerce')
        
        # 결측치 확인 및 제거
        missing_dates = df['date_parsed'].isna().sum()
        if missing_dates > 0:
            print(f"날짜 파싱 중 {missing_dates}개의 결측치가 발생하여 제외합니다.")
            df = df.dropna(subset=['date_parsed'])
        
        # 월별 집계
        df['year_month'] = df['date_parsed'].dt.strftime('%Y-%m')
        monthly_sentiment = df.groupby(['year_month', '감정분석결과']).size().unstack(fill_value=0)
        
        # 비율 계산
        monthly_sentiment['total'] = monthly_sentiment.sum(axis=1)
        sentiment_ratio = monthly_sentiment.div(monthly_sentiment['total'], axis=0).drop('total', axis=1)
        
        # 시각화
        plt.figure(figsize=(14, 8))
        sentiment_ratio.plot(kind='line', marker='o')
        plt.title('월별 감성 분석 트렌드')
        plt.xlabel('월')
        plt.ylabel('비율')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='감성')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/sentiment_trend_ratio.png")
        plt.close()
        
        # 절대값 트렌드
        plt.figure(figsize=(14, 8))
        monthly_sentiment.drop('total', axis=1).plot(kind='line', marker='o')
        plt.title('월별 감성 분석 트렌드 (건수)')
        plt.xlabel('월')
        plt.ylabel('리뷰 수')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='감성')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/sentiment_trend_absolute.png")
        plt.close()
        
        # 결과 저장
        monthly_sentiment.to_csv(f"{RESULTS_DIR}/monthly_sentiment.csv", encoding='utf-8')
        sentiment_ratio.to_csv(f"{RESULTS_DIR}/monthly_sentiment_ratio.csv", encoding='utf-8')
        
        print("시간별 감성 트렌드 분석 완료")
        
    except Exception as e:
        print(f"시간별 트렌드 분석 중 오류 발생: {str(e)}")

# 감성과 키워드의 연관성 분석 함수
def analyze_sentiment_keyword_correlation(df):
    """감성과 키워드 간의 연관성 분석"""
    
    if '감정분석결과' not in df.columns or 'text' not in df.columns:
        print("감성 또는 텍스트 컬럼이 없습니다.")
        return
    
    # 전처리 및 명사 추출
    df['nouns'] = df['text'].apply(lambda x: ' '.join(extract_nouns(preprocess_text(x))))
    
    # 긍정/부정 리뷰별 TF-IDF 분석
    positive_df = df[df['감정분석결과'] == '긍정']
    negative_df = df[df['감정분석결과'] == '부정']
    
    # 긍정 리뷰 TF-IDF
    positive_tfidf = analyze_tfidf(positive_df['text'].tolist())
    positive_tfidf_top = positive_tfidf.head(20)
    
    # 부정 리뷰 TF-IDF
    negative_tfidf = analyze_tfidf(negative_df['text'].tolist())
    negative_tfidf_top = negative_tfidf.head(20)
    
    # 시각화: 긍정 리뷰 TF-IDF
    plt.figure(figsize=(12, 8))
    sns.barplot(x='tfidf_score', y='word', data=positive_tfidf_top)
    plt.title('긍정 리뷰 TF-IDF 상위 키워드')
    plt.xlabel('TF-IDF 점수')
    plt.ylabel('단어')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/positive_tfidf_keywords.png")
    plt.close()
    
    # 시각화: 부정 리뷰 TF-IDF
    plt.figure(figsize=(12, 8))
    sns.barplot(x='tfidf_score', y='word', data=negative_tfidf_top)
    plt.title('부정 리뷰 TF-IDF 상위 키워드')
    plt.xlabel('TF-IDF 점수')
    plt.ylabel('단어')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/negative_tfidf_keywords.png")
    plt.close()
    
    # 결과 저장
    positive_tfidf.to_csv(f"{RESULTS_DIR}/positive_tfidf_scores.csv", index=False, encoding='utf-8')
    negative_tfidf.to_csv(f"{RESULTS_DIR}/negative_tfidf_scores.csv", index=False, encoding='utf-8')
    
    print("감성과 키워드 연관성 분석 완료")

# 감성 분석 요약 보고서 생성 함수
def generate_sentiment_summary_report(df):
    """감성 분석 요약 보고서 생성"""
    
    if '감정분석결과' not in df.columns:
        print("감성 컬럼이 없습니다.")
        return
    
    report = {
        "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_reviews": len(df),
        "sentiment_distribution": df['감정분석결과'].value_counts().to_dict(),
        "sentiment_ratio": df['감정분석결과'].value_counts(normalize=True).mul(100).round(2).to_dict()
    }
    
    # 텍스트 데이터가 있는 경우 키워드 정보 추가
    if 'text' in df.columns:
        # 상위 키워드
        all_word_counts = calculate_word_frequency(df['text'].tolist())
        top_keywords = dict(all_word_counts.most_common(30))
        report["top_keywords"] = top_keywords
        
        # 감성별 상위 키워드
        positive_texts = df[df['감정분석결과'] == '긍정']['text'].tolist()
        negative_texts = df[df['감정분석결과'] == '부정']['text'].tolist()
        
        positive_word_counts = calculate_word_frequency(positive_texts)
        negative_word_counts = calculate_word_frequency(negative_texts)
        
        report["positive_top_keywords"] = dict(positive_word_counts.most_common(20))
        report["negative_top_keywords"] = dict(negative_word_counts.most_common(20))
    
    # 보고서 저장
    with open(f"{RESULTS_DIR}/sentiment_summary_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print("감성 분석 요약 보고서 생성 완료")

# 메인 실행 함수
def main():
    print("===== 자전거 데이터 키워드 마이닝 및 감성 분석 시작 =====")
    
    # 데이터 로드
    print("데이터 로드 중...")
    sentiment_data = load_sentiment_data()
    
    if not isinstance(sentiment_data, pd.DataFrame):
        print("감성 분석 데이터가 로드되지 않았습니다.")
        return
    
    # 텍스트 컬럼 확인
    if 'text' not in sentiment_data.columns:
        print("텍스트 컬럼이 없습니다.")
        return
    
    # 1. 감성별 키워드 분석
    print("\n1. 감성별 주요 키워드 분석 중...")
    analyze_keywords_by_sentiment(sentiment_data)
    
    # 2. 시간별 감성 트렌드 분석
    print("\n2. 시간별 감성 트렌드 분석 중...")
    analyze_sentiment_trends(sentiment_data)
    
    # 3. 감성과 키워드 연관성 분석
    print("\n3. 감성과 키워드 연관성 분석 중...")
    analyze_sentiment_keyword_correlation(sentiment_data)
    
    # 4. 감성 분석 요약 보고서 생성
    print("\n4. 감성 분석 요약 보고서 생성 중...")
    generate_sentiment_summary_report(sentiment_data)
    
    print("\n===== 자전거 데이터 키워드 마이닝 및 감성 분석 완료 =====")
    print(f"분석 결과는 '{RESULTS_DIR}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main() 