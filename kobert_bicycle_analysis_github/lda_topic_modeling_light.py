#!/usr/bin/env python
# -*- coding: utf-8 -*-

# lda_topic_modeling_light.py
# LDA 토픽 모델링을 이용한 리뷰 데이터 분석 (경량 버전)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import warnings
from matplotlib import font_manager, rc
import platform
from collections import Counter
from wordcloud import WordCloud
from konlpy.tag import Okt
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import traceback
import sys

# Debugging helper
def debug_print(message):
    print(f"DEBUG: {message}")
    sys.stdout.flush()  # Force flush to ensure output is displayed immediately

# 경고 무시
warnings.filterwarnings('ignore')

# 한글 폰트 설정
debug_print("Setting up fonts...")
if platform.system() == 'Windows':
    font_path = font_manager.findfont(font_manager.FontProperties(family='Malgun Gothic'))
else:
    font_path = "NanumGothic.ttf"
    
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

# 기본 경로 설정
INPUT_DIR = "data"
OUTPUT_DIR = "output/topic_modeling"

# 출력 폴더 생성
debug_print(f"Creating output directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# KoNLPy 형태소 분석기 초기화
debug_print("Initializing KoNLPy Okt tokenizer...")
okt = Okt()

# 불용어 정의 - 한국어 불용어
STOPWORDS = [
    '이', '그', '저', '것', '수', '등', '및', '더', '를', '에', '의', '을', '은', '는', 
    '이다', '있다', '하다', '이다', '그것', '저것', '어떤', '무슨', '어느', '같은', '또한',
    '그리고', '한', '일', '이런', '저런', '그런', '어떻게', '왜', '어찌', '하다', '있다',
    '되다', '못하다', '없다', '아니다', '않다', '이렇다', '그렇다', '저렇다', '그러나',
    '그래도', '하지만', '그리고', '또는', '혹은', '자전거', '제품', '사용', '구매', '배송',
    '정도', '때문', '너무', '정말', '좋다', '좋아요', '합니다', '있어요', '입니다', 'ㅎㅎ', 'ㅋㅋㅋ'
]

def load_review_data(file_path):
    """리뷰 데이터 로드 함수"""
    debug_print(f"Loading review data from: {file_path}")
    
    # 다양한 인코딩 시도
    encodings = ['utf-8', 'cp949', 'euc-kr']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"파일을 {encoding} 인코딩으로 읽었습니다.")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError("어떤 인코딩으로도 파일을 읽을 수 없습니다.")
    
    debug_print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_text(text):
    """텍스트 전처리 함수"""
    
    if not isinstance(text, str):
        return ""
    
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 숫자 제거
    text = re.sub(r'\d+', ' ', text)
    
    # 공백 정리
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_nouns(text):
    """명사 추출 함수"""
    
    try:
        nouns = okt.nouns(text)
        # 불용어와 한 글자 단어 제거
        nouns = [noun for noun in nouns if noun not in STOPWORDS and len(noun) > 1]
        return nouns
    except Exception as e:
        print(f"명사 추출 중 오류: {str(e)}")
        return []

def prepare_data_for_lda(texts):
    """LDA 모델링을 위한 데이터 준비"""
    
    # 전처리 및 명사 추출
    processed_docs = []
    
    print("텍스트 전처리 및 명사 추출 중...")
    for i, text in enumerate(texts):
        if i % 500 == 0:  # 더 자주 상태 표시
            debug_print(f"Processing text {i}/{len(texts)}")
            
        processed_text = preprocess_text(text)
        nouns = extract_nouns(processed_text)
        processed_docs.append(" ".join(nouns))
    
    # 단어 벡터화
    debug_print("Vectorizing documents...")
    vectorizer = CountVectorizer(
        max_df=0.5,       # 전체 문서의 50% 이상에서 등장하는 단어는 제외
        min_df=5,         # 최소 5개 이상의 문서에서 등장해야 함
        max_features=1000  # 최대 1000개의 특성만 사용
    )
    
    X = vectorizer.fit_transform(processed_docs)
    debug_print(f"Document-term matrix created: {X.shape[0]} documents, {X.shape[1]} features")
    
    # 특성 이름
    feature_names = vectorizer.get_feature_names_out()
    
    return X, feature_names, processed_docs

def find_optimal_topics(X, feature_names, processed_docs, start=5, limit=11, step=1):
    """다양한 LDA 모델을 실행하여 최적의 토픽 수 찾기"""
    
    perplexity_values = []
    lda_models = []
    
    print("\n다양한 토픽 수에 대한 Perplexity 계산:")
    
    for num_topics in range(start, limit, step):
        debug_print(f"Training LDA model with {num_topics} topics...")
        
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=10,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )
        
        lda.fit(X)
        
        perplexity = lda.perplexity(X)
        perplexity_values.append(perplexity)
        lda_models.append(lda)
        
        print(f"토픽 수 {num_topics}의 Perplexity: {perplexity:.2f}")
    
    # perplexity는 낮을수록 좋음
    return lda_models, perplexity_values

def plot_perplexity_values(perplexity_values, start, limit, step):
    """Perplexity 시각화"""
    
    debug_print("Plotting perplexity values...")
    x = range(start, limit, step)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, perplexity_values, 'o-')
    plt.xlabel('토픽 수')
    plt.ylabel('Perplexity')
    plt.title('토픽 수에 따른 Perplexity 변화 (낮을수록 좋음)')
    plt.xticks(x)
    plt.grid(True)
    plt.tight_layout()
    
    plt_file = f"{OUTPUT_DIR}/perplexity_score.png"
    debug_print(f"Saving perplexity plot to {plt_file}")
    plt.savefig(plt_file)
    plt.close()
    
    # 최적의 토픽 수 찾기 (perplexity가 낮을수록 좋음)
    optimal_idx = np.argmin(perplexity_values)
    optimal_num_topics = x[optimal_idx]
    min_perplexity = perplexity_values[optimal_idx]
    
    print(f"\n최적의 토픽 수: {optimal_num_topics} (Perplexity: {min_perplexity:.2f})")
    
    return optimal_num_topics

def print_topics(model, feature_names, n_top_words=10):
    """토픽별 주요 단어 출력"""
    
    debug_print("Extracting top keywords for each topic...")
    print("\n토픽별 주요 단어:")
    
    # 파일로도 저장
    topics_file = f"{OUTPUT_DIR}/topics_keywords.txt"
    debug_print(f"Saving topics to {topics_file}")
    
    with open(topics_file, "w", encoding="utf-8") as f:
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            
            topic_str = f"토픽 #{topic_idx+1}: {', '.join(top_words)}"
            print(f"\n{topic_str}")
            f.write(f"{topic_str}\n")
    
    return {idx: [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] 
            for idx, topic in enumerate(model.components_)}

def create_topic_wordclouds(model, feature_names, num_topics):
    """토픽별 워드클라우드 생성"""
    
    print("\n토픽별 워드클라우드 생성 중...")
    
    # 각 토픽에 대한 워드클라우드 생성
    for topic_idx, topic in enumerate(model.components_):
        debug_print(f"Creating wordcloud for topic {topic_idx+1}")
        
        # 단어와 가중치
        word_weights = {feature_names[i]: topic[i] for i in range(len(feature_names))}
        
        # 워드클라우드 생성
        wordcloud = WordCloud(
            font_path=font_path,
            width=800, 
            height=600,
            background_color='white',
            max_words=100
        ).generate_from_frequencies(word_weights)
        
        # 시각화
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'토픽 #{topic_idx+1} 워드클라우드')
        plt.tight_layout()
        
        # 저장
        wc_file = f"{OUTPUT_DIR}/topic_{topic_idx+1}_wordcloud.png"
        debug_print(f"Saving wordcloud to {wc_file}")
        plt.savefig(wc_file)
        plt.close()
    
    print(f"워드클라우드가 {OUTPUT_DIR} 폴더에 저장되었습니다.")

def get_representative_documents(lda_model, X, processed_docs, num_docs=3):
    """각 토픽을 가장 잘 나타내는 문서 추출"""
    
    debug_print("Extracting representative documents for each topic...")
    print("\n토픽별 대표 문서 추출 중...")
    
    # 문서-토픽 행렬
    doc_topic = lda_model.transform(X)
    
    topic_rep_docs = []
    
    # 각 토픽별 대표 문서 찾기
    for topic_idx in range(lda_model.n_components):
        # 해당 토픽에 대한 확률이 높은 문서 찾기
        doc_indices = doc_topic[:, topic_idx].argsort()[::-1][:num_docs]
        
        # 결과 저장
        top_docs = [processed_docs[i] for i in doc_indices if i < len(processed_docs)]
        
        # 결과 저장
        topic_rep_docs.append({
            'topic_id': topic_idx,
            'top_docs': top_docs
        })
    
    # 결과 출력 및 저장
    rep_docs_file = f"{OUTPUT_DIR}/representative_documents.txt"
    debug_print(f"Saving representative documents to {rep_docs_file}")
    
    with open(rep_docs_file, 'w', encoding='utf-8') as f:
        for topic in topic_rep_docs:
            topic_id = topic['topic_id']
            top_docs = topic['top_docs']
            
            f.write(f"\n토픽 #{topic_id+1} 대표 문서:\n")
            print(f"\n토픽 #{topic_id+1} 대표 문서:")
            
            for i, doc in enumerate(top_docs, 1):
                f.write(f"{i}. {doc[:200]}...\n")
                print(f"{i}. {doc[:200]}...")
    
    print(f"\n대표 문서가 {OUTPUT_DIR}/representative_documents.txt에 저장되었습니다.")
    
    return topic_rep_docs

def create_topic_distribution_viz(model, X):
    """토픽 분포 시각화"""
    
    debug_print("Creating topic distribution visualization...")
    
    # 문서-토픽 행렬
    doc_topic = model.transform(X)
    
    # 각 토픽의 평균 비율 계산
    topic_means = doc_topic.mean(axis=0)
    
    # 토픽 인덱스
    topics = [f"토픽 {i+1}" for i in range(model.n_components)]
    
    # 바 차트 그리기
    plt.figure(figsize=(12, 6))
    plt.bar(topics, topic_means)
    plt.xlabel('토픽')
    plt.ylabel('평균 토픽 비율')
    plt.title('전체 문서의 토픽 분포')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    dist_file = f"{OUTPUT_DIR}/topic_distribution.png"
    debug_print(f"Saving topic distribution to {dist_file}")
    plt.savefig(dist_file)
    plt.close()
    
    print(f"토픽 분포 시각화가 {OUTPUT_DIR}/topic_distribution.png에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    print("===== LDA 토픽 모델링 분석 시작 (경량 버전) =====")
    
    # 데이터 로드
    try:
        reviews_file = f"{INPUT_DIR}/review_data.csv"
        df = load_review_data(reviews_file)
        
        if '리뷰내용' not in df.columns:
            raise ValueError(f"'리뷰내용' 컬럼이 필요합니다. 사용 가능한 컬럼: {df.columns.tolist()}")
            
        # 텍스트 데이터 추출
        texts = df['리뷰내용'].dropna().astype(str).tolist()
        print(f"{len(texts)}개의 리뷰 데이터를 분석합니다.")
        
        # LDA 모델링을 위한 데이터 준비
        X, feature_names, processed_docs = prepare_data_for_lda(texts)
        print(f"전처리 후 {X.shape[0]}개의 문서, {X.shape[1]}개의 특성이 분석 대상입니다.")
        
        # 최적의 토픽 수 찾기 (5~10개 토픽)
        start, limit, step = 5, 11, 1
        lda_models, perplexity_values = find_optimal_topics(X, feature_names, processed_docs, start, limit, step)
        
        # Perplexity 시각화 및 최적 토픽 수 선택
        optimal_num_topics = plot_perplexity_values(perplexity_values, start, limit, step)
        
        # 최적의 모델 선택
        optimal_model_index = optimal_num_topics - start
        lda_model = lda_models[optimal_model_index]
        
        # 토픽별 주요 단어 출력
        topic_keywords = print_topics(lda_model, feature_names)
        
        # 토픽별 워드클라우드 생성
        create_topic_wordclouds(lda_model, feature_names, optimal_num_topics)
        
        # 토픽별 대표 문서 추출
        get_representative_documents(lda_model, X, processed_docs)
        
        # 토픽 분포 시각화
        create_topic_distribution_viz(lda_model, X)
        
        debug_print("All processing completed successfully!")
        
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        debug_print(f"Exception details: {traceback.format_exc()}")
    
    print("\n===== LDA 토픽 모델링 분석 완료 =====")

if __name__ == "__main__":
    debug_print("Starting script execution...")
    main() 