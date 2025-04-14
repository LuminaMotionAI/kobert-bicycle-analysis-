#!/usr/bin/env python
# -*- coding: utf-8 -*-

# lda_topic_modeling.py
# LDA 토픽 모델링을 이용한 리뷰 데이터 분석

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

# Gensim 라이브러리 임포트
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess

# pyLDAvis 시각화 라이브러리 임포트
import pyLDAvis
import pyLDAvis.gensim_models

# NLTK
import nltk
from nltk.corpus import stopwords

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
INPUT_DIR = "data"
OUTPUT_DIR = "output/topic_modeling"

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# KoNLPy 형태소 분석기 초기화
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
    processed_texts = []
    
    print("텍스트 전처리 및 명사 추출 중...")
    for i, text in enumerate(texts):
        if i % 1000 == 0 and i > 0:
            print(f"{i}/{len(texts)} 처리 완료")
            
        processed_text = preprocess_text(text)
        nouns = extract_nouns(processed_text)
        processed_texts.append(nouns)
    
    # 명사가 없는 문서 제거
    processed_texts = [doc for doc in processed_texts if doc]
    
    # Dictionary 생성
    id2word = corpora.Dictionary(processed_texts)
    
    # 극단적으로 희귀하거나 일반적인 단어 필터링
    id2word.filter_extremes(no_below=5, no_above=0.5)
    
    # Corpus 생성
    corpus = [id2word.doc2bow(doc) for doc in processed_texts]
    
    return corpus, id2word, processed_texts

def compute_coherence_values(dictionary, corpus, texts, start=2, limit=11, step=1):
    """다양한 토픽 수에 대한 coherence score 계산"""
    
    coherence_values = []
    model_list = []
    
    print("\n다양한 토픽 수에 대한 Coherence Score 계산:")
    
    for num_topics in range(start, limit, step):
        print(f"토픽 수 {num_topics} 모델 학습 중...")
        
        model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        model_list.append(model)
        
        # Coherence Model
        coherence_model = CoherenceModel(
            model=model, 
            texts=texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        
        coherence_score = coherence_model.get_coherence()
        coherence_values.append(coherence_score)
        
        print(f"토픽 수 {num_topics}의 Coherence Score: {coherence_score:.4f}")
    
    return model_list, coherence_values

def plot_coherence_values(coherence_values, start, limit, step):
    """Coherence Score 시각화"""
    
    x = range(start, limit, step)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, coherence_values, 'o-')
    plt.xlabel('토픽 수')
    plt.ylabel('Coherence Score')
    plt.title('토픽 수에 따른 Coherence Score 변화')
    plt.xticks(x)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/coherence_score.png")
    plt.close()
    
    # 최적의 토픽 수 찾기
    optimal_num_topics = x[np.argmax(coherence_values)]
    max_coherence = max(coherence_values)
    
    print(f"\n최적의 토픽 수: {optimal_num_topics} (Coherence Score: {max_coherence:.4f})")
    
    return optimal_num_topics

def print_topics(model, num_topics, n_words=10):
    """토픽별 주요 단어 출력"""
    
    print("\n토픽별 주요 단어:")
    for idx, topic in model.show_topics(num_topics=num_topics, num_words=n_words, formatted=False):
        print(f"\n토픽 #{idx+1}:")
        print(", ".join([word for word, prob in topic]))

def visualize_topics(model, corpus, dictionary):
    """pyLDAvis를 이용한 토픽 시각화"""
    
    print("\nLDA 시각화 생성 중...")
    
    # pyLDAvis 시각화
    vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
    
    # HTML 파일로 저장
    pyLDAvis.save_html(vis, f"{OUTPUT_DIR}/lda_visualization.html")
    
    print(f"시각화가 {OUTPUT_DIR}/lda_visualization.html에 저장되었습니다.")

def create_topic_wordclouds(model, num_topics):
    """토픽별 워드클라우드 생성"""
    
    print("\n토픽별 워드클라우드 생성 중...")
    
    # 각 토픽에 대한 워드클라우드 생성
    for topic_id in range(num_topics):
        # 토픽의 단어와 가중치
        topic_words = dict(model.show_topic(topic_id, 30))
        
        # 워드클라우드 생성
        wordcloud = WordCloud(
            font_path=font_path,
            width=800, 
            height=600,
            background_color='white',
            max_words=100
        ).generate_from_frequencies(topic_words)
        
        # 시각화
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'토픽 #{topic_id+1} 워드클라우드')
        plt.tight_layout()
        
        # 저장
        plt.savefig(f"{OUTPUT_DIR}/topic_{topic_id+1}_wordcloud.png")
        plt.close()
    
    print(f"워드클라우드가 {OUTPUT_DIR} 폴더에 저장되었습니다.")

def get_representative_documents(lda_model, corpus, texts, num_docs=3):
    """각 토픽을 가장 잘 나타내는 문서 추출"""
    
    print("\n토픽별 대표 문서 추출 중...")
    
    topic_rep_docs = []
    
    # 각 문서의 주요 토픽 확인
    doc_topics = [sorted(topics, key=lambda x: x[1], reverse=True)[0] 
                  for topics in lda_model.get_document_topics(corpus)]
    
    # 각 토픽별 대표 문서 찾기
    for topic_id in range(lda_model.num_topics):
        # 해당 토픽을 주요 토픽으로 가지는 문서 인덱스 찾기
        doc_indices = [i for i, (t_id, _) in enumerate(doc_topics) if t_id == topic_id]
        
        # 해당 토픽에 대한 확률이 높은 순으로 정렬
        doc_probs = [(i, doc_topics[i][1]) for i in doc_indices]
        doc_probs = sorted(doc_probs, key=lambda x: x[1], reverse=True)
        
        # 상위 n개 문서 선택
        top_docs = [texts[i] for i, _ in doc_probs[:num_docs] if i < len(texts)]
        
        # 결과 저장
        topic_rep_docs.append({
            'topic_id': topic_id,
            'top_docs': top_docs
        })
    
    # 결과 출력 및 저장
    with open(f"{OUTPUT_DIR}/representative_documents.txt", 'w', encoding='utf-8') as f:
        for topic in topic_rep_docs:
            topic_id = topic['topic_id']
            top_docs = topic['top_docs']
            
            f.write(f"\n토픽 #{topic_id+1} 대표 문서:\n")
            print(f"\n토픽 #{topic_id+1} 대표 문서:")
            
            for i, doc in enumerate(top_docs, 1):
                doc_text = " ".join(doc)
                f.write(f"{i}. {doc_text[:200]}...\n")
                print(f"{i}. {doc_text[:200]}...")
    
    print(f"\n대표 문서가 {OUTPUT_DIR}/representative_documents.txt에 저장되었습니다.")
    
    return topic_rep_docs

def analyze_topic_trends(lda_model, corpus, df):
    """리뷰 데이터의 토픽 트렌드 분석"""
    
    # 날짜 컬럼이 있는 경우에만 수행
    date_cols = [col for col in df.columns if '날짜' in col.lower() or 'date' in col.lower()]
    
    if not date_cols:
        print("\n날짜 관련 컬럼이 없어 토픽 트렌드 분석을 건너뜁니다.")
        return
    
    print(f"\n토픽 트렌드 분석 중 (컬럼: {date_cols[0]})...")
    
    date_col = date_cols[0]
    
    try:
        # 날짜 변환
        df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # 문서별 주요 토픽 정보
        doc_topics = []
        for i, doc_bow in enumerate(corpus):
            topic_probs = lda_model.get_document_topics(doc_bow)
            main_topic = sorted(topic_probs, key=lambda x: x[1], reverse=True)[0]
            doc_topics.append({
                'doc_id': i,
                'main_topic': main_topic[0],
                'topic_prob': main_topic[1]
            })
        
        # 데이터프레임으로 변환
        topic_df = pd.DataFrame(doc_topics)
        
        # 원본 데이터와 합치기
        if len(topic_df) == len(df):
            df_with_topics = df.copy()
            df_with_topics['main_topic'] = topic_df['main_topic']
            df_with_topics['topic_prob'] = topic_df['topic_prob']
            
            # 월별 집계
            df_with_topics['year_month'] = df_with_topics['date_parsed'].dt.strftime('%Y-%m')
            monthly_topics = df_with_topics.groupby(['year_month', 'main_topic']).size().unstack(fill_value=0)
            
            # 비율 계산
            monthly_topics_ratio = monthly_topics.div(monthly_topics.sum(axis=1), axis=0)
            
            # 결과 저장
            monthly_topics.to_csv(f"{OUTPUT_DIR}/monthly_topics.csv", encoding='utf-8')
            monthly_topics_ratio.to_csv(f"{OUTPUT_DIR}/monthly_topics_ratio.csv", encoding='utf-8')
            
            # 시각화: 절대값
            plt.figure(figsize=(14, 8))
            monthly_topics.plot(kind='line', marker='o')
            plt.title('월별 토픽 트렌드 (건수)')
            plt.xlabel('월')
            plt.ylabel('리뷰 수')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title='토픽')
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/topic_trend_absolute.png")
            plt.close()
            
            # 시각화: 비율
            plt.figure(figsize=(14, 8))
            monthly_topics_ratio.plot(kind='line', marker='o')
            plt.title('월별 토픽 트렌드 (비율)')
            plt.xlabel('월')
            plt.ylabel('비율')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title='토픽')
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/topic_trend_ratio.png")
            plt.close()
            
            print(f"토픽 트렌드 분석 결과가 {OUTPUT_DIR} 폴더에 저장되었습니다.")
        else:
            print("토픽 정보와 원본 데이터의 크기가 일치하지 않아 트렌드 분석을 건너뜁니다.")
            
    except Exception as e:
        print(f"토픽 트렌드 분석 중 오류 발생: {str(e)}")

def main():
    """메인 실행 함수"""
    print("===== LDA 토픽 모델링 분석 시작 =====")
    
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
        corpus, id2word, processed_texts = prepare_data_for_lda(texts)
        print(f"전처리 후 {len(processed_texts)}개의 문서가 분석 대상입니다.")
        
        # 최적의 토픽 수 찾기
        start, limit, step = 5, 11, 1  # 5~10개의 토픽 탐색
        model_list, coherence_values = compute_coherence_values(id2word, corpus, processed_texts, start, limit, step)
        
        # Coherence Score 시각화
        optimal_num_topics = plot_coherence_values(coherence_values, start, limit, step)
        
        # 최적의 모델 선택
        optimal_model_index = optimal_num_topics - start
        lda_model = model_list[optimal_model_index]
        
        # 토픽별 주요 단어 출력
        print_topics(lda_model, optimal_num_topics)
        
        # 토픽 시각화
        visualize_topics(lda_model, corpus, id2word)
        
        # 토픽별 워드클라우드 생성
        create_topic_wordclouds(lda_model, optimal_num_topics)
        
        # 토픽별 대표 문서 추출
        get_representative_documents(lda_model, corpus, processed_texts)
        
        # 토픽 트렌드 분석
        analyze_topic_trends(lda_model, corpus, df)
        
        # 최종 모델 저장
        lda_model.save(f"{OUTPUT_DIR}/lda_model_{optimal_num_topics}_topics")
        print(f"\n최종 LDA 모델이 {OUTPUT_DIR}/lda_model_{optimal_num_topics}_topics에 저장되었습니다.")
        
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
    
    print("\n===== LDA 토픽 모델링 분석 완료 =====")

if __name__ == "__main__":
    main() 