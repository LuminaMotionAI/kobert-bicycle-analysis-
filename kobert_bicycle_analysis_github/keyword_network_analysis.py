#!/usr/bin/env python
# -*- coding: utf-8 -*-

# keyword_network_analysis.py
# 키워드 상관관계 및 네트워크 분석

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import warnings
from matplotlib import font_manager, rc
import platform
from collections import Counter
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import seaborn as sns
from itertools import combinations
import sys

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
OUTPUT_DIR = "output/keyword_network"

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
    print(f"리뷰 데이터 로드 중: {file_path}")
    
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
    
    print(f"데이터 로드 완료: {df.shape[0]}행, {df.shape[1]}열")
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

def create_document_keyword_matrix(texts):
    """문서-키워드 행렬 생성"""
    
    print("텍스트 전처리 및 명사 추출 중...")
    
    # 각 문서에서 명사 추출
    doc_nouns = []
    for i, text in enumerate(texts):
        if i % 1000 == 0 and i > 0:
            print(f"{i}/{len(texts)} 처리 완료")
            
        processed_text = preprocess_text(text)
        nouns = extract_nouns(processed_text)
        doc_nouns.append(nouns)
    
    # 모든 명사 수집 및 카운팅
    all_nouns = []
    for nouns in doc_nouns:
        all_nouns.extend(nouns)
    
    noun_counter = Counter(all_nouns)
    
    # 상위 키워드만 선택 (최소 5번 이상 등장)
    top_keywords = [keyword for keyword, count in noun_counter.items() if count >= 5]
    print(f"분석 대상 키워드: {len(top_keywords)}개")
    
    # 문서-키워드 행렬 생성
    doc_keyword_matrix = np.zeros((len(texts), len(top_keywords)))
    
    for i, nouns in enumerate(doc_nouns):
        for noun in nouns:
            if noun in top_keywords:
                j = top_keywords.index(noun)
                doc_keyword_matrix[i, j] += 1
    
    return doc_keyword_matrix, top_keywords

def create_keyword_cooccurrence_matrix(doc_keyword_matrix, keywords):
    """키워드 동시 출현 행렬 생성"""
    
    print("키워드 동시 출현 행렬 생성 중...")
    
    # 키워드 간 동시 출현 계산
    cooccurrence_matrix = np.zeros((len(keywords), len(keywords)))
    
    for doc_idx in range(doc_keyword_matrix.shape[0]):
        doc_keywords = doc_keyword_matrix[doc_idx]
        # 현재 문서에 있는 키워드 인덱스 찾기
        present_keywords = np.where(doc_keywords > 0)[0]
        
        # 모든 키워드 쌍에 대해 동시 출현 카운트 증가
        for i, j in combinations(present_keywords, 2):
            cooccurrence_matrix[i, j] += 1
            cooccurrence_matrix[j, i] += 1
    
    # 대각선 값 설정 (자기 자신과의 동시 출현은 해당 키워드의 총 출현 횟수)
    for i in range(len(keywords)):
        cooccurrence_matrix[i, i] = np.sum(doc_keyword_matrix[:, i])
    
    return cooccurrence_matrix

def calculate_cosine_similarity(matrix):
    """코사인 유사도 계산"""
    
    print("키워드 간 코사인 유사도 계산 중...")
    
    # 각 키워드 벡터의 코사인 유사도 계산
    similarity_matrix = cosine_similarity(matrix.T)
    
    return similarity_matrix

def create_similarity_heatmap(similarity_matrix, keywords):
    """유사도 히트맵 생성"""
    
    print("유사도 히트맵 생성 중...")
    
    # 상위 키워드만 선택 (가독성을 위해)
    top_n = min(30, len(keywords))
    
    # 키워드 선택 기준: 동시 출현 합계가 가장 높은 키워드
    keyword_importance = np.sum(similarity_matrix, axis=0)
    top_indices = np.argsort(keyword_importance)[-top_n:]
    
    # 선택된 키워드와 유사도 행렬
    selected_keywords = [keywords[idx] for idx in top_indices]
    selected_matrix = similarity_matrix[top_indices][:, top_indices]
    
    # 히트맵 생성
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        selected_matrix, 
        annot=True, 
        xticklabels=selected_keywords, 
        yticklabels=selected_keywords,
        cmap="YlGnBu", 
        fmt=".2f"
    )
    plt.title("키워드 간 코사인 유사도 (상위 키워드)")
    plt.tight_layout()
    
    # 저장
    plt_file = f"{OUTPUT_DIR}/keyword_similarity_heatmap.png"
    plt.savefig(plt_file)
    plt.close()
    
    print(f"히트맵 저장 완료: {plt_file}")
    
    return top_indices

def create_keyword_network(similarity_matrix, keywords, top_indices=None):
    """키워드 네트워크 생성"""
    
    print("키워드 네트워크 시각화 중...")
    
    # 분석 대상 키워드 선택
    if top_indices is None:
        # 상위 키워드만 선택 (가독성을 위해)
        top_n = min(50, len(keywords))
        keyword_importance = np.sum(similarity_matrix, axis=0)
        top_indices = np.argsort(keyword_importance)[-top_n:]
    
    selected_keywords = [keywords[idx] for idx in top_indices]
    selected_matrix = similarity_matrix[top_indices][:, top_indices]
    
    # 네트워크 그래프 생성
    G = nx.Graph()
    
    # 노드 추가
    for i, keyword in enumerate(selected_keywords):
        # 노드 중요도 = 해당 키워드의 모든 유사도 합계
        importance = np.sum(selected_matrix[i])
        G.add_node(keyword, weight=importance)
    
    # 에지 추가 (임계값 이상의 유사도만)
    threshold = 0.3  # 임계값 설정
    
    for i in range(len(selected_keywords)):
        for j in range(i+1, len(selected_keywords)):
            similarity = selected_matrix[i, j]
            if similarity > threshold:
                G.add_edge(selected_keywords[i], selected_keywords[j], weight=similarity)
    
    # 고립된 노드 제거
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    # 노드 크기 및 색상 설정
    node_sizes = [G.nodes[node]['weight'] * 500 for node in G.nodes()]
    node_colors = [G.nodes[node]['weight'] for node in G.nodes()]
    
    # 레이아웃 설정
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # 네트워크 시각화
    plt.figure(figsize=(16, 12))
    
    # 엣지 그리기
    edges = nx.draw_networkx_edges(
        G, pos, 
        alpha=0.7,
        width=[G[u][v]['weight'] * 3 for u, v in G.edges()],
        edge_color='lightgray'
    )
    
    # 노드 그리기
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        cmap=plt.cm.YlOrRd
    )
    
    # 레이블 그리기
    nx.draw_networkx_labels(
        G, pos,
        font_family=font_name,
        font_size=12,
        font_weight='bold'
    )
    
    plt.title("키워드 네트워크 분석")
    plt.colorbar(nodes, label="키워드 중요도")
    plt.axis('off')
    plt.tight_layout()
    
    # 저장
    plt_file = f"{OUTPUT_DIR}/keyword_network.png"
    plt.savefig(plt_file)
    plt.close()
    
    print(f"네트워크 시각화 저장 완료: {plt_file}")
    
    # 주요 연관 키워드 쌍 출력
    print("\n주요 키워드 연관 관계:")
    edge_weights = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
    edge_weights.sort(key=lambda x: x[2], reverse=True)
    
    # 상위 20개 연관 관계 저장
    with open(f"{OUTPUT_DIR}/top_keyword_relations.txt", "w", encoding="utf-8") as f:
        f.write("키워드1\t키워드2\t유사도\n")
        for u, v, w in edge_weights[:20]:
            line = f"{u}\t{v}\t{w:.3f}\n"
            f.write(line)
            print(f"- {u} ↔ {v}: {w:.3f}")
    
    # CSV 형식으로도 저장
    with open(f"{OUTPUT_DIR}/top_keyword_pairs.csv", "w", encoding="utf-8") as f:
        f.write("keyword1,keyword2,similarity\n")
        for u, v, w in edge_weights[:20]:
            line = f"{u},{v},{w:.3f}\n"
            f.write(line)
    
    print(f"키워드 관계 데이터 저장 완료: {OUTPUT_DIR}/top_keyword_pairs.csv")

def find_specific_theme_keywords(keywords, theme_list):
    """특정 테마 관련 키워드 찾기"""
    
    theme_keywords = {}
    for theme in theme_list:
        # 테마와 완전히 일치하거나 포함하는 키워드 찾기
        related_keywords = [kw for kw in keywords if theme in kw or kw == theme]
        theme_keywords[theme] = related_keywords
    
    return theme_keywords

def create_theme_network(similarity_matrix, keywords, theme_keywords, theme_list):
    """테마별 키워드 네트워크 생성"""
    
    print("테마별 키워드 네트워크 생성 중...")
    
    # 테마 키워드 인덱스 찾기
    theme_indices = {}
    for theme, related_kws in theme_keywords.items():
        indices = [keywords.index(kw) for kw in related_kws if kw in keywords]
        if indices:
            theme_indices[theme] = indices
    
    # 각 테마에 대해 분석 대상 키워드 선택
    for theme, indices in theme_indices.items():
        if not indices:
            continue
            
        print(f"\n테마 '{theme}' 관련 키워드 네트워크 분석:")
        
        # 유사도 기준 상위 연관 키워드 추가
        expanded_indices = set(indices)
        for idx in indices:
            # 해당 키워드와 유사도가 높은 상위 10개 키워드 찾기
            similar_indices = np.argsort(similarity_matrix[idx])[-11:]  # 자기 자신 포함
            expanded_indices.update(similar_indices)
        
        expanded_indices = list(expanded_indices)
        
        # 확장된 키워드 목록
        expanded_keywords = [keywords[idx] for idx in expanded_indices]
        print(f"- 분석 대상 키워드: {len(expanded_keywords)}개")
        
        # 네트워크 생성
        G = nx.Graph()
        
        # 노드 추가
        for i, idx in enumerate(expanded_indices):
            keyword = keywords[idx]
            importance = np.sum(similarity_matrix[idx])
            # 테마 직접 관련 키워드는 중요도 가중치 부여
            if idx in indices:
                importance *= 1.5
            G.add_node(keyword, weight=importance, is_theme=(idx in indices))
        
        # 에지 추가
        threshold = 0.25  # 임계값 설정
        for i, idx1 in enumerate(expanded_indices):
            for j, idx2 in enumerate(expanded_indices[i+1:], i+1):
                similarity = similarity_matrix[idx1, idx2]
                if similarity > threshold:
                    G.add_edge(keywords[idx1], keywords[idx2], weight=similarity)
        
        # 고립된 노드 제거
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        if len(G.nodes()) < 5:
            print(f"- 충분한 연결이 없어 테마 '{theme}' 네트워크를 생성하지 않습니다.")
            continue
        
        # 노드 크기 및 색상 설정
        node_sizes = []
        node_colors = []
        
        for node in G.nodes():
            size = G.nodes[node]['weight'] * 500
            if G.nodes[node]['is_theme']:
                size *= 1.5  # 테마 키워드는 더 크게
                node_colors.append('red')
            else:
                node_colors.append('skyblue')
            node_sizes.append(size)
        
        # 레이아웃 설정
        pos = nx.spring_layout(G, k=0.4, iterations=50, seed=42)
        
        # 네트워크 시각화
        plt.figure(figsize=(14, 10))
        
        # 엣지 그리기
        edges = nx.draw_networkx_edges(
            G, pos, 
            alpha=0.7,
            width=[G[u][v]['weight'] * 2.5 for u, v in G.edges()],
            edge_color='lightgray'
        )
        
        # 노드 그리기
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8
        )
        
        # 레이블 그리기
        nx.draw_networkx_labels(
            G, pos,
            font_family=font_name,
            font_size=12,
            font_weight='bold'
        )
        
        plt.title(f"'{theme}' 관련 키워드 네트워크")
        plt.axis('off')
        plt.tight_layout()
        
        # 저장
        plt_file = f"{OUTPUT_DIR}/theme_{theme}_network.png"
        plt.savefig(plt_file)
        plt.close()
        
        print(f"- 네트워크 시각화 저장 완료: {plt_file}")
        
        # 주요 연관 키워드 쌍 출력
        edge_weights = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
        edge_weights.sort(key=lambda x: x[2], reverse=True)
        
        # 상위 10개 연관 관계 저장
        with open(f"{OUTPUT_DIR}/theme_{theme}_relations.txt", "w", encoding="utf-8") as f:
            f.write(f"'{theme}' 관련 키워드 연관 관계\n")
            f.write("키워드1\t키워드2\t유사도\n")
            for u, v, w in edge_weights[:10]:
                line = f"{u}\t{v}\t{w:.3f}\n"
                f.write(line)
                print(f"- {u} ↔ {v}: {w:.3f}")
        
        # CSV 형식으로도 저장
        with open(f"{OUTPUT_DIR}/theme_{theme}_relations.csv", "w", encoding="utf-8") as f:
            f.write("keyword1,keyword2,similarity\n")
            for u, v, w in edge_weights[:10]:
                line = f"{u},{v},{w:.3f}\n"
                f.write(line)
        
        print(f"- 테마 '{theme}' 관계 CSV 파일 저장 완료")

def main():
    """메인 실행 함수"""
    print("===== 키워드 상관관계 및 네트워크 분석 시작 =====")
    
    # 데이터 로드
    try:
        reviews_file = f"{INPUT_DIR}/review_data.csv"
        df = load_review_data(reviews_file)
        
        if '리뷰내용' not in df.columns:
            raise ValueError(f"'리뷰내용' 컬럼이 필요합니다. 사용 가능한 컬럼: {df.columns.tolist()}")
            
        # 텍스트 데이터 추출
        texts = df['리뷰내용'].dropna().astype(str).tolist()
        print(f"{len(texts)}개의 리뷰 데이터를 분석합니다.")
        
        # 문서-키워드 행렬 생성
        doc_keyword_matrix, keywords = create_document_keyword_matrix(texts)
        
        # 키워드 동시 출현 행렬 생성
        cooccurrence_matrix = create_keyword_cooccurrence_matrix(doc_keyword_matrix, keywords)
        
        # 코사인 유사도 계산
        similarity_matrix = calculate_cosine_similarity(cooccurrence_matrix)
        
        # 유사도 히트맵 생성
        top_indices = create_similarity_heatmap(similarity_matrix, keywords)
        
        # 전체 키워드 네트워크 생성
        create_keyword_network(similarity_matrix, keywords, top_indices)
        
        # 특정 테마 관련 키워드 분석
        theme_list = ['어린이', '안전', '가성비', '디자인', '편의성']
        theme_keywords = find_specific_theme_keywords(keywords, theme_list)
        
        # 테마별 네트워크 생성
        create_theme_network(similarity_matrix, keywords, theme_keywords, theme_list)
        
        print("\n===== 키워드 상관관계 및 네트워크 분석 완료 =====")
        print(f"모든 결과물은 {OUTPUT_DIR} 폴더에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 