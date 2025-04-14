# LDA 토픽 모델링 분석

## 목적
리뷰/SNS 데이터에서 숨겨진 관심사(토픽)를 자동으로 추출하기 위한 기법

## 방법
이 패키지에서는 두 가지 방식의 LDA(Latent Dirichlet Allocation) 구현을 제공합니다:

1. **경량 버전 (scikit-learn 기반)**: Windows 환경에서 추가 종속성 없이 실행 가능
2. **확장 버전 (gensim 기반)**: Coherence Score 평가 및 대화형 시각화 제공 (추가 라이브러리 필요)

## 분석 프로세스

### 1. 텍스트 전처리
- 특수문자, 숫자 제거
- 한글 명사 추출 (KoNLPy Okt 사용)
- 불용어 제거 (자전거, 제품, 사용 등 일반적인 단어)
- 한 글자 명사 제거

### 2. 문서-단어 행렬 생성
- CountVectorizer(scikit-learn) 또는 Dictionary/Corpus(gensim) 생성
- 극단적으로 희귀하거나 빈번한 단어 필터링

### 3. 최적 토픽 수 결정
- 5~10개 범위에서 자동 탐색
- Perplexity(scikit-learn) 또는 Coherence Score(gensim) 측정
- 평가 지표 그래프 시각화

### 4. 토픽별 핵심 키워드 추출
- 각 토픽별 주요 단어 10개 추출
- 키워드 워드클라우드 생성
- 토픽 키워드 텍스트 파일 저장

### 5. 대표 문서 추출
- 각 토픽에 대한 확률이 가장 높은 문서 3개 추출
- 토픽별 대표 문서 텍스트 파일 저장

### 6. 시각화
- 토픽별 워드클라우드
- 토픽 분포 (바차트)
- 전체 토픽 시각화 (pyLDAvis, 확장 버전만 해당)

## 결과 확인
모든 분석 결과는 `output/topic_modeling/` 폴더에 저장됩니다:

- **perplexity_score.png**: 토픽 수에 따른 Perplexity 변화 (낮을수록 좋음)
- **coherence_score.png**: 토픽 수에 따른 Coherence Score 변화 (높을수록 좋음, 확장 버전만 해당)
- **topics_keywords.txt**: 토픽별 주요 키워드 목록
- **topic_[번호]_wordcloud.png**: 각 토픽별 워드클라우드
- **representative_documents.txt**: 토픽별 대표 문서
- **topic_distribution.png**: 전체 문서의 토픽 분포
- **lda_visualization.html**: 대화형 토픽 시각화 (확장 버전만 해당)

## 실행 방법

### 경량 버전 (권장)
```
python run_lda_topic_modeling_light.py
```

### 확장 버전 (추가 패키지 필요)
```
python run_lda_topic_modeling.py
```

## 평가 및 해석

### Perplexity
- 모델이 얼마나 새로운 문서를 잘 표현하는지 나타내는 지표
- 낮을수록 좋음

### Coherence Score
- 토픽 내 단어들의 의미적 일관성을 측정
- 높을수록 좋음
- 사람의 주관적 평가와 더 일치하는 경향

### 토픽 해석
- 각 토픽의 키워드를 바탕으로 해당 토픽이 나타내는 개념 유추
- 대표 문서를 통해 토픽의 구체적인 맥락 파악
- 토픽 분포를 통해 전체 데이터에서 각 토픽의 중요도 파악 