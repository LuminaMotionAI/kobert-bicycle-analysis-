# KoBERT Sentiment Analysis with KoNLPy

This package provides sentiment analysis and keyword extraction tools for Korean text, using KoBERT and KoNLPy.

## Requirements

- Python 3.8+
- Java JDK (required for KoNLPy)
- Required Python packages:
  - pandas
  - numpy
  - torch
  - transformers
  - konlpy
  - matplotlib
  - seaborn
  - wordcloud
  - scikit-learn
  - gensim (optional, for full LDA implementation)
  - pyLDAvis (optional, for interactive LDA visualization)

## Java Environment Setup

For KoNLPy to work properly, you need to have Java installed and the environment properly configured. 
This package includes helper scripts that set these environment variables automatically.

## Running the Scripts

All scripts should be run using the wrapper scripts provided, as they set up the Java environment properly:

### 1. Test KoNLPy Environment

```
python run_konlpy.py
```

This will test if KoNLPy can be initialized properly with the JVM.

### 2. Keyword Analysis

```
python run_konlpy_keywords.py
```

This will analyze the review data and extract top keywords, generating a word cloud visualization.

### 3. Sentiment Analysis

```
python run_kobert_sentiment.py
```

This script allows you to choose between two sentiment analysis models.

### 4. Text Mining Analysis

```
python run_text_mining.py
```

Performs comprehensive text mining on the bicycle review data.

### 5. Numerical Data Analysis

```
python bicycle_numerical_analysis.py
```

Performs numerical analysis on bicycle-related datasets.

### 6. Exploratory Data Analysis (EDA)

```
python bicycle_eda.py
```

Performs exploratory data analysis combining numerical and text analysis results.

### 7. LDA Topic Modeling

이 패키지는 두 가지 버전의 LDA 토픽 모델링 스크립트를 제공합니다:

### 경량 버전 (권장)
Windows 환경에서 추가 패키지 없이 실행 가능한 scikit-learn 기반 LDA 구현입니다. 다음 기능을 제공합니다:
- 최적의 토픽 수를 perplexity 기반으로 자동 탐색 (5-10개 범위)
- 각 토픽별 워드클라우드 생성
- 토픽별 대표 문서 추출
- 토픽 분포 시각화
- 키워드 중요도 분석

### 전체 버전
`gensim`과 `pyLDAvis`를 활용한 확장 기능을 제공합니다:
- Coherence Score를 통한 토픽 품질 평가
- 대화형 토픽 시각화
- 토픽별 키워드 네트워크

## 키워드 상관관계 및 네트워크 분석

이 기능은 리뷰 데이터에서 추출한 키워드들 간의 관계를 분석하고 시각화합니다:

### 기능
- **코사인 유사도 기반 히트맵**: 주요 키워드 간의 유사도를 히트맵으로 시각화
- **키워드 네트워크 그래프**: 키워드 간 연결 관계와 중요도를 네트워크 그래프로 표현
- **테마별 네트워크 분석**: 특정 테마('어린이', '안전', '가성비' 등)를 중심으로 연관 키워드 네트워크 시각화
- **상위 연관 키워드 추출**: 키워드 쌍의 유사도를 기준으로 상위 연관 관계 목록 제공

### 실행 방법
```
python run_keyword_network_analysis.py
```

### 결과물
분석 결과는 `output/keyword_network/` 폴더에 저장됩니다:
- `keyword_similarity_heatmap.png`: 상위 키워드 간 코사인 유사도 히트맵
- `keyword_network.png`: 전체 키워드 네트워크 그래프
- `top_keyword_relations.txt`: 상위 연관 키워드 쌍과 유사도 점수
- `theme_[테마명]_network.png`: 특정 테마 중심의 키워드 네트워크
- `theme_[테마명]_relations.txt`: 특정 테마 관련 연관 키워드 목록

## 페르소나 도출 및 시각화

고객 데이터를 분석하여 주요 고객 세그먼트를 대표하는 페르소나를 도출하고 시각화합니다:

### 기능
- **고객 군집화**: 연령/지역/구매/감성/검색 패턴을 기반으로 K-means 군집 분석
- **페르소나 프로필**: 각 고객 세그먼트의 특성을 반영한 구체적인 페르소나 설명
- **레이더 차트**: 페르소나별 특성을 다차원으로 시각화
- **고객 여정 타임라인**: 각 페르소나의 구매 여정과 단계별 만족도 분석
- **마케팅 채널 매핑**: 페르소나별 효과적인 마케팅 채널 및 전환율 분석

### 실행 방법
```
python run_persona_analysis.py
```

### 결과물
분석 결과는 `output/persona/` 폴더에 저장됩니다:
- `persona_radar_charts.png`: 페르소나별 특성 레이더 차트
- `persona_descriptions.txt`: 페르소나 상세 설명 및 특성
- `cluster_profiles.csv`: 군집 분석 결과 요약
- `customer_journey_timeline.png`: 페르소나별 고객 여정 타임라인
- `marketing_channel_effectiveness.png`: 채널 효과성 매트릭스
- `conversion_funnel_by_persona.png`: 페르소나별 전환 퍼널
- `marketing_channel_map.png`: 온/오프라인 마케팅 채널 중요도 맵

## 전략 도출 및 매출 개선 제안

분석 결과를 종합하여 실질적인 매출 증가를 위한 전략적 제안을 제시합니다:

### 기능
- **제품군 재구성**: 인기 키워드와 토픽 모델링 결과를 기반으로 제품 라인업 최적화
- **채널 전략**: 감성 분석과 키워드 네트워크 분석을 활용한 효과적인 마케팅 채널 전략
- **고객 맞춤 마케팅**: 페르소나별 타겟 메시지와 구매 여정 최적화
- **시간대/시즌별 판매 전략**: 계절 및 시간대별 최적화된 마케팅 캠페인

### 결과물
- `business_strategy_recommendations.md`: 종합적인 비즈니스 전략 제안 문서
- `marketing_strategy_summary.md`: 우선순위 및 실행 타임라인이 포함된 마케팅 전략 요약

### 주요 전략 영역
1. **제품 라인업 최적화**: 안전 강화 라인, 편의성 최적화 라인, 모듈형 커스터마이징 라인, 라이프스타일 특화 제품군
2. **마케팅 채널 최적화**: 콘텐츠 마케팅 강화, SNS 채널별 차별화, 부정 요소 선제적 해소, 옴니채널 전략
3. **고객 세그먼트별 접근**: 페르소나별 타겟 메시지, 구매 여정 최적화, 맞춤형 프로모션, 충성도 프로그램
4. **시즌별 전략**: 시즌별 프로모션 캘린더, 요일/시간대별 마케팅 최적화, 시즌 전환 전략, 연간 이벤트 계획

## Data Structure

- `data/`: Contains the input data files
- `output/`: Contains output files, visualizations, and analysis results
  - `output/topic_modeling/`: Contains LDA topic modeling results
- `Bicycle/`: Contains bicycle-related datasets

## Troubleshooting

If you encounter issues with KoNLPy or JVM:

1. Make sure Java is properly installed (JDK 8 or higher)
2. Verify that the paths in the wrapper scripts match your Java installation
3. If using a different Java version, update the paths in:
   - `run_konlpy.py`
   - `run_konlpy_keywords.py`
   - `run_kobert_sentiment.py`
   - `run_text_mining.py`
   - `run_lda_topic_modeling.py`
   - `run_lda_topic_modeling_light.py` 

## 대시보드 시각화

분석 결과를 종합적으로 시각화하는 인터랙티브 대시보드를 제공합니다:

### 기능
- **데이터 개요**: 지역별, 연령별, 성별 분포 시각화
- **감성 분석**: 긍정/부정 감성 분포 및 예측 결과 조회
- **토픽 모델링**: 주제 키워드, 토픽 분포, 워드클라우드 시각화
- **키워드 네트워크**: 키워드 유사도 히트맵과 네트워크 그래프
- **페르소나**: 페르소나 프로필, 레이더 차트, 고객 여정 시각화
- **마케팅 채널**: 채널 효과성, 전환 퍼널, 채널 맵 시각화
- **데이터 인사이트**: 데이터 기반 심층 통찰 제공

### 실행 방법
```
python run_dashboard.py
```

대시보드는 Streamlit을 사용하여 구현되었으며, 실행 시 자동으로 웹 브라우저가 열리고 `http://localhost:8501`에서 접근할 수 있습니다.

### Streamlit Cloud 배포
본 대시보드는 GitHub 저장소를 Streamlit Cloud에 연결하여 온라인으로 배포할 수 있습니다:

1. GitHub에 저장소 업로드
2. [Streamlit Cloud](https://streamlit.io/cloud)에서 계정 생성
3. GitHub 저장소 연결 및 `dashboard.py` 지정
4. 자동 배포 완료

필요한 파일:
- `dashboard.py`: 대시보드 메인 스크립트
- `run_dashboard.py`: 환경 설정 및 실행 스크립트
- `requirements.txt`: 필요 패키지 정의
- `NanumGothic.ttf`: 한글 폰트 파일
- `output/`: 분석 결과 폴더
- `data/`: 원본 데이터 폴더 