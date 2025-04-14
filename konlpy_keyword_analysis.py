# konlpy_keyword_analysis.py

import pandas as pd
import re
from collections import Counter
import os
import platform
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib import font_manager, rc

# JVM 설정 (KoNLPy 임포트 전에 설정해야 함)
import os
import sys
if platform.system() == 'Windows':
    # Java 경로 설정
    java_home = os.environ.get('JAVA_HOME')
    if not java_home:
        # 일반적인 Java 경로 시도
        if os.path.exists("C:/Program Files/Java"):
            java_dirs = os.listdir("C:/Program Files/Java")
            if java_dirs:
                # 가장 최신 버전 선택 (알파벳 순으로 정렬된 마지막 항목)
                newest_dir = sorted(java_dirs)[-1]
                java_home = f"C:/Program Files/Java/{newest_dir}"
                os.environ['JAVA_HOME'] = java_home
                print(f"JAVA_HOME 설정됨: {java_home}")

    if java_home:
        # JPype에서 참조할 JVM 경로 설정
        jvm_path = f"{java_home}/bin/server/jvm.dll"
        if os.path.exists(jvm_path):
            os.environ['JPY_JVM'] = jvm_path
            print(f"JVM 경로 설정됨: {jvm_path}")
        else:
            # 다른 jvm.dll 위치 시도
            jvm_path = f"{java_home}/jre/bin/server/jvm.dll"
            if os.path.exists(jvm_path):
                os.environ['JPY_JVM'] = jvm_path
                print(f"JVM 경로 설정됨: {jvm_path}")

# 이제 KoNLPy 임포트
from konlpy.tag import Okt

# 폰트 설정
if platform.system() == 'Windows':
    # Windows 시스템 폰트 사용
    font_path = font_manager.findfont(font_manager.FontProperties(family='Malgun Gothic'))
else:
    # NanumGothic 폰트 경로
    font_path = "NanumGothic.ttf"
    
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# 불용어 설정
stopwords = set(['자전거', '제품', '삼천리', '사용', '정도', '구매', '배송', '다시', '너무', '정말', '하고', '합니다', '있어요', '입니다', '좋아요', 'ㅋㅋㅋ'])

# 텍스트 전처리 및 키워드 추출 함수
def clean_and_tokenize(text):
    text = str(text)
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z\s]", "", text)
    okt = Okt()
    nouns = okt.nouns(text)
    return [word for word in nouns if len(word) > 1 and word not in stopwords]

try:
    # 데이터 불러오기 - 다양한 인코딩 시도
    encodings = ['utf-8', 'cp949', 'euc-kr']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv("data/review_data.csv", encoding=encoding)
            print(f"✅ 파일을 {encoding} 인코딩으로 읽었습니다.")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        print("[오류] 어떤 인코딩으로도 파일을 읽을 수 없습니다.")
        exit()
        
    if '리뷰내용' not in df.columns:
        print(f"[오류] '리뷰내용' 컬럼이 필요합니다. 사용 가능한 컬럼: {df.columns.tolist()}")
        exit()
        
    texts = df["리뷰내용"].dropna().tolist()

    # 키워드 수집
    all_keywords = []
    for t in texts:
        all_keywords.extend(clean_and_tokenize(t))

    counter = Counter(all_keywords)
    top_keywords = counter.most_common(50)

    # 시각화
    if top_keywords:
        words, freqs = zip(*top_keywords)
        wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate_from_frequencies(dict(top_keywords))

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("워드클라우드")

        plt.subplot(1, 2, 2)
        plt.barh(words[::-1][:20], freqs[::-1][:20], color='skyblue')  # 상위 20개만 표시
        plt.title("상위 키워드 빈도수")
        plt.tight_layout()

        # 출력 디렉터리 생성
        os.makedirs("output", exist_ok=True)
        plt.savefig("output/konlpy_keywords.png")
        print("✅ 키워드 분석 완료: output/konlpy_keywords.png")
        
        # 상위 키워드 출력
        print("\n===== 상위 키워드 =====")
        for i, (word, freq) in enumerate(top_keywords[:20], 1):
            print(f"{i}. {word}: {freq}회")
    else:
        print("[주의] 키워드가 추출되지 않았습니다.")
except Exception as e:
    print(f"오류 발생: {e}")
