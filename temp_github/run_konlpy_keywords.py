import os
import sys

# Set Java environment variables
java_home = r"C:\Program Files\Java\jdk-21"
jvm_path = r"C:\Program Files\Java\jdk-21\bin\server\jvm.dll"

# Set environment variables
os.environ['JAVA_HOME'] = java_home
os.environ['JPY_JVM'] = jvm_path  # For JPype to find the JVM

print("Java environment variables set:")
print(f"JAVA_HOME: {os.environ.get('JAVA_HOME')}")
print(f"JPY_JVM: {os.environ.get('JPY_JVM')}")

# Instead of directly executing the script, let's import and run the main functionality
print("\nRunning the keyword analysis...")

# Import necessary modules
import pandas as pd
import re
import os
import platform
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from matplotlib import font_manager, rc

# Import KoNLPy after setting environment variables
from konlpy.tag import Okt

# Korean font setting
if platform.system() == 'Windows':
    font_path = font_manager.findfont(font_manager.FontProperties(family='Malgun Gothic'))
else:
    font_path = "NanumGothic.ttf"
    
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# Set stopwords
stopwords = set(['자전거', '제품', '삼천리', '사용', '정도', '구매', '배송', '다시', '너무', '정말', '하고', '합니다', '있어요', '입니다', '좋아요', 'ㅋㅋㅋ'])

# Text cleaning and tokenizing function
def clean_and_tokenize(text):
    text = str(text)
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z\s]", "", text)
    okt = Okt()
    nouns = okt.nouns(text)
    return [word for word in nouns if len(word) > 1 and word not in stopwords]

try:
    # Load data - try different encodings
    encodings = ['utf-8', 'cp949', 'euc-kr']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv("data/review_data.csv", encoding=encoding)
            print(f"✅ Successfully read file with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        print("[Error] Could not read the file with any encoding")
        sys.exit(1)
        
    if '리뷰내용' not in df.columns:
        print(f"[Error] Column '리뷰내용' is required. Available columns: {df.columns.tolist()}")
        sys.exit(1)
        
    texts = df["리뷰내용"].dropna().tolist()
    print(f"Processing {len(texts)} reviews...")

    # Collect keywords
    all_keywords = []
    print("Extracting keywords...")
    for t in texts:
        all_keywords.extend(clean_and_tokenize(t))

    counter = Counter(all_keywords)
    top_keywords = counter.most_common(50)

    # Visualization
    if top_keywords:
        print(f"Creating visualizations for {len(top_keywords)} keywords...")
        words, freqs = zip(*top_keywords)
        wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate_from_frequencies(dict(top_keywords))

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Word Cloud")

        plt.subplot(1, 2, 2)
        plt.barh(words[::-1][:20], freqs[::-1][:20], color='skyblue')  # Only display top 20
        plt.title("Top Keywords Frequency")
        plt.tight_layout()

        # Create output directory
        os.makedirs("output", exist_ok=True)
        plt.savefig("output/konlpy_keywords.png")
        print("✅ Keyword analysis complete: output/konlpy_keywords.png")
        
        # Print top keywords
        print("\n===== Top Keywords =====")
        for i, (word, freq) in enumerate(top_keywords[:20], 1):
            print(f"{i}. {word}: {freq} occurrences")
    else:
        print("[Warning] No keywords were extracted")
except Exception as e:
    print(f"Error occurred: {e}") 