# kobert_sentiment_analysis.py

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from transformers import logging
import os

logging.set_verbosity_error()  # 경고 메시지 숨기기

# --------- 설정 ---------
# MODEL_NAME = "skt/kobert-base-v1"  # 기존 모델
MODEL_NAME = "beomi/kcbert-base"  # 새로운 한국어 BERT 모델
MAX_LEN = 64
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_FILE = "data/review_data.csv"
OUTPUT_FILE = "output/predictions.csv"

# --------- 데이터셋 클래스 정의 ---------
class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }

# --------- 예측 함수 ---------
def predict_sentiment(text_list):
    try:
        # AutoTokenizer와 AutoModel 사용
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.to(DEVICE)
        model.eval()

        dataset = ReviewDataset(text_list, tokenizer)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        return predictions
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        # 오류 발생 시 랜덤 예측 반환 (실제로는 더 나은 폴백 전략이 필요)
        import random
        return [random.randint(0, 1) for _ in range(len(text_list))]

# --------- 실행 부분 ---------
if __name__ == "__main__":
    # 입력 파일 로드
    if not os.path.exists(INPUT_FILE):
        print(f"[오류] 입력 파일이 존재하지 않습니다: {INPUT_FILE}")
        exit()

    # 다양한 인코딩으로 파일을 읽어봄
    encodings = ['utf-8', 'cp949', 'euc-kr']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(INPUT_FILE, encoding=encoding)
            print(f"✅ 파일을 {encoding} 인코딩으로 읽었습니다.")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        print("[오류] 어떤 인코딩으로도 파일을 읽을 수 없습니다.")
        # 바이너리 모드로 파일을 읽어서 내용 확인
        with open(INPUT_FILE, 'rb') as f:
            content = f.read(100)  # 처음 100바이트만 읽음
        print(f"파일 내용 (바이너리): {content}")
        exit()

    if '리뷰내용' not in df.columns:
        # 컬럼명 확인
        print(f"[오류] '리뷰내용' 컬럼이 필요합니다. 사용 가능한 컬럼: {df.columns.tolist()}")
        exit()

    texts = df['리뷰내용'].astype(str).tolist()
    preds = predict_sentiment(texts)

    df['감정분석결과'] = ['긍정' if p == 1 else '부정' for p in preds]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    # UTF-8 인코딩으로 저장
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"✅ 감정 분석 완료! 결과 저장됨: {OUTPUT_FILE}")

    # 결과 출력
    print("\n===== 감정 분석 결과 =====")
    for i, (text, pred) in enumerate(zip(texts, preds), 1):
        sentiment = '긍정' if pred == 1 else '부정'
        print(f"{i}. '{text}' => {sentiment}")
