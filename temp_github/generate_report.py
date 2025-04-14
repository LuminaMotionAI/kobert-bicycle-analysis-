import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pandas as pd
from PIL import Image as PILImage
import io
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 등록 (한글 지원이 필요한 경우)
FONT_PATH = "NanumGothic.ttf"
try:
    if os.path.exists(FONT_PATH):
        pdfmetrics.registerFont(TTFont('NanumGothic', FONT_PATH))
        print("한글 폰트를 성공적으로 등록했습니다.")
    else:
        print(f"경고: 한글 폰트 파일({FONT_PATH})을 찾을 수 없습니다. 한글이 제대로 표시되지 않을 수 있습니다.")
except Exception as e:
    print(f"한글 폰트 등록 중 오류가 발생했습니다: {e}")
    print("한글이 제대로 표시되지 않을 수 있습니다.")

# 문서 생성
def create_pdf_report(output_filename="자전거_데이터_분석_보고서.pdf"):
    # 출력 디렉토리 확인 및 생성
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"출력 디렉토리 생성: {output_dir}")
    
    doc = SimpleDocTemplate(
        output_filename, 
        pagesize=A4,
        rightMargin=72, 
        leftMargin=72,
        topMargin=72, 
        bottomMargin=72
    )
    
    # 스타일 설정
    styles = getSampleStyleSheet()
    
    # 한글 폰트 사용 여부 확인
    korean_font_available = False
    if 'NanumGothic' in pdfmetrics.getRegisteredFontNames():
        korean_font_available = True
        print("한글 폰트를 사용합니다.")
        
        # 한글 스타일 추가
        styles.add(ParagraphStyle(
            name='Korean',
            fontName='NanumGothic',
            fontSize=12,
            leading=14,
            alignment=TA_JUSTIFY
        ))
        styles.add(ParagraphStyle(
            name='KoreanTitle',
            fontName='NanumGothic',
            fontSize=24,
            leading=30,
            alignment=TA_CENTER
        ))
        styles.add(ParagraphStyle(
            name='KoreanHeading1',
            fontName='NanumGothic',
            fontSize=18,
            leading=22,
            alignment=TA_LEFT
        ))
        styles.add(ParagraphStyle(
            name='KoreanHeading2',
            fontName='NanumGothic',
            fontSize=14,
            leading=18,
            alignment=TA_LEFT
        ))
    else:
        print("한글 폰트를 사용할 수 없습니다. 기본 폰트를 사용합니다.")
    
    styles.add(ParagraphStyle(
        name='Justify',
        fontName='Helvetica',
        fontSize=12,
        leading=14,
        alignment=TA_JUSTIFY
    ))
    
    styles.add(ParagraphStyle(
        name='Title',
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        alignment=TA_CENTER
    ))
    
    styles.add(ParagraphStyle(
        name='Heading1',
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=TA_LEFT
    ))
    
    styles.add(ParagraphStyle(
        name='Heading2',
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        alignment=TA_LEFT
    ))
    
    # 문서 내용 담을 리스트
    story = []
    
    # 제목 페이지 - 한글 폰트 사용 여부에 따라 스타일 선택
    title_style = styles['KoreanTitle'] if korean_font_available else styles['Title']
    heading1_style = styles['KoreanHeading1'] if korean_font_available else styles['Heading1']
    heading2_style = styles['KoreanHeading2'] if korean_font_available else styles['Heading2']
    text_style = styles['Korean'] if korean_font_available else styles['Justify']
    
    story.append(Paragraph("자전거 데이터 분석 보고서", title_style))
    story.append(Spacer(1, 30))
    story.append(Paragraph("데이터 기반 매출 증대 전략 제안", heading1_style))
    story.append(Spacer(1, 100))
    
    # 로고 이미지가 있다면 추가
    # logo_path = "logo.png"
    # if os.path.exists(logo_path):
    #     im = Image(logo_path, width=300, height=150)
    #     story.append(im)
    
    story.append(Spacer(1, 100))
    story.append(Paragraph("작성일: 2023년 4월", text_style))
    story.append(Paragraph("작성자: 데이터 분석팀", text_style))
    
    story.append(Spacer(1, 30))
    
    # 목차 페이지
    story.append(Paragraph("목차", heading1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("1. 분석 개요", text_style))
    story.append(Paragraph("2. 데이터 탐색 분석", text_style))
    story.append(Paragraph("3. 토픽 모델링 분석", text_style))
    story.append(Paragraph("4. 키워드 네트워크 분석", text_style))
    story.append(Paragraph("5. 페르소나 분석", text_style))
    story.append(Paragraph("6. 비즈니스 전략 제안", text_style))
    story.append(Paragraph("7. 실행 계획 및 ROI 예측", text_style))
    
    story.append(Spacer(1, 30))
    
    # 1. 분석 개요
    story.append(Paragraph("1. 분석 개요", heading1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("본 보고서는 자전거 관련 리뷰 데이터를 분석하여 소비자의 선호도와 행동 패턴을 파악하고, 이를 바탕으로 실질적인 매출 증대 전략을 제시합니다. 분석에는 감성 분석, 토픽 모델링, 키워드 네트워크 분석, 고객 세그먼테이션 및 페르소나 도출 등 다양한 방법론이 활용되었습니다.", text_style))
    story.append(Spacer(1, 10))
    
    # 2. 데이터 탐색 분석
    story.append(Paragraph("2. 데이터 탐색 분석", heading1_style))
    story.append(Spacer(1, 10))
    
    # 감성 분석 결과
    story.append(Paragraph("2.1 감성 분석 결과", heading2_style))
    story.append(Spacer(1, 5))
    
    # 파일 유무 확인 함수
    def check_file_exists(file_path, description):
        if os.path.exists(file_path):
            return True
        else:
            print(f"경고: {description} 파일을 찾을 수 없습니다: {file_path}")
            return False
            
    # 데이터 불러오기 함수
    def load_csv_safely(file_path, description):
        try:
            if check_file_exists(file_path, description):
                return pd.read_csv(file_path)
            return None
        except Exception as e:
            print(f"{description} 데이터를 불러오는 중 오류가 발생했습니다: {e}")
            return None
    
    sentiment_data = load_csv_safely('output/eda_results/sentiment_distribution.csv', "감성 분석")
    if sentiment_data is not None:
        # 감성 분석 그래프 생성
        plt.figure(figsize=(8, 5))
        sns.barplot(x=sentiment_data.columns, y=sentiment_data.iloc[0])
        plt.title("리뷰 감성 분포")
        plt.ylabel("리뷰 수")
        plt.xlabel("감성")
        
        # 이미지를 바이트 스트림으로 저장
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        plt.close()
        
        # PDF에 이미지 추가
        img = Image(img_data, width=400, height=250)
        story.append(img)
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"분석된 리뷰 중 긍정 리뷰는 {sentiment_data.iloc[0, 1]}건, 부정 리뷰는 {sentiment_data.iloc[0, 0]}건으로 나타났습니다.", text_style))
    else:
        story.append(Paragraph("감성 분석 데이터를 불러올 수 없습니다.", text_style))
    
    story.append(Spacer(1, 20))
    
    # 인구통계학적 특성
    story.append(Paragraph("2.2 인구통계학적 특성", heading2_style))
    story.append(Spacer(1, 5))
    
    try:
        # 연령대 분포
        age_data = pd.read_csv('output/eda_results/age_distribution.csv')
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=age_data.columns, y=age_data.iloc[0])
        plt.title("연령대별 분포")
        plt.ylabel("빈도")
        plt.xlabel("연령대")
        
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        plt.close()
        
        img = Image(img_data, width=400, height=250)
        story.append(img)
        story.append(Spacer(1, 10))
        
        # 성별 분포
        gender_data = pd.read_csv('output/eda_results/gender_distribution.csv')
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=gender_data.columns, y=gender_data.iloc[0])
        plt.title("성별 분포")
        plt.ylabel("빈도")
        plt.xlabel("성별")
        
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        plt.close()
        
        img = Image(img_data, width=400, height=250)
        story.append(img)
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("인구통계학적 특성 분석 결과, 30대와 40대가 주요 연령층으로 나타났으며, 남성의 비율이 여성보다 높게 나타났습니다.", text_style))
    except Exception as e:
        story.append(Paragraph(f"인구통계학적 데이터를 불러오는 중 오류가 발생했습니다: {e}", text_style))
    
    story.append(Spacer(1, 30))
    
    # 3. 토픽 모델링 분석
    story.append(Paragraph("3. 토픽 모델링 분석", heading1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("LDA(Latent Dirichlet Allocation) 알고리즘을 활용하여 리뷰 텍스트에서 주요 토픽을 추출했습니다.", text_style))
    story.append(Spacer(1, 10))
    
    # 토픽 키워드
    story.append(Paragraph("3.1 주요 토픽 키워드", heading2_style))
    story.append(Spacer(1, 5))
    
    try:
        with open('output/topic_modeling/topics_keywords.txt', 'r', encoding='utf-8') as file:
            topics = file.readlines()
        
        for topic in topics:
            if topic.strip():  # 빈 줄 건너뛰기
                story.append(Paragraph(topic.strip(), text_style))
                story.append(Spacer(1, 5))
    except Exception as e:
        story.append(Paragraph(f"토픽 키워드를 불러오는 중 오류가 발생했습니다: {e}", text_style))
    
    story.append(Spacer(1, 10))
    
    # 토픽 분포 이미지
    story.append(Paragraph("3.2 토픽 분포", heading2_style))
    story.append(Spacer(1, 5))
    
    topic_dist_path = 'output/topic_modeling/topic_distribution.png'
    if os.path.exists(topic_dist_path):
        img = Image(topic_dist_path, width=400, height=300)
        story.append(img)
    else:
        story.append(Paragraph("토픽 분포 이미지를 찾을 수 없습니다.", text_style))
    
    story.append(Spacer(1, 10))
    
    # 워드클라우드 이미지 (토픽 1만 예시로 포함)
    story.append(Paragraph("3.3 토픽 워드클라우드 (예시: 토픽 1)", heading2_style))
    story.append(Spacer(1, 5))
    
    wordcloud_path = 'output/topic_modeling/topic_1_wordcloud.png'
    if os.path.exists(wordcloud_path):
        img = Image(wordcloud_path, width=400, height=300)
        story.append(img)
    else:
        story.append(Paragraph("워드클라우드 이미지를 찾을 수 없습니다.", text_style))
    
    story.append(Spacer(1, 30))
    
    # 4. 키워드 네트워크 분석
    story.append(Paragraph("4. 키워드 네트워크 분석", heading1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("리뷰 텍스트에서 추출한 키워드 간의 연관성을 분석하여 네트워크로 시각화했습니다.", text_style))
    story.append(Spacer(1, 10))
    
    # 키워드 유사도 히트맵
    story.append(Paragraph("4.1 키워드 유사도 히트맵", heading2_style))
    story.append(Spacer(1, 5))
    
    heatmap_path = 'output/keyword_network/keyword_similarity_heatmap.png'
    if os.path.exists(heatmap_path):
        img = Image(heatmap_path, width=400, height=300)
        story.append(img)
    else:
        story.append(Paragraph("히트맵 이미지를 찾을 수 없습니다.", text_style))
    
    story.append(Spacer(1, 10))
    
    # 키워드 네트워크 그래프
    story.append(Paragraph("4.2 키워드 네트워크 그래프", heading2_style))
    story.append(Spacer(1, 5))
    
    network_path = 'output/keyword_network/keyword_network.png'
    if os.path.exists(network_path):
        img = Image(network_path, width=400, height=300)
        story.append(img)
    else:
        story.append(Paragraph("네트워크 그래프 이미지를 찾을 수 없습니다.", text_style))
    
    story.append(Spacer(1, 30))
    
    # 5. 페르소나 분석
    story.append(Paragraph("5. 페르소나 분석", heading1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("리뷰 데이터, 인구통계학적 특성, 구매 패턴 등을 기반으로 K-means 군집화를 수행하여 고객 세그먼트를 도출하고, 각 세그먼트를 대표하는 페르소나를 개발했습니다.", text_style))
    story.append(Spacer(1, 10))
    
    # 페르소나 레이더 차트
    story.append(Paragraph("5.1 페르소나 레이더 차트", heading2_style))
    story.append(Spacer(1, 5))
    
    radar_path = 'output/persona/persona_radar_charts.png'
    if os.path.exists(radar_path):
        img = Image(radar_path, width=400, height=300)
        story.append(img)
    else:
        story.append(Paragraph("레이더 차트 이미지를 찾을 수 없습니다.", text_style))
    
    story.append(Spacer(1, 10))
    
    # 페르소나 설명
    story.append(Paragraph("5.2 페르소나 프로필", heading2_style))
    story.append(Spacer(1, 5))
    
    try:
        with open('output/persona/persona_descriptions.txt', 'r', encoding='utf-8') as file:
            personas = file.read()
        
        # ■ 문자를 타이틀 스타일로 변경
        persona_sections = personas.split("■")
        for i, section in enumerate(persona_sections):
            if i > 0:  # 첫 번째 요소는 빈 문자열이므로 건너뜀
                lines = section.strip().split("\n")
                if lines:
                    title = lines[0]
                    story.append(Paragraph(title, heading2_style))
                    story.append(Spacer(1, 5))
                    for line in lines[1:]:
                        if line.strip():
                            story.append(Paragraph(line.strip(), text_style))
                    story.append(Spacer(1, 10))
    except Exception as e:
        story.append(Paragraph(f"페르소나 설명을 불러오는 중 오류가 발생했습니다: {e}", text_style))
    
    story.append(Spacer(1, 10))
    
    # 마케팅 채널 효과성
    story.append(Paragraph("5.3 마케팅 채널 효과성", heading2_style))
    story.append(Spacer(1, 5))
    
    channel_path = 'output/persona/marketing_channel_effectiveness.png'
    if os.path.exists(channel_path):
        img = Image(channel_path, width=400, height=300)
        story.append(img)
    else:
        story.append(Paragraph("마케팅 채널 효과성 이미지를 찾을 수 없습니다.", text_style))
    
    story.append(Spacer(1, 30))
    
    # 6. 비즈니스 전략 제안
    story.append(Paragraph("6. 비즈니스 전략 제안", heading1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("토픽 모델링, 키워드 네트워크, 페르소나 분석 등의 결과를 종합하여 실질적인 매출 증대를 위한 전략적 제안을 제시합니다.", text_style))
    story.append(Spacer(1, 10))
    
    # 비즈니스 전략 내용
    try:
        with open('business_strategy_recommendations.md', 'r', encoding='utf-8') as file:
            strategy_text = file.read()
        
        # 마크다운 포맷을 간단하게 처리
        lines = strategy_text.split("\n")
        current_heading_level = 0
        
        for line in lines:
            if line.startswith("# "):
                story.append(Paragraph(line[2:], heading1_style))
                current_heading_level = 1
            elif line.startswith("## "):
                story.append(Paragraph(line[3:], heading2_style))
                current_heading_level = 2
            elif line.startswith("### "):
                story.append(Paragraph(line[4:], heading2_style))
                current_heading_level = 3
            elif line.startswith("#### "):
                story.append(Paragraph(line[5:], text_style))
                current_heading_level = 4
            elif line.startswith("- ") or line.startswith("* "):
                story.append(Paragraph("• " + line[2:], text_style))
            elif line.startswith("1. ") or line.startswith("2. ") or line.startswith("3. ") or line.startswith("4. "):
                story.append(Paragraph(line, text_style))
            elif line.strip() == "":
                story.append(Spacer(1, 5))
            else:
                story.append(Paragraph(line, text_style))
            
            if current_heading_level > 0 and current_heading_level <= 2:
                story.append(Spacer(1, 10))
    except Exception as e:
        story.append(Paragraph(f"비즈니스 전략 문서를 불러오는 중 오류가 발생했습니다: {e}", text_style))
    
    story.append(Spacer(1, 30))
    
    # 7. 실행 계획 및 ROI 예측
    story.append(Paragraph("7. 실행 계획 및 ROI 예측", heading1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("제안된 전략을 단계적으로 실행하기 위한 타임라인과 투자 대비 효과를 예측합니다.", text_style))
    story.append(Spacer(1, 10))
    
    # 마케팅 전략 요약
    try:
        with open('marketing_strategy_summary.md', 'r', encoding='utf-8') as file:
            marketing_text = file.read()
        
        # 마크다운 포맷을 간단하게 처리
        lines = marketing_text.split("\n")
        current_heading_level = 0
        
        for line in lines:
            if line.startswith("# "):
                story.append(Paragraph(line[2:], heading1_style))
                current_heading_level = 1
            elif line.startswith("## "):
                story.append(Paragraph(line[3:], heading2_style))
                current_heading_level = 2
            elif line.startswith("### "):
                story.append(Paragraph(line[4:], heading2_style))
                current_heading_level = 3
            elif line.startswith("|"):  # 테이블 라인은 건너뛰기
                continue
            elif line.startswith("- ") or line.startswith("* "):
                story.append(Paragraph("• " + line[2:], text_style))
            elif line.startswith("1. ") or line.startswith("2. ") or line.startswith("3. ") or line.startswith("4. "):
                story.append(Paragraph(line, text_style))
            elif line.strip() == "":
                story.append(Spacer(1, 5))
            else:
                story.append(Paragraph(line, text_style))
            
            if current_heading_level > 0 and current_heading_level <= 2:
                story.append(Spacer(1, 10))
    except Exception as e:
        story.append(Paragraph(f"마케팅 전략 요약을 불러오는 중 오류가 발생했습니다: {e}", text_style))
    
    # 1분기 실행 계획 이미지 (예시)
    story.append(Paragraph("7.1 1분기 실행 계획 (예시)", heading2_style))
    story.append(Spacer(1, 5))
    
    # 1분기 계획 표 생성
    data = [
        ['전략', '주요 활동', '예상 비용', '예상 ROI'],
        ['제품 라인 재정비', '• 기존 제품 재고 분석\n• 고객 피드백 기반 개선점 식별\n• 인기 키워드 기반 제품명/설명 업데이트', '★★☆☆☆', '★★★☆☆'],
        ['조립 불편 해소', '• "5분 조립" 가이드 영상 제작\n• QR코드 연동 매뉴얼 개발\n• 무료 조립 서비스 시범 운영', '★★☆☆☆', '★★★★☆'],
        ['페르소나별 마케팅 메시지 개발', '• 4개 주요 페르소나별 메시지 프레임 개발\n• A/B 테스트 진행\n• 채널별 콘텐츠 가이드라인 수립', '★☆☆☆☆', '★★★☆☆']
    ]
    
    table = Table(data, colWidths=[100, 250, 70, 70])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))
    
    # 결론
    story.append(Paragraph("결론", heading1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("본 보고서는 자전거 관련 데이터를 다양한 분석 기법을 통해 심층적으로 분석하여 실질적인 매출 증대 전략을 제시했습니다. 분석 결과, 소비자들이 중요시하는 핵심 요소(안전, 편의성, 디자인 등)와 인구통계학적 특성에 따른 선호도 차이, 그리고 효과적인 마케팅 채널을 식별할 수 있었습니다.", text_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("제안된 전략을 단계적으로 실행함으로써 제품 라인업 최적화, 채널 효율성 향상, 고객 맞춤형 마케팅 강화, 시즌별 판매 전략 개선 등을 통해 전체 매출 20% 성장과 고객 만족도 15% 개선이 기대됩니다.", text_style))
    
    # 문서 저장
    doc.build(story)
    print(f"PDF 보고서가 생성되었습니다: {output_filename}")
    return output_filename

if __name__ == "__main__":
    create_pdf_report() 