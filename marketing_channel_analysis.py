#!/usr/bin/env python
# -*- coding: utf-8 -*-

# marketing_channel_analysis.py
# 페르소나별 마케팅 채널 분석 및 매핑

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from matplotlib import font_manager, rc
import platform
import seaborn as sns
import matplotlib.patches as mpatches

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
OUTPUT_DIR = "output/persona"

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_channel_effectiveness_matrix():
    """페르소나별 마케팅 채널 효과성 매트릭스 생성"""
    
    print("마케팅 채널 효과성 매트릭스 분석 중...")
    
    # 페르소나 및 채널 정의
    personas = [
        "페르소나 1: 김서준 (30대 남성, 직장인)",
        "페르소나 2: 이지현 (20대 여성, 전문직)",
        "페르소나 3: 박민우 (40대 남성, 가장)",
        "페르소나 4: 최현우 (20대 남성, 대학생)"
    ]
    
    channels = [
        "포털 검색",
        "SNS (인스타그램)",
        "SNS (유튜브)",
        "온라인 커뮤니티",
        "블로그 리뷰",
        "오프라인 매장",
        "친구/지인 추천",
        "대중교통 광고",
        "TV 광고"
    ]
    
    # 채널별 효과성 점수 (0-10) - 각 페르소나에 대한 채널 효율성
    effectiveness = np.array([
        # 페르소나 1 (30대 남성 직장인)
        [7, 6, 9, 8, 8, 7, 6, 4, 5],
        # 페르소나 2 (20대 여성 전문직)
        [6, 9, 8, 7, 9, 5, 7, 6, 4],
        # 페르소나 3 (40대 남성 가장)
        [8, 4, 6, 7, 7, 9, 8, 5, 7],
        # 페르소나 4 (20대 남성 대학생)
        [6, 7, 10, 9, 6, 3, 8, 7, 3]
    ])
    
    # 효과성 매트릭스 히트맵 생성
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        effectiveness, 
        annot=True, 
        cmap="YlGnBu", 
        xticklabels=channels, 
        yticklabels=personas,
        cbar_kws={'label': '채널 효과성 점수 (0-10)'}
    )
    plt.title('페르소나별 마케팅 채널 효과성 매트릭스', fontsize=16, pad=20)
    plt.xlabel('마케팅 채널', fontsize=12, labelpad=10)
    plt.ylabel('고객 페르소나', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/marketing_channel_effectiveness.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 데이터프레임으로 효과성 매트릭스 저장
    effectiveness_df = pd.DataFrame(effectiveness, index=personas, columns=channels)
    effectiveness_df.to_csv(f"{OUTPUT_DIR}/marketing_channel_effectiveness.csv", encoding='utf-8')
    
    print(f"마케팅 채널 효과성 매트릭스가 {OUTPUT_DIR}/marketing_channel_effectiveness.png에 저장되었습니다.")
    return effectiveness_df

def create_channel_conversion_funnel():
    """페르소나별 채널 전환 퍼널 시각화"""
    
    print("채널 전환 퍼널 시각화 생성 중...")
    
    # 전환 단계
    stages = ['인지', '관심', '고려', '구매', '충성도']
    
    # 각 페르소나별 전환율 (%)
    conversion_rates = [
        # 페르소나 1 (30대 남성 직장인)
        [100, 65, 40, 28, 15],
        # 페르소나 2 (20대 여성 전문직)
        [100, 72, 45, 25, 12],
        # 페르소나 3 (40대 남성 가장)
        [100, 58, 38, 30, 22],
        # 페르소나 4 (20대 남성 대학생)
        [100, 70, 42, 20, 8]
    ]
    
    # 최적 채널 조합 (각 단계별 가장 효과적인 채널)
    optimal_channels = [
        # 페르소나 1 (30대 남성 직장인)
        {
            '인지': ['유튜브', '포털 검색'],
            '관심': ['온라인 커뮤니티', '블로그 리뷰'],
            '고려': ['블로그 리뷰', '오프라인 매장'],
            '구매': ['오프라인 매장', '공식 웹사이트'],
            '충성도': ['SNS 팔로우', '멤버십 프로그램']
        },
        # 페르소나 2 (20대 여성 전문직)
        {
            '인지': ['인스타그램', '유튜브'],
            '관심': ['블로그 리뷰', '인플루언서'],
            '고려': ['온라인 커뮤니티', '리뷰 영상'],
            '구매': ['온라인몰', '인스타그램 샵'],
            '충성도': ['SNS 팔로우', '추천 프로그램']
        },
        # 페르소나 3 (40대 남성 가장)
        {
            '인지': ['포털 검색', 'TV 광고'],
            '관심': ['전문 매장', '블로그 리뷰'],
            '고려': ['오프라인 매장', '지인 추천'],
            '구매': ['대리점', '직영점'],
            '충성도': ['A/S 프로그램', '멤버십 혜택']
        },
        # 페르소나 4 (20대 남성 대학생)
        {
            '인지': ['유튜브', '커뮤니티'],
            '관심': ['친구 추천', '온라인 커뮤니티'],
            '고려': ['가격 비교 사이트', '중고거래 앱'],
            '구매': ['중고거래', '할인 행사'],
            '충성도': ['SNS 공유', '친구 추천']
        }
    ]
    
    persona_names = [
        "김서준 (30대 남성, 직장인)", 
        "이지현 (20대 여성, 전문직)",
        "박민우 (40대 남성, 가장)",
        "최현우 (20대 남성, 대학생)"
    ]
    
    # 각 페르소나별 전환 퍼널 생성
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    axs = axs.flatten()
    
    for i, (persona, rates, channels) in enumerate(zip(persona_names, conversion_rates, optimal_channels)):
        ax = axs[i]
        
        # 퍼널 그래프 생성
        ax.bar(stages, rates, width=0.6, color=plt.cm.tab10(i), alpha=0.8)
        
        # 데이터 레이블 추가
        for j, rate in enumerate(rates):
            ax.text(j, rate + 2, f"{rate}%", ha='center', fontweight='bold')
        
        # 채널 정보 표시
        for j, stage in enumerate(stages):
            if stage in channels:
                channel_text = '\n'.join(channels[stage])
                ax.annotate(
                    channel_text, 
                    xy=(j, rates[j] / 2), 
                    ha='center', 
                    va='center',
                    color='white',
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc='royalblue', alpha=0.6)
                )
        
        # 그래프 설정
        ax.set_title(f'페르소나 {i+1}: {persona}', fontsize=14, pad=20)
        ax.set_ylim(0, 105)
        ax.set_ylabel('전환율 (%)', fontsize=12)
        ax.set_xlabel('고객 여정 단계', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/conversion_funnel_by_persona.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"채널 전환 퍼널 시각화가 {OUTPUT_DIR}/conversion_funnel_by_persona.png에 저장되었습니다.")

def create_marketing_channel_map():
    """마케팅 채널 맵 시각화"""
    
    print("마케팅 채널 맵 시각화 생성 중...")
    
    # 채널 정의
    channels = {
        'online': [
            '포털 검색 (네이버, 구글)',
            '유튜브',
            '인스타그램',
            '페이스북',
            '온라인 커뮤니티',
            '블로그',
            '쇼핑몰 리뷰',
            '중고거래 플랫폼'
        ],
        'offline': [
            '오프라인 매장',
            '대리점',
            '자전거 전시회',
            '옥외 광고',
            '지인 추천',
            '동호회',
            'TV/라디오',
            '잡지/신문'
        ]
    }
    
    # 페르소나별 주요 채널 (온라인/오프라인)
    persona_channels = {
        '페르소나 1\n(30대 남성 직장인)': {
            'online': [3, 5, 4, 2, 5, 4, 3, 1],  # 채널별 중요도 (1-5)
            'offline': [4, 3, 2, 2, 3, 1, 2, 1]
        },
        '페르소나 2\n(20대 여성 전문직)': {
            'online': [3, 4, 5, 3, 4, 5, 4, 1],
            'offline': [3, 2, 1, 3, 4, 1, 2, 3]
        },
        '페르소나 3\n(40대 남성 가장)': {
            'online': [4, 3, 2, 1, 3, 3, 4, 2],
            'offline': [5, 4, 3, 2, 4, 3, 4, 3]
        },
        '페르소나 4\n(20대 남성 대학생)': {
            'online': [3, 5, 4, 2, 5, 3, 3, 5],
            'offline': [2, 1, 1, 3, 4, 2, 1, 1]
        }
    }
    
    # 시각화 생성
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    axs = axs.flatten()
    
    for i, (persona, channel_data) in enumerate(persona_channels.items()):
        ax = axs[i]
        
        # 데이터 준비
        online_values = channel_data['online']
        offline_values = channel_data['offline']
        
        x = np.arange(len(channels['online']))
        width = 0.35
        
        # 바 차트 생성
        ax.bar(x - width/2, online_values, width, label='온라인 채널', color='cornflowerblue', alpha=0.8)
        ax.bar(x + width/2, offline_values, width, label='오프라인 채널', color='lightcoral', alpha=0.8)
        
        # 그래프 설정
        ax.set_title(persona, fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(channels['online'], rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 6)
        ax.set_ylabel('중요도 (1-5)', fontsize=12)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 보조 x축 추가 (오프라인 채널 이름)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x)
        ax2.set_xticklabels(channels['offline'], rotation=45, ha='left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/marketing_channel_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"마케팅 채널 맵 시각화가 {OUTPUT_DIR}/marketing_channel_map.png에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    print("===== 페르소나별 마케팅 채널 분석 시작 =====")
    
    # 채널 효과성 매트릭스 생성
    create_channel_effectiveness_matrix()
    
    # 채널 전환 퍼널 시각화
    create_channel_conversion_funnel()
    
    # 마케팅 채널 맵 생성
    create_marketing_channel_map()
    
    print("\n===== 마케팅 채널 분석 완료 =====")
    print(f"모든 결과물은 {OUTPUT_DIR} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main() 