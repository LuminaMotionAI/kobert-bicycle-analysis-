#!/usr/bin/env python
# -*- coding: utf-8 -*-

# run_persona_analysis.py
# 페르소나 분석 실행 스크립트

import os
import sys

print("===== 6단계: 페르소나 도출 및 시각화 =====")
print("고객 군집화 → 대표 페르소나 3~5개 도출")

# 출력 폴더 생성
output_dir = "output/persona"
os.makedirs(output_dir, exist_ok=True)

# 1. 페르소나 군집 분석 실행
print("\n[1/2] 페르소나 군집 분석 실행 중...")
exec(open("persona_analysis.py", encoding="utf-8").read())

# 2. 마케팅 채널 분석 실행
print("\n[2/2] 마케팅 채널 분석 실행 중...")
exec(open("marketing_channel_analysis.py", encoding="utf-8").read())

print("\n===== 페르소나 분석 완료 =====")
print(f"모든 결과물은 {output_dir} 폴더에 저장되었습니다.")
print("""
생성된 페르소나:
1. 김서준 (30대 남성 직장인) - "주말에 가족과 나들이 가는 30대 가장"
2. 이지현 (20대 여성 전문직) - "건강 관리와 취미를 위한 자전거를 찾는 여성"
3. 박민우 (40대 남성 가장) - "가족 모두가 함께 자전거 여행을 즐기길 원하는 40대"
4. 최현우 (20대 남성 대학생) - "통학과 여가를 위한 저가형 자전거를 찾는 대학생"
""") 