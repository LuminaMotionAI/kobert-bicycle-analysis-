import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def get_python_version():
    """현재 파이썬 버전을 반환합니다."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    return f"{major}.{minor}"

def is_package_installed(package_name):
    """패키지가 설치되어 있는지 확인합니다."""
    return importlib.util.find_spec(package_name) is not None

def install_requirements():
    """필요한 패키지를 설치합니다."""
    python_version = get_python_version()
    print(f"파이썬 버전: {python_version}")
    
    # 필수 패키지 리스트
    required_packages = ['streamlit', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'networkx', 'pillow', 'wordcloud']
    packages_to_install = []
    
    # 설치가 필요한 패키지 확인
    for package in required_packages:
        if not is_package_installed(package.split('==')[0]):
            packages_to_install.append(package)
    
    if packages_to_install:
        print(f"다음 패키지를 설치합니다: {', '.join(packages_to_install)}")
        
        # Python 3.13 이상에서는 pip 사용 방식이 다름
        if float(python_version.split('.')[0]) >= 3 and float(python_version.split('.')[1]) >= 13:
            print("Python 3.13 이상 버전이 감지되었습니다. 새로운 pip 설치 방식을 사용합니다.")
            try:
                for package in packages_to_install:
                    subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", package], check=True)
                print("모든 패키지가 성공적으로 설치되었습니다.")
                return True
            except subprocess.CalledProcessError as e:
                print(f"패키지 설치 중 오류가 발생했습니다: {e}")
                return False
        else:
            # 기존 방식 사용
            try:
                subprocess.run([sys.executable, "-m", "pip", "install"] + packages_to_install, check=True)
                print("모든 패키지가 성공적으로 설치되었습니다.")
                return True
            except subprocess.CalledProcessError as e:
                print(f"패키지 설치 중 오류가 발생했습니다: {e}")
                return False
    else:
        print("모든 필수 패키지가 이미 설치되어 있습니다.")
        return True

def check_output_folders():
    """필요한 출력 폴더가 존재하는지 확인하고, 결과를 반환합니다."""
    required_folders = [
        'output/eda_results',
        'output/topic_modeling',
        'output/keyword_network',
        'output/persona'
    ]
    
    missing_folders = []
    for folder in required_folders:
        if not os.path.exists(folder):
            missing_folders.append(folder)
    
    if missing_folders:
        print(f"경고: 다음 폴더가 존재하지 않습니다: {', '.join(missing_folders)}")
        print("분석을 먼저 실행하여 필요한 데이터를 생성해야 합니다.")
        return False
    return True

def generate_pdf_report():
    """PDF 보고서를 생성합니다."""
    try:
        from generate_report import create_pdf_report
        print("PDF 보고서 생성을 시작합니다...")
        create_pdf_report("output/자전거_데이터_분석_보고서.pdf")
        print("PDF 보고서가 성공적으로 생성되었습니다.")
        return True
    except ImportError:
        print("generate_report 모듈을 찾을 수 없습니다.")
        return False
    except Exception as e:
        print(f"PDF 보고서 생성 중 오류가 발생했습니다: {e}")
        return False

def run_dashboard():
    """대시보드를 실행합니다."""
    if not install_requirements():
        print("필요한 패키지를 설치할 수 없어 대시보드를 실행할 수 없습니다.")
        return
    
    if not check_output_folders():
        print("필요한 데이터 폴더가 없어 대시보드를 실행할 수 없습니다.")
        return
    
    # 대시보드 스크립트 경로
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"대시보드 스크립트를 찾을 수 없습니다: {dashboard_path}")
        return
    
    try:
        print("대시보드를 시작합니다...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path), "--server.port=8501"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"대시보드 실행 중 오류가 발생했습니다: {e}")
    except KeyboardInterrupt:
        print("사용자에 의해 대시보드가 종료되었습니다.")

if __name__ == "__main__":
    run_dashboard() 