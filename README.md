텍스트 파일 장르 분류기
프로젝트 소개
이 프로젝트는 웹 크롤링을 통해 수집한 텍스트 데이터를 기반으로, 새로운 텍스트 파일의 장르를 자동으로 분류하고 예측하는 시스템입니다. 머신러닝 모델을 학습시키고, 이를 활용한 웹 애플리케이션을 구축하여 사용자가 직접 장르 예측을 체험할 수 있도록 설계되었습니다.

주요 기능
데이터 수집: 구텐베르크 프로젝트 웹사이트에서 장르별 텍스트 파일 크롤링

데이터 전처리: NLTK 라이브러리를 사용한 불용어 제거, 토큰화, 표제어 추출 등 텍스트 정규화

모델 학습: TF-IDF 벡터화와 로지스틱 회귀 모델을 사용한 텍스트 분류기 구축

웹 애플리케이션: 사용자가 텍스트 파일을 업로드하면 장르를 예측하고 워드 클라우드를 시각화하여 결과를 제공하는 웹 UI

기술 스택
개발 언어: Python 3.x

프레임워크: Flask

주요 라이브러리:

데이터 수집: requests, BeautifulSoup

자연어 처리: nltk

머신러닝: scikit-learn

시각화: wordcloud, matplotlib

기타: joblib


시작하기 (Getting Started)
프로젝트를 로컬 환경에서 실행하려면 아래 단계를 따라주세요.

1. 환경 설정 및 의존성 설치
프로젝트를 클론합니다.

git clone [GitHub 저장소 링크]
cd TextGenreClassifier

가상 환경을 만들고 활성화합니다.

python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

필요한 라이브러리를 설치합니다.

pip install Flask scikit-learn nltk wordcloud joblib

NLTK 데이터를 다운로드합니다. 아래 스크립트를 실행하면 필요한 데이터를 모두 다운로드합니다.

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

2. 데이터 수집 및 전처리
preprocess_data.py 스크립트를 실행하여 구텐베르크 프로젝트에서 데이터를 수집하고 전처리합니다.

python preprocess_data.py

3. 모델 학습
train_model.py를 실행하여 모델을 학습시키고 .pkl 파일을 생성합니다.

python train_model.py

4. 웹 애플리케이션 실행
모든 준비가 완료되면 app.py를 실행하여 웹 서버를 시작합니다.

python app.py

브라우저에서 http://127.0.0.1:5000에 접속하여 장르 예측을 시도해 보세요.

라이센스
이 프로젝트는 MIT 라이센스 하에 배포됩니다.
