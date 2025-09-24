import nltk

print("NLTK 데이터 다운로드를 시작합니다...")
# 기존에 다운로드했던 리소스들
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# 새로 추가된 리소스
nltk.download('punkt_tab')

print("모든 NLTK 데이터 다운로드가 완료되었습니다.")