import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK 데이터 다운로드 (최초 1회만 실행)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def clean_text(text):
    """텍스트에서 불필요한 부분(헤더, 푸터, 특수문자 등)을 제거합니다."""
    # 구텐베르크 프로젝트 헤더/푸터 제거 (정규식 사용)
    text = re.sub(r'\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*', '', text, flags=re.DOTALL)
    text = re.sub(r'\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*', '', text, flags=re.DOTALL)
    text = re.sub(r'[\r\n]+', ' ', text)  # 개행문자를 공백으로 변환

    # 소문자 변환
    text = text.lower()

    # 알파벳만 남기고 나머지 문자 제거
    text = re.sub(r'[^a-z\s]', '', text)

    # 연속된 공백을 하나의 공백으로 축소
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_and_load_data(data_path='data_by_genre'):
    """
    지정된 경로에서 장르별 텍스트 데이터를 읽고 전처리하여 DataFrame으로 반환합니다.
    Args:
        data_path (str): 책 파일이 저장된 기본 경로.
    Returns:
        pd.DataFrame: 'text'와 'label' 컬럼을 가진 데이터프레임.
    """
    data = []
    
    # 불용어 리스트
    stop_words = set(stopwords.words('english'))
    # 표제어 추출기
    lemmatizer = WordNetLemmatizer()

    # 기본 경로 내의 모든 장르 폴더 순회
    for genre_folder in os.listdir(data_path):
        genre_path = os.path.join(data_path, genre_folder)
        if os.path.isdir(genre_path):
            print(f"'{genre_folder}' 장르 데이터 전처리 중...")
            
            # 장르 폴더 내의 모든 .txt 파일 순회
            for filename in os.listdir(genre_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(genre_path, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        
                        # 1. 텍스트 클렌징
                        cleaned_text = clean_text(text)
                        
                        # 2. 토큰화 (단어 단위로 분리)
                        tokens = nltk.word_tokenize(cleaned_text)
                        
                        # 3. 불용어 제거 및 표제어 추출
                        processed_tokens = [
                            lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
                        ]
                        
                        # 4. 다시 문자열로 결합
                        final_text = ' '.join(processed_tokens)
                        
                        if final_text:
                            data.append({
                                'text': final_text,
                                'label': genre_folder
                            })
                            
                    except Exception as e:
                        print(f"'{file_path}' 파일 처리 중 오류 발생: {e}")
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # NLTK 데이터가 다운로드되지 않았다면 주석 해제 후 실행
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    
    processed_df = preprocess_and_load_data()
    print("\n--- 데이터 전처리 완료 ---")
    print(f"최종 데이터셋 크기: {len(processed_df)}개")
    print("데이터셋 미리보기:")
    print(processed_df.head())

    # 전처리된 데이터를 CSV 파일로 저장
    processed_df.to_csv('processed_books.csv', index=False)
    print("\n전처리된 데이터가 'processed_books.csv' 파일로 저장되었습니다.")