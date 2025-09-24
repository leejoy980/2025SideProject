import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

# NLTK 데이터가 다운로드되지 않았다면 주석 해제 후 실행
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def clean_text(text):
    """모델 학습에 사용된 전처리 함수와 동일"""
    text = re.sub(r'\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*', '', text, flags=re.DOTALL)
    text = re.sub(r'\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*', '', text, flags=re.DOTALL)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text_for_prediction(text):
    """예측을 위한 텍스트 전처리 및 토큰화"""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    cleaned_text = clean_text(text)
    tokens = nltk.word_tokenize(cleaned_text)
    
    processed_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]
    
    return ' '.join(processed_tokens)

def predict_genre(text_to_predict, model, vectorizer):
    """
    단일 텍스트에 대한 장르를 예측합니다.
    Args:
        text_to_predict (str): 예측할 텍스트.
        model: 학습된 머신러닝 모델 (joblib.load).
        vectorizer: 학습된 TF-IDF 벡터라이저 (joblib.load).
    Returns:
        str: 예측된 장르.
    """
    # 텍스트 전처리
    preprocessed_text = preprocess_text_for_prediction(text_to_predict)
    
    # 벡터화 (transform 사용)
    text_vector = vectorizer.transform([preprocessed_text])
    
    # 예측
    prediction = model.predict(text_vector)
    
    return prediction[0]

if __name__ == "__main__":
    # 1. 저장된 모델과 벡터라이저 불러오기
    try:
        model = joblib.load('genre_classifier_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except FileNotFoundError:
        print("오류: 'genre_classifier_model.pkl' 또는 'tfidf_vectorizer.pkl' 파일을 찾을 수 없습니다.")
        print("모델 학습(train_model.py)을 먼저 실행하여 파일을 생성해야 합니다.")
        exit()

    # 2. 예측할 텍스트 파일이 있는 폴더 경로 설정
    input_dir = 'text'
    if not os.path.exists(input_dir):
        print(f"오류: '{input_dir}' 폴더를 찾을 수 없습니다. 해당 폴더를 생성하고 예측할 파일을 넣어주세요.")
        exit()

    # 3. 폴더 내 모든 .txt 파일 순회
    print(f"\n--- '{input_dir}' 폴더 내 파일 장르 예측 시작 ---")
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 4. 장르 예측
                predicted_genre = predict_genre(content, model, vectorizer)
                
                # 5. 결과 출력
                print(f"파일: '{filename}' -> 예측된 장르: {predicted_genre}")
                
            except Exception as e:
                print(f"'{filename}' 파일 처리 중 오류 발생: {e}")
    
    print("\n--- 장르 예측 완료 ---")