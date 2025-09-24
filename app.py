import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from flask import Flask, render_template, request, jsonify
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# NLTK 데이터 다운로드 (최초 1회만 실행)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

app = Flask(__name__)

# --- 전처리 함수 (이전 코드와 동일) ---
def clean_text(text):
    text = re.sub(r'\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*', '', text, flags=re.DOTALL)
    text = re.sub(r'\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*', '', text, flags=re.DOTALL)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text_for_prediction(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    cleaned_text = clean_text(text)
    tokens = nltk.word_tokenize(cleaned_text)
    
    processed_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]
    
    return ' '.join(processed_tokens)

# --- 워드 클라우드 생성 함수 추가 ---
def generate_wordcloud_image(text):
    if not text.strip(): # 텍스트가 비어있으면 워드클라우드 생성 불가
        return None
        
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # 워드클라우드를 이미지로 변환하여 Base64로 인코딩
    img_buffer = BytesIO()
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close() # Matplotlib 경고 방지
    
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return img_base64

# --- 모델 및 벡터라이저 로드 ---
try:
    model = joblib.load('genre_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("모델과 벡터라이저를 성공적으로 로드했습니다.")
except FileNotFoundError:
    print("오류: 모델 파일이 없습니다. train_model.py를 먼저 실행하세요.")
    model = None
    vectorizer = None

# --- 웹 페이지 라우트 ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': '모델 파일이 로드되지 않았습니다.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': '파일이 제공되지 않았습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400

    if file:
        try:
            content = file.read().decode('utf-8')
            preprocessed_text = preprocess_text_for_prediction(content)
            
            if not preprocessed_text:
                return jsonify({'error': '파일 내용이 비어 있거나 전처리 후 텍스트가 남지 않았습니다.'}), 400
                
            text_vector = vectorizer.transform([preprocessed_text])
            prediction = model.predict(text_vector)[0]
            
            # 워드 클라우드 이미지 생성 (Base64 인코딩)
            wordcloud_img_base64 = generate_wordcloud_image(preprocessed_text)
            
            return jsonify({'genre': prediction, 'wordcloud': wordcloud_img_base64})

        except Exception as e:
            return jsonify({'error': f'파일 처리 중 오류 발생: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)