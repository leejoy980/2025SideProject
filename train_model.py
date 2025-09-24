import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_evaluate_model(csv_path='processed_books.csv'):
    """
    전처리된 CSV 파일에서 데이터를 로드하고 모델을 학습시킨 후 평가합니다.
    Args:
        csv_path (str): 전처리된 데이터가 저장된 CSV 파일 경로.
    """
    try:
        # 데이터 로드
        df = pd.read_csv(csv_path)
        
        # 데이터가 비어 있는지 확인
        if df.empty:
            print("오류: 데이터프레임이 비어 있습니다. 전처리 단계에서 문제가 발생했을 수 있습니다.")
            return

        print(f"총 {len(df)}개의 데이터 포인트가 로드되었습니다.")
        
        # 특성과 레이블 분리
        X = df['text']
        y = df['label']

        # 학습 데이터와 테스트 데이터로 분리 (80:20 비율)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"\n학습 데이터 크기: {len(X_train)}")
        print(f"테스트 데이터 크기: {len(X_test)}")
        
        # 텍스트 데이터를 TF-IDF 벡터로 변환
        print("\nTF-IDF 벡터화 진행 중...")
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        print("TF-IDF 벡터화 완료.")

        # 로지스틱 회귀 모델 학습
        print("\n로지스틱 회귀 모델 학습 중...")
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X_train_vec, y_train)
        print("모델 학습 완료.")

        # 모델 예측 및 평가
        print("\n모델 성능 평가 중...")
        y_pred = model.predict(X_test_vec)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"\n모델 정확도: {accuracy:.4f}")
        print("\n분류 리포트:")
        print(report)
        
        # 학습된 모델과 벡터라이저 저장
        joblib.dump(model, 'genre_classifier_model.pkl')
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
        
        print("\n모델과 벡터라이저가 'genre_classifier_model.pkl' 및 'tfidf_vectorizer.pkl' 파일로 저장되었습니다.")

    except FileNotFoundError:
        print(f"오류: '{csv_path}' 파일을 찾을 수 없습니다. 데이터 전처리 단계를 다시 확인해 주세요.")
    except Exception as e:
        print(f"모델 학습 및 평가 중 오류 발생: {e}")

if __name__ == "__main__":
    train_and_evaluate_model()