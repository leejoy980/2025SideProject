import requests
from bs4 import BeautifulSoup
import os
import re
import time

# --- 이전에 정의했던 .txt 파일 다운로드 함수 ---
def download_book_text(url, save_path):
    """
    구텐베르크 프로젝트의 순수 텍스트 파일을 다운로드합니다.
    회사 네트워크 환경이라면 verify=False 옵션을 추가해야 합니다.
    """
    try:
        response = requests.get(url, stream=True, verify=False) # verify=False 유지
        response.raise_for_status()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  -> '{os.path.basename(save_path)}' 다운로드 완료.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  -> '{url}' 다운로드 중 오류 발생: {e}")
        return False
    except Exception as e:
        print(f"  -> '{url}' 처리 중 알 수 없는 오류 발생: {e}")
        return False

# --- 수정된 기능: 책장 페이지 크롤링 및 책 ID 추출 ---
def get_book_ids_from_bookshelf(bookshelf_url, num_pages=2):
    """
    구텐베르크 책장(Bookshelf) 페이지에서 책 ID를 추출합니다.
    Args:
        bookshelf_url (str): 책장 페이지의 URL.
        num_pages (int): 몇 페이지까지 탐색할지 (각 페이지당 약 25개 책).
    Returns:
        list: 추출된 책 ID 목록.
    """
    book_ids = set() # 중복 제거를 위해 set 사용

    for i in range(num_pages):
        page_num = i + 1
        page_url = f"{bookshelf_url}?start_index={i * 25 + 1}" # 각 페이지는 25개씩 표시
        print(f"[{time.strftime('%H:%M:%M')}] 책장 페이지 요청 중 ({page_num}/{num_pages}): {page_url}")
        
        try:
            response = requests.get(page_url, verify=False) # verify=False 유지
            response.raise_for_status()
            # print(f"[{time.strftime('%H:%M:%S')}] 페이지 {page_num} 요청 성공. 상태 코드: {response.status_code}") # 디버깅용 로그 제거

            soup = BeautifulSoup(response.text, 'html.parser')

            current_page_book_links_found = 0
            for link_li in soup.find_all('li', class_='booklink'):
                a_tag = link_li.find('a', href=re.compile(r'/ebooks/\d+$'))
                if a_tag and a_tag.get('href'):
                    book_id_match = re.search(r'/ebooks/(\d+)', a_tag['href'])
                    if book_id_match:
                        book_ids.add(book_id_match.group(1))
                        current_page_book_links_found += 1
            
            print(f"[{time.strftime('%H:%M:%M')}] 페이지 {page_num}에서 {current_page_book_links_found}개의 책 링크 추출. 현재까지 총 {len(book_ids)}개 ID 확보.")
            
            next_link_found = False
            # --- 수정된 부분: span 태그와 class='links'를 사용해 '다음' 링크 찾기 ---
            # 'links' 클래스를 가진 span 태그를 찾고, 그 안에 'next'라는 텍스트를 가진 'a' 태그를 찾습니다.
            for nav_span in soup.find_all('span', class_='links'):
                next_a_tag = nav_span.find('a', string=re.compile(r'next', re.IGNORECASE)) # 'next' 텍스트를 대소문자 구분 없이 찾기
                if next_a_tag and next_a_tag.get('href'):
                    next_link_found = True
                    break # 찾았으면 루프 종료
            
            if not next_link_found:
                print(f"[{time.strftime('%H:%M:%M')}] 페이지 {page_num}에서 다음 페이지 링크를 찾을 수 없어 탐색 중단.")
                break # 다음 페이지가 없으면 루프 종료
            
            time.sleep(1) # 서버 부하를 줄이기 위해 페이지 요청 간 딜레이
        
        except requests.exceptions.RequestException as e:
            print(f"[{time.strftime('%H:%M:%M')}] 오류: 책장 페이지 '{page_url}' 요청 중 오류 발생: {e}")
            break
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%M')}] 오류: 책장 페이지 '{page_url}' 파싱 중 알 수 없는 오류 발생: {e}")
            break

    return list(book_ids)

# --- main 부분은 동일하게 유지 ---
if __name__ == "__main__":
    books_to_crawl_per_genre = 200

    genres = {
        "Children & Young Adult Reading": "http://www.gutenberg.org/ebooks/bookshelf/636",
        "Crime, Thrillers & Mystery": "http://www.gutenberg.org/ebooks/bookshelf/640",
        "Law & Criminology": "http://www.gutenberg.org/ebooks/bookshelf/689",
        "Psychialtry/Psychology": "https://www.gutenberg.org/ebooks/bookshelf/688",
        "Science - Physics": "http://www.gutenberg.org/ebooks/bookshelf/667",
        "Economics": "http://www.gutenberg.org/ebooks/bookshelf/696",
        "Travel Writing": "http://www.gutenberg.org/ebooks/bookshelf/648",
        "Art": "http://www.gutenberg.org/ebooks/bookshelf/675",
        "Fashion": "http://www.gutenberg.org/ebooks/bookshelf/676",
        "Novels": "http://www.gutenberg.org/ebooks/bookshelf/645"
    }

    output_base_dir = "data_by_genre"

    for genre, bookshelf_url in genres.items():
        print(f"\n--- {genre} 장르 책 크롤링 시작 (목표: {books_to_crawl_per_genre}권) ---")
        
        genre_output_dir = os.path.join(output_base_dir, genre)
        os.makedirs(genre_output_dir, exist_ok=True)

        num_pages_to_scan = (books_to_crawl_per_genre // 25) + 2 
        book_ids = get_book_ids_from_bookshelf(bookshelf_url, num_pages=num_pages_to_scan)
        print(f"'{genre}' 장르에서 최종 {len(book_ids)}개의 책 ID 추출 완료.")

        downloaded_count = 0
        for book_id in book_ids:
            if downloaded_count >= books_to_crawl_per_genre:
                print(f"'{genre}' 장르의 목표 권수({books_to_crawl_per_genre}권)에 도달하여 중단합니다.")
                break

            txt_url = f"http://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            save_path = os.path.join(genre_output_dir, f"{genre}_{book_id}.txt")
            
            if download_book_text(txt_url, save_path):
                downloaded_count += 1
            
            time.sleep(0.5) 

        print(f"--- {genre} 장르 크롤링 완료. 총 {downloaded_count}권 다운로드. ---")