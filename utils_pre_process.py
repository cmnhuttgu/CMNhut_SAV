import re
import csv
import emoji
import pandas as pd
from deep_translator import GoogleTranslator

errorWord_dict = pd.read_csv('dictionary/my_error_word.csv')
token_dict = pd.read_csv('dictionary/my_token_word.csv')
stopWord_dict = pd.read_csv('dictionary/my_stop_word.csv')

# ================================== #
# ===== Hàm chuyển ngữ dữ liệu ===== #
# ================================== #
def translate_csv(text, src, dest):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    try:
        translated = GoogleTranslator(source=src, target=dest).translate(text)
        return translated if translated else text
    except Exception as e:
        print(f"Lỗi dịch: {text} | {e}")
        return text
    
# ============================= #
# ===== Chuẩn hóa dữ liệu ===== #
# ============================= #
def change_word(tweet):
    tweet = tweet.lower()
    tweet = tweet.replace("http : / / ", 'http://')
    tweet = tweet.replace("class = ' ", 'class=')
    tweet = tweet.replace("href = ' / ", 'href=/')

    tweet = re.sub(r'(http\S+)|(class\S+)|(href\S+)|(RT\S+)|(@\S+)|(\#\S+)', ' ', tweet)

    tweet = tweet.replace(':', ' ')
    tweet = tweet.replace('_', ' ')
    tweet = emoji.demojize(tweet)
    tweet = re.sub(r'(:[a-zA-Z0-9_-]+:)', r' \1 ', tweet)

    tweet = tweet.replace('-', '_')
    tweet = tweet.replace(':', '_')

    # Chuẩn hóa từ theo tự điển
    for index, row in errorWord_dict.iterrows():
        if row['incorrect_Word'] in tweet:
            tweet  = tweet.replace(row['incorrect_Word'], row['correct_Word'])

    # Token theo tự điển
    for index, row in token_dict.iterrows():
        if row['Word'] in tweet:
            tweet  = tweet.replace(row['Word'], row['Token'])

    tweet = tweet.replace('__', ' _')
    return tweet

# =========================== #
# ===== Xóa từ vô nghĩa ===== #
# =========================== #
def remove_stopWord(tweet):
    # Chuẩn hóa từ theo tự điển
    for index, row in stopWord_dict.iterrows():
        if row['stop_Word'] in tweet:
            tweet  = tweet.replace(row['stop_Word'], ' ')

    return tweet

# =================================== #
# ===== Tiền xử lý 01 tweet ===== #
# =================================== #
def pre_process_review(tweet):
    tweet = change_word(tweet)
    tweet = remove_stopWord(tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip() # xóa khoảng trắng thừa
    return tweet

# ================================== #
# ===== Tiền xử lý 01 file CSV ===== #
# ================================== #
def pre_process_csv(input_file, output_file):
    try:
        data = pd.read_csv(input_file, encoding='utf-8')
    except FileNotFoundError:
        print("File input không tồn tại!")
        return

    print("Input length:", len(data))

    # Mở file output một lần
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Ghi header
        writer.writerow(['review_vi', 'review_pre', 'sentiment'])
        n = 0
        for index, row in data.iterrows():
            n += 1
            review_org = row['review_vi']
            review_pre = pre_process_review(review_org)
            sentiment = row['sentiment']

            # Ghi ngay dòng vừa xử lý
            writer.writerow([review_org, review_pre, sentiment])

            print("✔ Đã ghi xong dòng thứ: ", n)
            
        print("✅ Hoàn thành toàn bộ file!")
