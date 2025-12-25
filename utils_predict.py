import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM, SimpleRNN, Bidirectional, GRU, Conv1D, GlobalMaxPooling1D

# =================================== #
# ===== Huấn luyện mô hình LSTM ===== #
# =================================== #
def train_LSTM_My_SAV(X_train, X_test, y_train, y_test,
    max_words=10000,
    embedding_dim=100,
    max_seq_length=100,
    lstm_units=64, # 128 đối với dữ liệu trên 22k dòng
    dense_units=64,
    learning_rate=0.0001,
    batch_size=32, # 64 đối với dữ liệu trên 22k dòng
    epochs=100,
    runs=1,
    accuracy_save=94.80,
    save_dir="model/My_LSTM_"):
    
    sl = 0  # số lần trên 95%

    for i in range(1, runs + 1):
        # Build model
        model = Sequential()
        model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_seq_length, trainable=True, mask_zero=True))
        model.add(LSTM(lstm_units))
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=learning_rate)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        print(f"\n🔄 Chạy lần thứ {i}/{runs} ...")

        history = model.fit(X_train, y_train,
            validation_split=0.1,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping], verbose=0)

        score = model.evaluate(X_test, y_test, verbose=0)
        accuracy_percentage = score[1] * 100

        print(f"Lượt chạy lần thứ: {i}")
        print(f"🎯 Accuracy: {accuracy_percentage:.2f}%")

        # Nếu accuracy ≥ threshold thì lưu
        if accuracy_percentage >= accuracy_save:
            sl += 1
            os.makedirs("model/", exist_ok=True)

            score_path = f"{save_dir}score_{accuracy_percentage:.2f}.npy"
            np.save(score_path, score)
            model_path = f"{save_dir}model_{accuracy_percentage:.2f}.h5"
            model.save(model_path)
            history_path = f"{save_dir}history_{accuracy_percentage:.2f}.pkl"
            save_history(history, history_path)
            print(f"📁 Đã lưu model đạt {accuracy_percentage:.2f}% vào thư mục.")

    tl = (sl / runs) * 100
    print(f"\n📌 Kết quả cuối cùng:")
    print(f"Số lần đạt độ chính xác ≥ {accuracy_save}%: {sl}/{runs}")
    print(f"Tỉ lệ: {tl:.2f}%")

# ============================================== #
# ===== Hàm lưu history huấn luyện mô hình ===== #
# ============================================== #
# Hàm lưu history vào các tập tin
def save_history(history, history_filename):
    with open(history_filename, 'wb') as file:
        pickle.dump(history, file)

# =============================================== #
# ===== Hàm load history huấn luyện mô hình ===== #
# =============================================== #
def load_history(history_filename):
    with open(history_filename, 'rb') as file:
        history = pickle.load(file)
    return history

# ============================================= #
# ===== Hàm vẽ biểu đồ huấn luyện mô hình ===== #
# ============================================= #
import matplotlib.pyplot as plt
import os

def draw_chart(model_name, history, score, save_dir="charts"):
    os.makedirs(save_dir, exist_ok=True)

    # Đánh giá mô hình
    test_loss = score[0]
    test_accuracy = score[1]
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    # ===================== #
    # 1️⃣ BIỂU ĐỒ TRAIN / VAL RIÊNG
    # ===================== #
    plt.figure(figsize=(12, 4))

    # Train
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.title('Tập Train')
    plt.xlabel('Epoch')
    plt.legend()

    # Validation
    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Tập Validation')
    plt.xlabel('Epoch')
    plt.legend()

    file1 = f"{save_dir}/train_val_overview.png"
    plt.savefig(file1, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📁 Đã lưu file train_val_overview.png")

    # ===================== #
    # 2️⃣ BIỂU ĐỒ SO SÁNH LOSS / ACC
    # ===================== #
    plt.figure(figsize=(12, 4))

    # Loss comparison
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.legend()

    # Accuracy comparison
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.legend()

    file2 = f"{save_dir}/loss_accuracy_comparison.png"
    plt.savefig(file2, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"📁 Đã lưu file loss_accuracy_comparison.png")

def draw_chart_1(model, history, score):
    # Đánh giá mô hình trên tập kiểm thử để lấy thông tin về loss và accuracy
    test_loss = score[0]
    test_accuracy = score[1]

    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    # Vẽ biểu đồ loss và accuracy từ history
    plt.figure(figsize=(12, 4))

    # Biểu đồ loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['accuracy'], label='Accuracy')

    plt.title('Tập Train')
    plt.xlabel('Epoch')
    plt.legend()

    # Biểu đồ loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')

    plt.title('Tập Val')
    plt.xlabel('Epoch')
    plt.legend()

    # Hiển thị biểu đồ
    plt.show()

    # Vẽ biểu đồ loss và accuracy từ history
    plt.figure(figsize=(12, 4))

    # Biểu đồ loss train
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.legend()

    # Biểu đồ accuracy train
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.legend()

    # Hiển thị biểu đồ
    plt.show()

# ========================================== #
# ===== Hàm dự đoán cảm xúc một review ===== #
# ========================================== #
# * max_words = 10.000 (tốt nhất. Cấu hình máy tính không đáp ứng nên chọn 2500)
# - Tokenzier sẽ chỉ giữ lại 10.000 từ xuất hiện nhiều nhất trong toàn bộ dữ liệu.
# - Tất cả từ còn lại → được gán thành token OOV (Out-Of-Vocabulary).

# * max_seq_length = 100
# - Độ dài tối đa của mỗi câu (chuỗi số) khi đưa vào LSTM.
# - Mỗi câu sau khi chuyển thành chuỗi số có độ dài không được vượt quá 100 token.
# - Nếu câu quá dài → cắt bớt từ đầu hoặc cuối để còn 100.
# - Nếu câu quá ngắn → pad thêm số 0 cho đủ 100.
# ============================================================================ #
# ===== Với padding="post" và truncating="post" ===== #
# ===== Bắt buộc khi Embedding huấn luyện mô hình phải có mask_zero=True ===== #
# ============================================================================ #
def process_X_token_review(new_review, vt='post', tokenizer_path = "input/tokenizer.joblib"):
    max_seq_length = 100

    # 1️⃣ Load hoặc tạo tokenizer
    if os.path.exists(tokenizer_path):
        tokenizer = joblib.load(tokenizer_path)
    else:
        print("Không tìm thấy file tokenizer.joblib")

    # 2️⃣ Text → sequence
    sequences = tokenizer.texts_to_sequences([new_review])

    # 3️⃣ Pad giống lúc train
    X_token = pad_sequences(sequences, maxlen=max_seq_length, padding=vt, truncating=vt)
    return X_token

def process_X_token_csv(review_token, vt='post', tokenizer_path = "input/tokenizer.joblib"):
    max_words = 10000
    max_seq_length = 100

    # 1️⃣ Load hoặc tạo tokenizer
    if os.path.exists(tokenizer_path):
        tokenizer = joblib.load(tokenizer_path)
    else:
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(review_token)
        joblib.dump(tokenizer, tokenizer_path)

    # 2️⃣ Text → sequence
    sequences = tokenizer.texts_to_sequences(review_token)

    # 3️⃣ Pad giống lúc train
    X_token = pad_sequences(sequences, maxlen=max_seq_length, padding=vt, truncating=vt)
    return X_token

def predict_score(model, X_token):
    score = model.predict(X_token)[0]
    return score

def predict_sentiment(score):
    if score > 0.5:
        return 'positive'
    elif score < 0.5:
        return 'negative'
    else:
        return 'neutral'

# =========================================================== #
# ===== Hàm vẽ biểu đồ thống kê cảm xúc sau khi dự đoán ===== #
# =========================================================== #
def plot_sentiment_pie():
    file_path = 'input/all_reviews.csv'
    sentiment_data = pd.read_csv(file_path)

    # Đếm số lần xuất hiện của mỗi nhãn
    label_counts = sentiment_data['sentiment'].value_counts()

    # Vẽ biểu đồ hình tròn thể hiện số lượng và % với labels từ cột "Label"
    sentiment_plt = plt.figure(figsize=(4, 3))
    plt.pie(label_counts, autopct=lambda p: 'SL:{:.0f} \n({:.1f}%)'.format(p * sum(label_counts) / 100, p), startangle=140, textprops={'fontsize': 12})
    #plt.title('Phân phối nhãn trong dữ liệu')
    plt.axis('equal')
    plt.legend(label_counts.index, loc='best')  # Đặt nhãn và vị trí tốt nhất

    # Lưu biểu đồ dưới dạng file jpg
    sentiment_plt.savefig('static/sentiment_plt.png')

# ===================== #
# ===== WordCloud ===== #
# ===================== #
def generate_wordcloud(
    X_token='input/My_review_token.joblib',
    sentiment='input/My_sentiment.joblib',
    save_dir='output/wordclouds',
    dpi=300
):
    tweets = joblib.load(X_token)      # list of list
    labels = joblib.load(sentiment)    # Series or list

    positive_text = []
    negative_text = []

    for tweet, label in zip(tweets, labels):
        tweet_text = ' '.join(tweet)   # list → string
        if label == 'positive':
            positive_text.append(tweet_text)
        elif label == 'negative':
            negative_text.append(tweet_text)

    positive_wordcloud = WordCloud(
        width=800, height=400, background_color='white'
    ).generate(' '.join(positive_text))

    negative_wordcloud = WordCloud(
        width=800, height=400, background_color='black', colormap='Reds'
    ).generate(' '.join(negative_text))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.title('Positive Sentiment')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(negative_wordcloud, interpolation='bilinear')
    plt.title('Negative Sentiment')
    plt.axis('off')

    # ====== LƯU FILE PNG ======
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'wordcloud.png')
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f'Đã lưu wordcloud tại: {filepath}')

    plt.show()
    plt.close()

# =========================== #
# ===== WordCloud Token ===== #
# =========================== #
def plot_word_occurrences(
    word,
    X_token='input/My_review_token.joblib',
    sentiment='input/My_sentiment.joblib',
    save_dir='output/charts',
    dpi=300
):
    tweets = joblib.load(X_token)
    labels = joblib.load(sentiment)

    word = word.lower()

    occurrences_positive = 0
    occurrences_negative = 0

    for tweet, label in zip(tweets, labels):
        tweet_text = ' '.join(tweet).lower()
        count = tweet_text.count(word)

        if label == 'positive':
            occurrences_positive += count
        elif label == 'negative':
            occurrences_negative += count

    labels_plot = ['Positive', 'Negative']
    counts = [occurrences_positive, occurrences_negative]

    plt.figure(figsize=(6, 4))
    plt.bar(
        labels_plot,
        counts,
        color=['blue', 'red']   # ⭐ Positive: vàng, Negative: đỏ
    )
    plt.title(f'Tần suất của từ "{word}" theo nhãn')

    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')

    # ====== LƯU FILE PNG ======
    os.makedirs(save_dir, exist_ok=True)
    filename = 'word_occurrence.png'
    filepath = os.path.join(save_dir, filename)

    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f'Đã lưu biểu đồ tại: {filepath}')

    plt.show()
    plt.close()
