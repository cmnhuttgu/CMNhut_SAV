import os
import joblib
import pandas as pd

# ============================================ #
# ===== Dịch và lưu Review_EN, Review_VI ===== #
# ============================================ #
from utils_pre_process import translate_csv
def translate_csv_to_csv(input_csv, output_csv):

    first_write = True  # để ghi header 1 lần
    n = 0
    for chunk in pd.read_csv(input_csv, chunksize=1):
        n += 1
        chunk["review_en"] = chunk["review_org"].apply(
            lambda x: translate_csv(x, "vi", "en")
        )
        chunk["review_vi"] = chunk["review_en"].apply(
            lambda x: translate_csv(x, "en", "vi")
        )

        # Ghi ngay dòng vừa xử lý
        chunk.to_csv(
            output_csv,
            mode="w" if first_write else "a",
            index=False,
            header=first_write,
            encoding="utf-8-sig"
        )

        first_write = False
        print("✔ Đã ghi xong dòng thứ: ", n)

    print("✅ Hoàn thành toàn bộ file!")

# =========================================================== #
# ===== Lưu Review + Sentiment vào CSV theo thứ tự file ===== #
# =========================================================== #
def save_review_to_csv(review, sentiment, file, thu_tu=1):
    # Tìm file CSV chưa tồn tại
    while True:
        ten_file = f"{file}_{thu_tu}.csv"
        if not os.path.exists(ten_file):
            break
        thu_tu += 1
    # Lưu tên file CSV vào joblib
    joblib.dump(ten_file, "input/review_fname.joblib")
    # Tạo dataframe
    df = pd.DataFrame({'review_vi': [review], 'sentiment': [sentiment]})
    # Ghi CSV
    df.to_csv(ten_file, index=False, encoding="utf-8-sig")

# =========================================================== #
# ===== Lưu Review + Sentiment vào CSV theo thứ tự file ===== #
# =========================================================== #
def save_review_csv_to_csv(review, sentiment, file, thu_tu=1):
    # Tìm file CSV chưa tồn tại
    while True:
        ten_file = f"{file}_{thu_tu}.csv"
        if not os.path.exists(ten_file):
            break
        thu_tu += 1
    # Lưu tên file CSV vào joblib
    joblib.dump(ten_file, "input/review_fname.joblib")
    # Tạo dataframe
    df = pd.DataFrame({'review_vi': review, 'sentiment': sentiment})
    # Ghi CSV
    df.to_csv(ten_file, index=False, encoding="utf-8-sig")

# ======================================================= #
# ===== Lưu All Review + Sentiment vào CSV tổng hợp ===== #
# ======================================================= #
def save_all_reviews_to_csv(review, sentiment, file):
    # Tạo dataframe
    df = pd.DataFrame({'review_vi': [review], 'sentiment': [sentiment]})
    # Ghi CSV. Nếu file đã tồn tại thì chỉ append thêm, không ghi header
    if os.path.exists(file):
        df.to_csv(file, mode='a', header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(file, index=False, encoding="utf-8-sig")

# =================================================================== #
# ===== Lưu Sup Review + Sup Sentiment vào CSV theo thứ tự file ===== #
# =================================================================== #
def save_sup_review_to_csv(review, sentiment, file, thu_tu=1):
    # Tìm file CSV chưa tồn tại
    while True:
        ten_file = f"{file}_{thu_tu}.csv"
        if not os.path.exists(ten_file):
            break
        thu_tu += 1
    # Lưu tên file bằng joblib
    joblib.dump(ten_file, 'input/sup_review_fname.joblib')
    # Tạo DataFrame
    df = pd.DataFrame({'review_vi': review, 'sentiment': sentiment})
    # Ghi CSV
    df.to_csv(ten_file, index=False, encoding='utf-8-sig')

# ================================================================================================ #
# ===== Lưu All Review + All Sup Review + All Sentiment + All Sup Sentiment vào CSV tổng hợp ===== #
# ================================================================================================ #
def save_all_reviews_and_sup_review_to_csv(review, sentiment, note, file):
     # Tạo DataFrame
    df = pd.DataFrame({'review_vi': [review], 'sentiment': [sentiment], 'ghi chú': [note]})
    # Ghi CSV. Nếu file đã tồn tại thì chỉ append thêm, không ghi header
    if os.path.exists(file):
        df.to_csv(file, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(file, index=False, encoding='utf-8-sig')
