import re
import pandas as pd
from pyvi import ViTokenizer, ViPosTagger

from utils_pre_process import pre_process_review

# ==================================================
# ===== Hàm xác định nhãn của các từ trong câu =====
# ==================================================
def extract_pos_features(text):
    # Tiền xử lý
    sentence = pre_process_review(text)
    # Tokenize
    token = ViTokenizer.tokenize(sentence)
    # Gắn nhãn từ loại
    words, pos_tags = ViPosTagger.postagging(token)
    # Danh sách kết quả
    subjects = []
    verbs = []
    adjectives = []
    adverbs = []
    punctuation = []
    conjunction = []
    preposition = []
    numeral = []
    # Phân loại theo POS tag
    for word, pos in zip(words, pos_tags):
        if pos in ['N', 'Np', 'Nc', 'Ny', 'X', 'P']:
            subjects.append(word)
        elif pos == 'V':
            verbs.append(word)
        elif pos == 'R':
            adverbs.append(word)
        elif pos == 'A':
            adjectives.append(word)
        elif pos == 'C':
            conjunction.append(word)
        elif pos == 'E':
            preposition.append(word)
        elif pos == 'M':
            numeral.append(word)
        elif pos in ['CH', 'F']:
            punctuation.append(word)
    # -------------------------------
    #  Tạo chuỗi từ + nhãn ví dụ: tôi(P) yêu(V) em(N)
    # -------------------------------
    tagged_sentence = " ".join([f"{w}({p})" for w, p in zip(words, pos_tags)])
    # -------------------------------
    # In kết quả
    # -------------------------------
    print("Câu đã nhập vào:", text)
    print("Câu đã tokenize:", token)
    print("Gắn nhãn từ loại:", tagged_sentence)
    print("Danh sách chủ ngữ:", subjects)
    print("Danh sách động từ:", verbs)
    print("Danh sách trạng từ:", adverbs)
    print("Danh sách tính từ:", adjectives)
    print("Danh sách liên từ:", conjunction)
    print("Danh sách giới từ:", preposition)
    print("Danh sách số từ:", numeral)
    print("Danh sách dấu câu:", punctuation)

# ======================================= #
# ===== TÁCH CÂU PHỨC THÀNH CÂU ĐƠN ===== #
# ======================================= #
def load_word_list(csv_file, column_name):
    df = pd.read_csv(csv_file, encoding='utf-8')
    return (df[column_name].dropna().astype(str).str.strip().tolist())

PRONOUNS = load_word_list("dictionary/pronouns.csv", "pronoun")
conjunctions_raw = load_word_list("dictionary/conjunctions.csv", "conjunction")

conjunctions = []
for c in conjunctions_raw:
    if c == ",":
        conjunctions.append(",")
    else:
        conjunctions.append(f" {c} ")

# Tách câu theo dấu câu
def split_by_punctuation(text: str):
    sentences = re.split(r'[.!?]', text)
    return [s.strip() for s in sentences if s.strip()]

# Tách câu theo liên từ
def split_complex_sentence(sentence):
    parts = [sentence]
    for conj in conjunctions:
        temp = []
        for part in parts:
            temp.extend(part.split(conj))
        parts = temp
    return [p.strip() for p in parts if p.strip()]

# Tách câu theo tính từ
# Giữ lại toàn bộ thông tin tính từ sau tính từ trước đến tính từ sau.
def split_by_adjectives_1(sentences):
    new_sentences = []

    for s in sentences:
        s_tok = ViTokenizer.tokenize(s)
        words, tags = ViPosTagger.postagging(s_tok)

        adj_positions = [i for i, t in enumerate(tags) if t == "A"]

        # Nếu không có hoặc chỉ có 1 tính từ → không tách
        if len(adj_positions) <= 1:
            new_sentences.append(s)
            continue

        first_adj_pos = adj_positions[0]
        prefix = words[:first_adj_pos]

        # Nếu prefix rỗng → KHÔNG THỂ lấy prefix[0]
        if len(prefix) == 0:
            new_sentences.append(s)
            continue

        for i, pos in enumerate(adj_positions):
            adj_word = words[pos]

            if i == 0:
                new_words = prefix + [adj_word]
            else:
                prev_adj_pos = adj_positions[i - 1]
                segment = words[prev_adj_pos + 1 : pos + 1]

                # Tiếp tục fix trường hợp segment cũng rỗng
                if len(segment) == 0:
                    continue

                new_words = [prefix[0]] + segment

            new_sentence = " ".join(new_words)
            new_sentences.append(new_sentence)

    return new_sentences

# Tách câu theo tính từ
# Giữ lại động từ (V), trạng từ (R), danh từ (N/Np/Nc), tính từ (A)
def split_by_adjectives_2(sentences):
    new_sentences = []

    for s in sentences:
        s_tok = ViTokenizer.tokenize(s)
        words, tags = ViPosTagger.postagging(s_tok)

        # tìm vị trí tính từ
        adj_positions = [i for i, t in enumerate(tags) if t == "A"]

        if len(adj_positions) <= 1:
            new_sentences.append(s)
            continue

        # prefix = phần trước tính từ đầu tiên
        first_adj_pos = adj_positions[0]
        prefix = words[:first_adj_pos]

        # ===== Tính từ 1 =====
        new_sentences.append(" ".join(prefix + [words[first_adj_pos]]))

        # ===== Từ tính từ thứ 2 trở đi =====
        for idx in range(1, len(adj_positions)):
            cur_pos = adj_positions[idx]
            prev_pos = adj_positions[idx - 1]

            subject = prefix[0]  # chủ ngữ

            # đoạn lấy: (prev_pos + 1) → cur_pos
            seg_words = words[prev_pos + 1 : cur_pos + 1]
            seg_tags = tags[prev_pos + 1 : cur_pos + 1]

            # giữ lại động từ (V), trạng từ (R), danh từ (N/Np/Nc), tính từ (A)
            allowed = {"V", "R", "N", "Np", "Nc", "A"}
            filtered = [w for w, t in zip(seg_words, seg_tags) if t in allowed]

            # nếu lọc rỗng thì giữ nguyên tính từ hiện tại
            if not filtered:
                filtered = [words[cur_pos]]

            new_sentence = " ".join([subject] + filtered)
            new_sentences.append(new_sentence)

    return new_sentences

# Thêm chủ ngữ cho câu
def add_subjects(sentences):
    new_sentences = []
    last_subject = None

    for s in sentences:
        s_tokenized = ViTokenizer.tokenize(s)
        words, tags = ViPosTagger.postagging(s_tokenized)

        subject_in_sentence = None
        for i, (word, tag) in enumerate(zip(words, tags)):
            if word in PRONOUNS:
                subject_in_sentence = word
                break

            if tag in ["Np", "N", "Nc", "Ny", "X", "P"]:
                subject_in_sentence = word
                break

        if subject_in_sentence and words[0] == subject_in_sentence:
            last_subject = subject_in_sentence
            new_sentences.append(s)
        else:
            if last_subject:
                s = last_subject + " " + s
            new_sentences.append(s)

    return new_sentences

# ===== Tách một câu phức thành câu đơn ===== #
def split_sentences(text):
    sentence = pre_process_review(text)
    sentences_raw = split_by_punctuation(sentence)

    simple_sentences = []
    for s in sentences_raw:
        simple_sentences.extend(split_complex_sentence(s))

    simple_sentences = add_subjects(simple_sentences)
    simple_sentences = split_by_adjectives_1(simple_sentences)
    return simple_sentences

# ===== Tách câu phức từ file CSV thành câu đơn và lưu kết quả vào file CSV ===== #
def split_sentences_csv(input_path, output_path):
    df = pd.read_csv(input_path)
    results = []

    for idx, row in df.iterrows():
        review_text = str(row["review_vi"]).strip()

        # Thêm review gốc
        results.append({
            "review_vi": review_text,
            "note": f"review gốc {idx+1}"
        })

        # Tách câu đơn
        processed_sentences = split_sentences(review_text)

        # Thêm từng câu
        for i, sent in enumerate(processed_sentences, start=1):
            results.append({
                "review_vi": sent,
                "note": f"review gốc {idx+1} - câu {i}"
            })

    # Xuất CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
