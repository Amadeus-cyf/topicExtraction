# tokenize transcriptions and remove trivial words (words with length < 3 and combining words)
def read_process_text_to_corpus(text):
    words = read_raw_text_to_corpus(text)
    processed_word = []
    com = "'"
    for word in words:
        if len(word) < 3 or com in word:
            continue
        processed_word.append(word)
    return processed_word


def read_raw_text_to_corpus(text):
    words = text.split(' ')
    return words
