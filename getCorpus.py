import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from getTranscriptions import get_transcription_by_id

def read_process_text_to_corpus(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filterd_words = []
    for word in words:
        if word not in stop_words and (len(word) >= 3 or (word.isupper() and len(word) > 1)) and "'" not in word:
            filterd_words.append(word)
    return filterd_words
