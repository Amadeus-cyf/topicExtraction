import nltk
import getTranscriptions as gt 

def readTextToCorpus(text):
    return nltk.word_tokenize(text)

def removeWordsLessThan3(corpus):
    filterCorpus = []
    for word in corpus:
        if len(word) > 3:
            filterCorpus.append(word)
    return filterCorpus

def main():
    text = gt.getTranscriptionById('5b3ee68c-5407-4a1f-a330-e55d5d239ffb')
    corpus = readTextToCorpus(text)
    corpus = removeWordsLessThan3(corpus)

if __name__ == '__main__':
    main()