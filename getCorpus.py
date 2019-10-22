import getTranscriptions as gt


def read_text_to_corpus(text):
    words = text.split(' ')
    return words


def main():
    text = gt.get_transcription_by_id('5b3ee68c-5407-4a1f-a330-e55d5d239ffb')
    corpus = read_text_to_corpus(text)


if __name__ == '__main__':
    main()