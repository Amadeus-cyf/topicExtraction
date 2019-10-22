from topicModel import TopicModel
from background import Background
from getTranscriptions import get_transcription_by_id
from getCorpus import read_text_to_corpus


# set the background model of the lda
def set_background():
    transcriptions = get_transcription_by_id("50d6556d-cf5a-4f3e-a891-651d1f271d16")
    transcriptions2 = get_transcription_by_id("cfb96003-5aa7-480a-a4d1-16983e4e7a37")
    transcription_tokens = read_text_to_corpus(transcriptions)
    transcription_tokens += read_text_to_corpus(transcriptions2)
    background_model = Background()
    background_model.add_word_to_background(transcription_tokens)
    return background_model


def set_topic_model():
    background = set_background()
    transcriptions = get_transcription_by_id("c0bfea1c-8550-4dcf-a04c-0c4f3d6d76e2")
    transcription_tokens = read_text_to_corpus(transcriptions)
    topic_model = TopicModel()
    topic_model.create_model(transcription_tokens, background)
    print(topic_model.get_topic_words(20))



if __name__ == "__main__":
    set_topic_model()
