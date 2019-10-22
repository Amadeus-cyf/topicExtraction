from topicModel import TopicModel
from background import Background
from getTranscriptions import get_transcription_by_id
from getCorpus import read_process_text_to_corpus


class Extraction:
    topic_model = None
    background_model = None
    word_topic_prob = {}       # P(w|topic_model)
    hidden_word_prob = {}      # P(z|w)
    prob_z_0 = 0.0             # topic model
    prob_z_1 = 0.0             # background model
    topic_prob = 0.6         # probability of using topic model, 1-topic_prob is probability of using background model

    # set the background model of the lda
    def set_background(self):
        transcriptions = get_transcription_by_id("b0c92d50-f05e-4513-bf48-995a2b995850")
        transcriptions2 = get_transcription_by_id("c85c8b74-4dd9-42c2-914b-90494fde86f8")
        #transcriptions3 = get_transcription_by_id("")
        transcription_tokens = read_process_text_to_corpus(transcriptions)
        transcription_tokens += read_process_text_to_corpus(transcriptions2)
        #transcription_tokens += read_process_text_to_corpus(transcriptions3)
        self.background_model = Background()
        self.background_model.add_word_to_background(transcription_tokens)

    def set_topic_model(self):
        transcriptions = get_transcription_by_id("3b653f0c-c210-4f13-a615-5d87a823e96e")
        transcription_tokens = read_process_text_to_corpus(transcriptions)
        self.topic_model = TopicModel()
        self.topic_model.create_model(transcription_tokens, self.background_model)
        print(self.topic_model.word_map)

    def init_word_topic_prob(self):
        for word in self.topic_model.word_map.keys():
            self.word_topic_prob[word] = self.topic_model.word_map[word] / self.topic_model.size

    def init_hidden_topic_prob(self):
        for word in self.topic_model.word_map.keys():
            self.hidden_word_prob[word] = 0.0

    # return p(z=0|w), which is stored in hidden_word_prob
    def expectation(self, word):
        if word not in self.background_model.word_map:
            background_prob = 0.0001
        else:
            background_prob = self.background_model.word_map[word] / self.background_model.size
        self.hidden_word_prob[word] = (self.topic_prob * self.word_topic_prob[word]) / (self.topic_prob * self.word_topic_prob[word] + (1-self.topic_prob) * background_prob)

    # return p(word|topic), which is stored in topic_word_prob
    def maximization(self, word):
        total = 0.0
        for wo in self.hidden_word_prob:
            total += self.topic_model.word_map[wo] * self.hidden_word_prob[wo]
        self.word_topic_prob[word] = self.topic_model.word_map[word] * self.hidden_word_prob[word] / total

    # maximum likelihood estimate
    def maximum_estimate(self, current, gap, word):
        prev = 0.0
        words = self.topic_model.word_map.keys()
        while abs(current-prev) >= gap:
            for wo in words:
                self.expectation(wo)
            for wo in words:
                self.maximization(wo)
            prev = current
            current = self.word_topic_prob[word]
        return self.word_topic_prob[word]

    def top_prob_words(self, num):
        sorted_map = sorted(self.word_topic_prob.items(), key=lambda x: x[1], reverse=True)
        topics = []
        for word in sorted_map:
            topics.append(word)
            num -= 1
            if num <= 0:
                break
        return topics


# testing
if __name__ == "__main__":
    extract = Extraction()
    extract.set_background()
    extract.set_topic_model()
    extract.init_word_topic_prob()
    extract.init_hidden_topic_prob()
    word_prob_map = {}
    for w in extract.topic_model.word_map:
        init = extract.topic_model.word_map[w] / extract.topic_model.size
        word_prob_map[w] = extract.maximum_estimate(init, 0.00000001, w)
        extract.init_word_topic_prob()
        extract.init_hidden_topic_prob()
    sorted_prob = sorted(word_prob_map.items(), key=lambda x: x[1], reverse=True)
    print(sorted_prob[:10])
