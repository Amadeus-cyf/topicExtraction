from topicModel import TopicModel
from background import Background
from getTranscriptions import get_transcription_by_id
from getCorpus import read_process_text_to_corpus
import random

class Extraction:
    topic_model = None
    background_model = None
    word_topic_prob = {}                    # P(w|topic_model)
    hidden_word_prob = {}                   # P(z|w)
    prob_z_0 = 0.0                          # topic model
    prob_z_1 = 0.0                          # background model
    topic_prob = 0.6                       # probability of using topic model, 1-topic_prob is probability of using background model
    mixture_word_topic_prob = {}            # probability of word in mixture model, map word to prob 
    transcription_id = "b7c99a7c-0223-44f8-ada1-ea538f3f7d54"
    # cs 374 for test

    # set the background model of the lda
    def set_background(self):
        #transcriptions_1 = get_transcription_by_id("fee4abaa-f0c4-427e-a591-46047c95a781")
        transcriptions_2 = get_transcription_by_id("de12a60b-afaf-462a-990b-507ffef68f0b")
        #transcription_tokens_1 = read_process_text_to_corpus(transcriptions_1)
        transcription_tokens_2 = read_process_text_to_corpus(transcriptions_2)
        self.background_model = Background()
        #self.background_model.add_word_to_background(transcription_tokens_1)
        self.background_model.add_word_to_background(transcription_tokens_2)

    # set the topic model
    def set_topic_model(self):
        transcriptions = get_transcription_by_id(self.transcription_id)
        transcription_tokens = read_process_text_to_corpus(transcriptions)
        self.topic_model = TopicModel()
        self.topic_model.create_model(transcription_tokens, self.background_model)
        #print(self.topic_model.word_map)

    def init_word_topic_prob(self):
        for word in self.topic_model.word_map.keys():
            self.word_topic_prob[word] = self.topic_model.word_map[word] / self.topic_model.size
        #print(sorted(self.word_topic_prob.items(), key=lambda x: x[1], reverse=True))

    def init_hidden_topic_prob(self):
        for word in self.topic_model.word_map.keys():
            self.hidden_word_prob[word] = 0.0

    # calculated p(z=0|w), which is stored in hidden_word_prob
    def expectation(self, word):
        if word not in self.background_model.word_map:
            background_prob = 0.0001
        else:
            background_prob = self.background_model.word_map[word] / self.background_model.size
        self.hidden_word_prob[word] = (self.topic_prob * self.word_topic_prob[word]) / (self.topic_prob * self.word_topic_prob[word] + (1-self.topic_prob) * background_prob)

    # calculated p(word|topic_model), which is stored in word_topic_prob
    def maximization(self, word):
        total = 0.0
        for wo in self.hidden_word_prob:
            total += self.topic_model.word_map[wo] * self.hidden_word_prob[wo]
        self.word_topic_prob[word] = self.topic_model.word_map[word] * self.hidden_word_prob[word] / total

    # maximum likelihood estimate of P(w|topic) of given word
    # current is the initial P(z=0|w)
    # gap is the convergence
    def maximum_estimate(self, current, gap, word):
        prev = 0.0
        #words = self.topic_model.word_map.keys()
        while True:
            self.expectation(word)
            self.maximization(word)
            prev = current
            current = self.word_topic_prob[word]
            if current - prev < gap:
                break
        return self.word_topic_prob[word]

    # get top probability words from the topic model
    def top_prob_words(self, num):
        sorted_map = sorted(self.word_topic_prob.items(), key=lambda x: x[1], reverse=True)
        topics = []
        for word in sorted_map:
            topics.append(word)
            num -= 1
            if num <= 0:
                break
        return topics

    # get word prob from mixture model
    def get_word_prob_from_mixture_model(self):
        transcriptions = get_transcription_by_id(self.transcription_id)
        transcription_tokens = read_process_text_to_corpus(transcriptions)
        for word in transcription_tokens:
            prob = 0.0
            # word only appears in background model
            if word not in self.word_topic_prob:
                prob = self.background_model.get_probability(word) * (1-self.topic_prob)
            else:
                prob = self.background_model.get_probability(word) * (1-self.topic_prob) + self.word_topic_prob[word] * self.topic_prob
            if word not in self.mixture_word_topic_prob:
                self.mixture_word_topic_prob[word] = prob
            else:
                if prob > self.mixture_word_topic_prob[word]:
                    self.mixture_word_topic_prob[word] = prob
    
    # get top topics from mixture model
    def get_top_topic(self):
        self.mixture_word_topic_prob= sorted(self.mixture_word_topic_prob.items(), key=lambda x: x[1], reverse=True)
        count = 0
        topics = []
        for key in self.mixture_word_topic_prob:
            topics.append(key[0])
            count+=1
            if count >= 20:
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
    # do the maximum-estimate for each word in the transcirptions
    for w in extract.topic_model.word_map:
        init = extract.topic_model.word_map[w] / extract.topic_model.size
        word_prob_map[w] = extract.maximum_estimate(init, 0.000001, w)
        extract.init_word_topic_prob()
        extract.init_hidden_topic_prob()
    extract.get_word_prob_from_mixture_model()
    topics = extract.get_top_topic()
    #print(sorted(extract.word_topic_prob.items(), key =lambda x: x[1], reverse=True))
    print(topics)