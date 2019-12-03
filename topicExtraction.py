from topicModel import TopicModel
from background import Background
from getTranscriptions import get_transcription_by_id
from getCorpus import read_process_text_to_corpus
import random

class Extraction:
    document_words = None                 # map word in transcription to count
    vocab = 0
    topic_model_1 = None
    topic_model_2 = None
    topic_model_3 = None
    background_model = None
    background_word_topic_prob = {}          # P(w|B)
    word_topic_prob_1 = {}                    # P(w|topic_model_1)
    hidden_word_prob_1 = {}                   # P(z_0_1|w)
    word_topic_prob_2 = {}                    # P(w|topic_model_2)
    hidden_word_prob_2= {}                   # P(z_0_2|w)
    word_topic_prob_3 = {}                    # P(w|topic_model_3)
    hidden_word_prob_3 = {}                   # P(z_0_3|w)
    prob_z_0_1 = 0.0                          # topic model 1
    prob_z_0_2 = 0.0                          # topic model 2
    prob_z_0_3 = 0.0                          # topic model 3
    prob_z_1 = 0.0                          # background model
    topic_coverage_1 = 1/3                
    topic_coverage_2 = 1/3
    topic_coverage_3 = 1/3
    topic_prob = 0.8                        # total probability of all topic models, 1-topic_prob is probability of using background model
    mixture_word_topic_prob = {}            # probability of word in mixture model, map word to prob 
    transcription_id = "fa62d651-7172-424a-8853-0fd67b8b80c1" 
    # 374"b7c99a7c-0223-44f8-ada1-ea538f3f7d54"
    transcription_tokens = None
    # cs 125 for test

    # set the background model of the lda
    def set_background(self):
        transcriptions_1 = get_transcription_by_id("a3579844-1afe-4727-8a4c-e80d04e507a0") 
        # 374 "fee4abaa-f0c4-427e-a591-46047c95a781"
        #transcriptions_2 = get_transcription_by_id("9a2f0b4d-d4bd-4e61-9f97-758006f7af37")
        transcriptions_3 = get_transcription_by_id("2b014e8b-21a6-41e3-bcf6-076e3fef9bd6")
        # 374 "de12a60b-afaf-462a-990b-507ffef68f0b"
        transcription_tokens_1 = read_process_text_to_corpus(transcriptions_1)
        #transcription_tokens_2 = read_process_text_to_corpus(transcriptions_2)
        transcription_tokens_3 = read_process_text_to_corpus(transcriptions_3)
        self.background_model = Background()
        self.background_model.add_word_to_background(transcription_tokens_1)
        #self.background_model.add_word_to_background(transcription_tokens_2)
        self.background_model.add_word_to_background(transcription_tokens_3)
        # set background word prob
        for word in self.background_model.word_map:
            self.background_word_topic_prob[word] = self.background_model.word_map[word] / self.background_model.size

    # set the topic model
    def set_topic_model(self):
        transcriptions = get_transcription_by_id(self.transcription_id)
        self.transcription_tokens = read_process_text_to_corpus(transcriptions)
        length = int(len(self.transcription_tokens)/3)
        total_topic_model = TopicModel()
        total_topic_model.create_model(self.transcription_tokens, self.background_model)
        self.document_words = total_topic_model.word_map
        self.vocab = total_topic_model.size
        # create topic model 1 and input first 1/3 transcriptions text
        self.topic_model_1 = TopicModel()
        self.topic_model_1.create_model(self.transcription_tokens[:length], self.background_model)
        #print(self.topic_model.word_map)
        # create topic model 2 and input second 1/3 transcriptions text
        self.topic_model_2 = TopicModel()
        self.topic_model_2.create_model(self.transcription_tokens[length:length*2], self.background_model)
        # create topic model 3 and input last 1/3 transcriptions text
        self.topic_model_3 = TopicModel()
        self.topic_model_3.create_model(self.transcription_tokens[length*2:], self.background_model)

    def init_word_topic_prob(self):
        # initialize topic model 1
        for word in self.topic_model_1.word_map:
            self.word_topic_prob_1[word] = self.topic_model_1.word_map[word] / self.topic_model_1.size
        #print(sorted(self.word_topic_prob.items(), key=lambda x: x[1], reverse=True))
        # initialize  topic model 2
        for word in self.topic_model_2.word_map:
            self.word_topic_prob_2[word] = self.topic_model_2.word_map[word] / self.topic_model_2.size
        # initialize  topic model 3
        for word in self.topic_model_3.word_map:
            self.word_topic_prob_3[word] = self.topic_model_3.word_map[word] / self.topic_model_3.size

    def init_hidden_topic_prob(self):
        for word in self.topic_model_1.word_map.keys():
            self.hidden_word_prob_1[word] = 0.0
        for word in self.topic_model_2.word_map.keys():
            self.hidden_word_prob_2[word] = 0.0
        for word in self.topic_model_3.word_map.keys():
            self.hidden_word_prob_1[word] = 0.0

    # calculated p(z=0|w), which is stored in hidden_word_prob
    def expectation(self, word):
        #hidden_word_prob[word] = (topic_coverage * word_topic_prob[word]) / (self.topic_prob/3 * word_topic_prob[word] + (1-self.topic_prob) * background_prob)
        # calculate P(Zd,w = topic j) j could be topic 1, topic 2 or topic 3
        total_topic_model_prob_sum = 0.0
        if word in self.word_topic_prob_1:
             # word in first topic input text
            total_topic_model_prob_sum += self.topic_coverage_1 * self.word_topic_prob_1[word]
        else:
             # word not in first topic input text
            total_topic_model_prob_sum += self.topic_coverage_1 * 0.0001
        if word in self.word_topic_prob_2:
             # word in second topic input text
            total_topic_model_prob_sum += self.topic_coverage_2 * self.word_topic_prob_2[word]
        else:
             # word not in second topic input text
            total_topic_model_prob_sum += self.topic_coverage_2 * 0.0001
        if word in self.word_topic_prob_3:
             # word in third topic input text
            total_topic_model_prob_sum += self.topic_coverage_3 * self.word_topic_prob_3[word]
        else:
             # word not in third topic input text
            total_topic_model_prob_sum += self.topic_coverage_3 * 0.0001
        self.word_topic_prob_1[word] = self.topic_coverage_1 * self.word_topic_prob_1[word]/total_topic_model_prob_sum
        self.word_topic_prob_2[word] = self.topic_coverage_2 * self.word_topic_prob_2[word]/total_topic_model_prob_sum
        self.word_topic_prob_3[word] = self.topic_coverage_3 * self.word_topic_prob_3[word]/total_topic_model_prob_sum
        # calculate P(Zd,w = B)
        if word not in self.background_model.word_map:
            word_prob_in_background = 0.0001
        else:
            word_prob_in_background = self.background_model.word_map[word] / self.background_model.size
        # calculate P(Zd,w = B) which is same as P(w|B)
        background_prob = (1-self.topic_prob) * word_prob_in_background
        self.background_word_topic_prob[word] = background_prob / (background_prob + total_topic_model_prob_sum)
        

    # calculated p(word|topic_model), which is stored in word_topic_prob
    def maximization(self, word):
        '''
        total = 0.0
        for wo in self.hidden_word_prob:
            total += self.topic_model.word_map[wo] * self.hidden_word_prob[wo]
        self.word_topic_prob[word] = self.topic_model.word_map[word] * self.hidden_word_prob[word] / total
        '''
        # caulculate topic coverage for each topic
        word_prob_sum_1 = self.get_one_topic_model_prob_sum(word, self.topic_model_1, self.word_topic_prob_1)
        word_prob_sum_2 = self.get_one_topic_model_prob_sum(word, self.topic_model_2, self.word_topic_prob_2)
        word_prob_sum_3 = self.get_one_topic_model_prob_sum(word, self.topic_model_3, self.word_topic_prob_3)
        total_word_prob_sum = word_prob_sum_1 + word_prob_sum_2 + word_prob_sum_2
        # calculate topic coverage for each topic model
        self.topic_coverage_1 = word_prob_sum_1/total_word_prob_sum
        self.topic_coverage_2 = word_prob_sum_2/total_word_prob_sum
        self.topic_coverage_3 = word_prob_sum_3/total_word_prob_sum
        # calculate P(w | topic j)
        self.word_topic_prob_1[word] = self.document_words[word] * (1 - self.background_word_topic_prob[word]) * self.word_topic_prob_1[word] / word_prob_sum_1
        self.word_topic_prob_2[word] = self.document_words[word] * (1 - self.background_word_topic_prob[word]) * self.word_topic_prob_2[word] / word_prob_sum_2
        self.word_topic_prob_3[word] = self.document_words[word] * (1 - self.background_word_topic_prob[word]) * self.word_topic_prob_3[word] / word_prob_sum_3

    # get sum of probs of all words in transcriotions in one topic model
    def get_one_topic_model_prob_sum(self, word, topic_model, word_topic_prob):
        word_prob_sum = 0.0
        for word in self.document_words:
            if word not in self.document_words:
                continue
            if word not in self.background_word_topic_prob:
                self.background_word_topic_prob[word] = 0.0
            word_prob_sum += self.document_words[word] * (1 - self.background_word_topic_prob[word]) * word_topic_prob[word]
        return word_prob_sum

    # maximum likelihood estimate of P(w|topic) of given word
    # current is the initial P(z=0|w)
    # gap is the convergence
    def maximum_estimate(self, word, current_1, current_2, current_3, gap):
        prev_1 = 0.0
        prev_2 = 0.0
        prev_3 = 0.0
        #words = self.topic_model.word_map.keys()
        while True:
            self.expectation(word)
            self.maximization(word)
            prev_1 = current_1
            current_1 = self.word_topic_prob_1[word]
            prev_2 = current_2
            current_2 = self.word_topic_prob_1[word]
            prev_3 = current_3
            current_3 = self.word_topic_prob_1[word]
            if current_1 - prev_1 < gap and current_2 - prev_2 < gap and current_3 - prev_3 < gap:
                break
        return self.word_topic_prob_1[word] * self.topic_coverage_1 + self.word_topic_prob_2[word] * self.topic_coverage_2 + self.word_topic_prob_3[word] * self.topic_coverage_3

    # get word prob from mixture model
    def get_word_prob_from_mixture_model(self):
        for word in self.document_words:
            print(word)
            # word only appears in background model
            init = self.document_words[word] / self.vocab
            if init == 0:
                init = 0.001
            print(init)
            self.mixture_word_topic_prob[word] = self.maximum_estimate(word, init, init, init, 0.000001) * self.topic_prob + (1-self.topic_prob) * self.background_word_topic_prob[word]
    
    '''
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
    '''

         # get top probability words from mixture model
    def get_top_prob_topics(self, num):
        self.get_word_prob_from_mixture_model()
        sorted_map = sorted(self.mixture_word_topic_prob.items(), key=lambda x: x[1], reverse=True)
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
    extract.get_word_prob_from_mixture_model()
    topics = extract.get_top_prob_topics(10)
    print(topics)
    '''
    # do the maximum-estimate for each word in the transcirptions
    for w in extract.topic_model.word_map:
        init = extract.topic_model.word_map[w] / extract.topic_model.size
        word_prob_map[w] = extract.maximum_estimate(init, 0.000001, w)
        extract.init_word_topic_prob()
        extract.init_hidden_topic_prob()
    extract.get_word_prob_from_mixture_model()
    topics = extract.get_top_topic()
    #print(sorted(extract.word_topic_prob.items(), key =lambda x: x[1], reverse=True))
    '''