class TopicModel:
    word_map = {}
    size = 0

    def create_model(self, transcriptions, background):
        top_background_word = background.get_top_freq_words(450)
        for word in transcriptions:
            if word not in top_background_word:
                if word in self.word_map:
                    self.word_map[word] += 1
                else:
                    self.word_map[word] = 1
            else:
                self.word_map[word] = 1
            if word not in self.word_map:
                self.size += 1

    # get potential topic words from topic model, testing purpose
    def get_topic_words(self, num):
        topic_words = []
        sorted_map = sorted(self.word_map.items(), key=lambda x: x[1], reverse=True)
        count = 0
        for word in sorted_map:
            topic_words.append(word)
            count += 1
            if count >= num:
                break
        return topic_words
