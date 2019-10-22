class Background:
    word_map = {}               # map word to word count
    size = 0                   # size of the model

    # given a list of transcriptions, create background model
    def add_word_to_background(self, transcriptions):
        for word in transcriptions:
            if word in self.word_map:
                self.word_map[word] += 1
            else:
                self.word_map[word] = 1
            self.size += 1

    # return top num most frequent words in the map
    def get_top_freq_words(self, num):
        top_words = []
        sorted_map = sorted(self.word_map.items(), key=lambda x: x[1], reverse=True)
        count = 0
        for word in sorted_map:
            top_words.append(word[0])
            count += 1
            if count >= num:
                break
        return top_words

    # get probability of word in the background model
    def get_probability(self, word):
        if word not in self.word_map:
            return 0
        return self.word_map[word]/self.size
