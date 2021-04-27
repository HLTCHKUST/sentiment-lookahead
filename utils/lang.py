import nltk
import spacy


class Lang:
    def __init__(self):
        self.unk_idx = 0
        self.pad_idx = 1
        self.sou_idx = 2
        self.eou_idx = 3

        self.word2index = {'__unk__': self.unk_idx, '__pad__': self.pad_idx, '__sou__': self.sou_idx, '__eou__': self.eou_idx}
        self.word2count = {'__unk__': 0, '__pad__': 0, '__sou__': 0, '__eou__': 0}
        self.index2word = {self.unk_idx: "__unk__", self.pad_idx: "__pad__", self.sou_idx: "__sou__", self.eou_idx: "__eou__"} 
        self.n_words = 4 # Count default tokens

        self.nlp = spacy.load("en_core_web_sm")
        # add special case rule
        special_case = [{spacy.symbols.ORTH: u"__eou__"}]
        self.nlp.tokenizer.add_special_case(u"__eou__", special_case)

    def __len__(self):
        return self.n_words

    def tokenize(self, s):
        # return nltk.word_tokenize(s)
        return self.nlp.tokenizer(s)

    def addSentence(self, sentence):
        for word in self.tokenize(sentence):
            self.addWord(word.text)

    def addSentences(self, sentences):
        for sentence in sentences:
            for word in self.tokenize(sentence):
                self.addWord(word.text)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def transform(self, sentences):
        # given unokenized sentences (or iterator), transform to idx mapping
        return [[self.word2index[token.text] for token in self.tokenize(sentence) if not token.is_space] for sentence in sentences]

    def transform_one(self, sentence):
        try:
        # given unokenized sentence, transform to idx mapping
            return [self.word2index[token.text] for token in self.tokenize(sentence) if not token.is_space]
        except KeyError as e:
            print(e)
            print(sentence)
            for token in self.tokenize(sentence):
                if not token.is_space:
                    print(token.text, token.text in self.word2index)
            exit(1)

    def transform_unk(self, sentence):
        # transform with unk
        ret = []
        for token in self.tokenize(sentence):
            if token.text in self.word2index:
                ret.append(self.word2index[token.text])
            else:
                ret.append(self.unk_idx)
        return ret

    def reverse(self, sentences):
        # given transformed sentences, reverse it
        return [[self.index2word[idx] for idx in sentence] for sentence in sentences]

    def reverse_one(self, sentence):
        # given transformed sentence, reverse it
        return [self.index2word[idx] for idx in sentence]

    # def trim(self, min_freq=100):
    #     print('vocab size before trimming: ', len(self))
    #     self.word2count[self.unk_idx] = min_freq
    #     self.word2count[self.pad_idx] = min_freq
    #     self.word2count[self.sou_idx] = min_freq
    #     self.word2count[self.eou_idx] = min_freq

    #     self.word2count = {k: v for k, v in self.word2count if v >= 100}
    #     trimmed_word2index = {'__unk__': self.unk_idx, '__pad__': self.pad_idx, '__sou__': self.sou_idx, '__eou__': self.eou_idx}
    #     trimmed_index2word = {self.unk_idx: "__unk__", self.pad_idx: "__pad__", self.sou_idx: "__sou__", self.eou_idx: "__eou__"} 

    #     self.word2index = trimmed_word2index
    #     print('vocab size after trimming: ', len(self))
    #     return self
