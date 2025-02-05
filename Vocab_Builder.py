import csv
import re
from collections import Counter
import sys
import numpy as np
import random
from torch.utils.data import Dataset

class VocabularyBuilder:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.vocab = {}
        self.stopwords = set(["the", "a", "an"])
        self.index = 3
        self.longest_sentence_length = 0
        self.sentences_list = []
    
    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def build_vocab(self):
        word_count = Counter()
        
        with open(self.csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                sys.stdout.write(f"Processing row {reader.line_num}...\r")
                sys.stdout.flush()
                for col in ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']:
                    sentence = self._clean_text(row[col])
                    self.sentences_list.append(sentence)

                    words = sentence.split()
                    word_count.update(words)

                    self.longest_sentence_length = max(self.longest_sentence_length, len(words))
        
        for word, count in word_count.items():
            if count > 20 and word not in self.stopwords:
                self.vocab[word] = self.index
                self.index += 1

    def get_vocab(self):
        return self.vocab
    
    def get_longest_sentence_length(self):
        return self.longest_sentence_length
    
    def get_sentences_list(self):
        return self.sentences_list

class DatasetBuilder:
    def __init__(self, vocab_dict, max_len=20):
        self.vocab_dict = vocab_dict
        self.max_len = max_len
        self.dataset = []

    def encode_sentence(self, sentence):
        words = sentence.lower().split()
        encoded_sentence = []
        
        for word in words:
            if word == "<mask>":
                encoded_sentence.append(0)
                continue
            if word in self.vocab_dict:
                encoded_sentence.append(self.vocab_dict[word])
            else:
                encoded_sentence.append(2)
        
        padding_needed = self.max_len - len(encoded_sentence)
        encoded_sentence = [1] * padding_needed + encoded_sentence
        
        return encoded_sentence[:self.max_len]
    
    def mask_words_sequentially(self, encoded_sentence):
        masked_sentences = []
        target_words = []
        
        for i in range(len(encoded_sentence)):
            if encoded_sentence[i] > 2:
                masked_sentence = encoded_sentence.copy()
                target_word = encoded_sentence[i]
                masked_sentence[i] = 0
                masked_sentences.append(masked_sentence)
                target_words.append(target_word)
        
        return masked_sentences, target_words

    def build_dataset(self, sentences):
        x_data = []
        y_data = []
        
        for sentence in sentences:
            encoded_sentence = self.encode_sentence(sentence)
            masked_sentences, target_words = self.mask_words_sequentially(encoded_sentence)
            x_data.extend(masked_sentences)
            y_data.extend(target_words)
        
        self.dataset = list(zip(x_data, y_data))
        return np.array(x_data), np.array(y_data)

    def total_dataset(self):
        return len(self.dataset)

    def get_sample(self, index):
        if index < 0 or index >= len(self.dataset):
            raise IndexError("Index out of bounds")
        return self.dataset[index]
    
    def split_dataset(self, test_size=0.2):
        total_samples = len(self.dataset)
        test_samples = int(total_samples * test_size)
        train_samples = total_samples - test_samples
        
        random.seed(43)
        random.shuffle(self.dataset)
        
        train_data = self.dataset[:train_samples]
        test_data = self.dataset[train_samples:]
        
        X_train, Y_train = zip(*train_data)
        X_test, Y_test = zip(*test_data)
        
        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

class ClozeTestDataset(Dataset):
    def __init__(self, X, Y):
        super(ClozeTestDataset, self).__init__()

        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.X.shape[0]

# vocab_builder = VocabularyBuilder("dataset/dataset_2017.csv")
# vocab_builder.build_vocab()

# vocab_dict = vocab_builder.get_vocab()

# sentences = [
#     "I love you very much",
#     "This is another test",
#     "Testing vocab encoding"
# ]

# dataset_builder = DatasetBuilder(vocab_dict, max_len=10)

# x, y = dataset_builder.build_dataset(sentences)

# print("X:", x)
# print("Y:", y)

# print(f"Total samples: {dataset_builder.total_dataset()}")

# sample = dataset_builder.get_sample(0)
# print(f"Sample 0: {sample}")

# vocab_builder = VocabularyBuilder("dataset/dataset_2017.csv")
# vocab_builder.build_vocab()

# vocab_dict = vocab_builder.get_vocab()

# print()
# print(len(vocab_dict.items()))
# print(vocab_builder.get_longest_sentence_length())

# 5526
# 19
