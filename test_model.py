import torch
from build_model import LSTMModel
from Vocab_Builder import VocabularyBuilder, DatasetBuilder, ClozeTestDataset
from torch.utils.data import DataLoader
from utils import index2word, encode2sentence

vocab_builder = VocabularyBuilder("dataset/dataset_2017.csv")
vocab_builder.build_vocab()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_dict = vocab_builder.get_vocab()
index_dict = index2word(vocab_dict)

DATASET_PATH = "dataset/dataset_2017.csv"
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 3
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 4096

VOCAB_SIZE = len(vocab_dict.items()) + 3

sentences = vocab_builder.get_sentences_list()

max_len = vocab_builder.get_longest_sentence_length()

dataset_builder = DatasetBuilder(vocab_dict, max_len=max_len)

model = LSTMModel(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_classes=VOCAB_SIZE)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()
model = model.to(device)

my_sentence = "I go to school with my <mask>"

encode_sentence = dataset_builder.encode_sentence(my_sentence)

encode_sentence = torch.tensor(encode_sentence).to(device).unsqueeze(0)

print(encode_sentence)
with torch.no_grad():
    outputs = model(encode_sentence)
    index = torch.argmax(outputs).item()

    print(index_dict[index])    
