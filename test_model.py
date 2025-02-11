import torch
from build_model import LSTMModel, TransformerModel
from Vocab_Builder import VocabularyBuilder, DatasetBuilder, ClozeTestDataset
from torch.utils.data import DataLoader
from utils import index2word, encode2sentence

DATASET_PATH = "dataset/main_dataset.csv"
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 3
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 4096

vocab_builder = VocabularyBuilder(DATASET_PATH)
vocab_builder.build_vocab()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_dict = vocab_builder.get_vocab()
index_dict = index2word(vocab_dict)

VOCAB_SIZE = len(vocab_dict.items()) + 3
# VOCAB_SIZE = 8199

MAX_LEN = vocab_builder.get_longest_sentence_length()
# MAX_LEN = 27

dataset_builder = DatasetBuilder(vocab_dict, max_len=MAX_LEN)

model = TransformerModel(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_classes=VOCAB_SIZE)
model.load_state_dict(torch.load("kaggle.pth"))
model.eval()
model = model.to(device)

my_sentence = "I go to school with my <mask>"
my_sentence = "I went to <mask> to buy some milk"
my_sentence = "He grabbed <mask> keys locked door and headed out for quick run in park"
my_sentence = "He took book and walked <mask>"
my_sentence = "After long day at work <mask> decided to relax by taking walk in park and enjoying fresh air"

encode_sentence = dataset_builder.encode_sentence(my_sentence)

encode_sentence = torch.tensor(encode_sentence).to(device).unsqueeze(0)

with torch.no_grad():
    outputs = model(encode_sentence)
    outputs = torch.softmax(outputs, 1)
    values, indices = torch.topk(outputs, 5)

    for index in indices[0]:
        predicted_word = index_dict[index.item()]
        completed_sentence = my_sentence.replace("<mask>", predicted_word)

        print(completed_sentence)
