from Vocab_Builder import VocabularyBuilder, DatasetBuilder, ClozeTestDataset
from build_model import LSTMModel
# from torchsummary import summary
from torchinfo import summary
import torch

vocab_builder = VocabularyBuilder("dataset/dataset_2017.csv")
vocab_builder.build_vocab()

vocab_dict = vocab_builder.get_vocab()

vocab_size = len(vocab_dict.items()) - 2

sentences = vocab_builder.get_sentences_list()

max_len = vocab_builder.get_longest_sentence_length()

dataset_builder = DatasetBuilder(vocab_dict, max_len=max_len)

x, y = dataset_builder.build_dataset(sentences)

print(f"Total samples: {dataset_builder.total_dataset()}")

sample = dataset_builder.get_sample(0)

X_train, Y_train, X_test, Y_test = dataset_builder.split_dataset(test_size=0.2)

train_set = ClozeTestDataset(X_train, Y_train)

print(len(train_set))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(vocab_size=vocab_size, embedding_dim=256, hidden_dim=256, num_layers=3, num_classes=vocab_size)
model = model.to(device)
batch_size = 512

x = torch.zeros(batch_size, max_len).long().to(device)

summary(model, input_data=x)

# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# LSTMModel                                [512, 5524]               --
# ├─Embedding: 1-1                         [512, 19, 256]            1,414,144
# ├─LSTM: 1-2                              [512, 19, 256]            1,579,008
# ├─Linear: 1-3                            [512, 5524]               1,419,668
# ==========================================================================================
# Total params: 4,412,820
# Trainable params: 4,412,820
# Non-trainable params: 0
# Total mult-adds (Units.GIGABYTES): 16.81
# ==========================================================================================
# Input size (MB): 0.08
# Forward/backward pass size (MB): 62.47
# Params size (MB): 17.65
# Estimated Total Size (MB): 80.20
# ==========================================================================================