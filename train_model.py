import torch.nn as nn
from Vocab_Builder import VocabularyBuilder, DatasetBuilder, ClozeTestDataset
from build_model import LSTMModel, TransformerModel
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import train
from torch.utils.data import DataLoader

if __name__ == "__main__":
    DATASET_PATH = "dataset/main_dataset.csv"
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 256
    NUM_LAYERS = 3
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    BATCH_SIZE = 4096

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_builder = VocabularyBuilder(DATASET_PATH)
    vocab_builder.build_vocab()

    vocab_dict = vocab_builder.get_vocab()

    VOCAB_SIZE = len(vocab_dict.items()) + 3

    sentences = vocab_builder.get_sentences_list()

    max_len = vocab_builder.get_longest_sentence_length()

    print(f"\nVocab size: {VOCAB_SIZE}, max length: {max_len}")

    dataset_builder = DatasetBuilder(vocab_dict, max_len=max_len)

    x, y = dataset_builder.build_dataset(sentences)

    print(f"Total samples: {dataset_builder.total_dataset()}")

    sample = dataset_builder.get_sample(0)

    X_train, Y_train, X_test, Y_test = dataset_builder.split_dataset(test_size=0.2)

    train_set = ClozeTestDataset(X_train, Y_train)
    test_set = ClozeTestDataset(X_test, Y_test)

    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")

    train_data_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_data_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # model = LSTMModel(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_classes=VOCAB_SIZE)
    model = TransformerModel(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, num_classes=VOCAB_SIZE)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, valid_accuracies, valid_losses = train(model, train_data_loader, test_data_loader, criterion, optimizer, device, NUM_EPOCHS)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(valid_losses, label="Validation Loss", color='red', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracies, label="Validation Accuracy", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.show()

    torch.save(model.state_dict(), "trained_model.pth")
    print("Model saved successfully.")