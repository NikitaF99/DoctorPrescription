
import torch
from utility import ctc_decode, levenshtein_distance
from model import CRNN
import rapidfuzz.distance.Levenshtein as Levenshtein
from torch.utils.data import  DataLoader
from config import BATCH_SIZE
from main import create_dataset
import pandas as pd
from main import preprocess

def load_data_params():
    df_train = pd.read_csv('../dataset/Training/training_labels.csv')
    df_test = pd.read_csv('../dataset/Testing/testing_labels.csv')
    df_val = pd.read_csv('../dataset/Validation/validation_labels.csv')

    
    df_train, df_test, df_val = preprocess(df_train, df_test, df_val)

    all_labels = pd.concat([df_train['processed_label'], df_test['processed_label'], df_val['processed_label']])
    all_chars = sorted(list(set(''.join(all_labels.astype(str).tolist()))))

    char_to_int = {char: i for i, char in enumerate(all_chars)}
    int_to_char = {i: char for i, char in enumerate(all_chars)}
    max_label_length = max(all_labels.astype(str).apply(len))
    num_classes = len(char_to_int) + 1

    lexicon = set()
    lexicon.update(df_train['processed_label'])
    lexicon.update(df_test['processed_label'])
    lexicon.update(df_val['processed_label'])

    return char_to_int, int_to_char, max_label_length, num_classes, df_test, lexicon

def evaluation(test_dataloader, num_classes, device, int_to_char, char_to_int, lexicon):
    model = CRNN(num_classes=num_classes).to(device)   # initialize same architecture as before
    model.load_state_dict(torch.load("./crnn_model1.pth", map_location=torch.device('cpu')))
    model.eval()
    test_decoded_predictions = []
    test_true_labels = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass to get the model outputs (log probabilities)
            outputs = model(images) # outputs is (time_steps, batch_size, num_classes)

            # Decode the predictions, now with lexicon search
            decoded_predictions = ctc_decode(outputs, int_to_char, lexicon=lexicon) # Pass the lexicon
            test_decoded_predictions.extend(decoded_predictions)

            # Convert true labels back to strings for comparison
            true_labels_list = []
            for label_tensor in labels:
                # Remove padding (len(char_to_int)) and convert to characters
                true_label_chars = [int_to_char.get(l.item(), '') for l in label_tensor if l.item() != len(char_to_int)]
                true_labels_list.append(''.join(true_label_chars))
            test_true_labels.extend(true_labels_list)

    total_characters = 0
    total_edit_distance = 0

    for true_label, predicted_label in zip(test_true_labels, test_decoded_predictions):
        total_characters += len(true_label)
        total_edit_distance += levenshtein_distance(true_label, predicted_label)

    # Avoid division by zero if there are no characters
    cer = total_edit_distance / total_characters if total_characters > 0 else 0

    print(f"\nCharacter Error Rate (CER) on Test Set: {cer:.4f}")

    # Optionally, print some example predictions and true labels
    print("\nExample Predictions vs. True Labels:")
    for i in range(min(10, len(test_decoded_predictions))):
        print(f"True: {test_true_labels[i]}, Predicted: {test_decoded_predictions[i]}")
    
    # WER
    total_words = 0
    total_word_edit_distance = 0

    for true_label, predicted_label in zip(test_true_labels, test_decoded_predictions):
        true_words = true_label.split(" ")
        predicted_words = predicted_label.split(" ")

        total_words += len(true_words)
        # Using rapidfuzz's distance for word lists
        total_word_edit_distance += Levenshtein.distance(true_words, predicted_words)

    # Avoid division by zero if there are no words
    wer = total_word_edit_distance / total_words if total_words > 0 else 0

    print(f"\nWord Error Rate (WER) on Test Set: {wer:.4f}")


def main():
    char_to_int, int_to_char, max_label_length, num_classes, df_test, lexicon = load_data_params()
    dataset = create_dataset(df_test, char_to_int=char_to_int, max_label_length=max_label_length)
    dataloader =  DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    evaluation(dataloader, num_classes, 'cpu', int_to_char, char_to_int, lexicon)


if __name__ == "__main__":
    main()

