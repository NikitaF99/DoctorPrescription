import pandas as pd
import torch
import cv2
from preprocess import apply_preprocessing_to_row, create_ctc_labels, image_to_tensor
from utility import get_device
from config import *
from dataset import PrescriptionDataset
from model import CRNN, EarlyStopping
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def create_dataloaders(dataset, batch_size, shuffle = False):
    return DataLoader(dataset, batch_size, shuffle)

def create_dataset(dataframe, char_to_int,max_label_length):
    return PrescriptionDataset(dataframe, char_to_int=char_to_int, max_label_length=max_label_length)

def preprocess(df_train, df_test,df_val):
    print("Preprocessing training data...")

    # processed_train_results_list = [apply_preprocessing_to_row(row, '/content/dataset/Training/training_words') for index, row in df_train.iterrows()]
    processed_train_results_list = []
    for index, row in df_train.iterrows():
        result = apply_preprocessing_to_row(row, '../dataset/Training/training_words')
        processed_train_results_list.append(result)


    processed_train_df = pd.DataFrame(processed_train_results_list, index=df_train.index, columns=['preprocessed_image', 'processed_label'])
    df_train['preprocessed_image'] = processed_train_df['preprocessed_image']
    df_train['processed_label'] = processed_train_df['processed_label']
    labels = set()
    labels.update(df_train['processed_label'])
    print(labels)

    print("Preprocessing testing data...")
    processed_test_results_list = [apply_preprocessing_to_row(row, '../dataset/Testing/testing_words') for index, row in df_test.iterrows()]
    processed_test_df = pd.DataFrame(processed_test_results_list, index=df_test.index, columns=['preprocessed_image', 'processed_label'])
    df_test['preprocessed_image'] = processed_test_df['preprocessed_image']
    df_test['processed_label'] = processed_test_df['processed_label']

    print("Preprocessing validation data...")
    processed_val_results_list = [apply_preprocessing_to_row(row, '../dataset/Validation/validation_words') for index, row in df_val.iterrows()]
    processed_val_df = pd.DataFrame(processed_val_results_list, index=df_val.index, columns=['preprocessed_image', 'processed_label'])
    df_val['preprocessed_image'] = processed_val_df['preprocessed_image']
    df_val['processed_label'] = processed_val_df['processed_label']


    # Create a character to integer mapping and vice versa
    # Get all unique characters from the labels
    all_labels = pd.concat([df_train['processed_label'], df_test['processed_label'], df_val['processed_label']])
    all_chars = sorted(list(set(''.join(all_labels.astype(str).tolist()))))

    char_to_int = {char: i for i, char in enumerate(all_chars)}
    int_to_char = {i: char for i, char in enumerate(all_chars)}

    print('In func')
    print(char_to_int)

    max_label_length = max(all_labels.astype(str).apply(len))

    df_train['preprocessed_image_crnn'] = df_train['preprocessed_image'].apply(lambda x: image_to_tensor(x))
    df_train['ctc_label'] = df_train['processed_label'].apply(lambda x: create_ctc_labels(x, char_to_int, max_label_length))

    df_val['preprocessed_image_crnn'] = df_val['preprocessed_image'].apply(lambda x: image_to_tensor(x))
    df_val['ctc_label'] = df_val['processed_label'].apply(lambda x: create_ctc_labels(x, char_to_int, max_label_length))

    df_test['preprocessed_image_crnn'] = df_test['preprocessed_image'].apply(lambda x: image_to_tensor(x))
    df_test['ctc_label'] = df_test['processed_label'].apply(lambda x: create_ctc_labels(x, char_to_int, max_label_length))

    print("Training DataFrame with CRNN preprocessed images and CTC labels:")
    print(df_train.head())

    print("\nValidation DataFrame with CRNN preprocessed images and CTC labels:")
    print(df_val.head())

    print("\nTesting DataFrame with CRNN preprocessed images and CTC labels:")
    print(df_test.head())

    return df_train, df_test, df_val

def main():
    # Device
    device = get_device()
    print("Using device:", device)
    df_train = pd.read_csv('../dataset/Training/training_labels.csv')
    df_test = pd.read_csv('../dataset/Testing/testing_labels.csv')
    df_val = pd.read_csv('../dataset/Validation/validation_labels.csv')

    
    df_train, df_test, df_val = preprocess(df_train, df_test, df_val)

    all_labels = pd.concat([df_train['processed_label'], df_test['processed_label'], df_val['processed_label']])
    all_chars = sorted(list(set(''.join(all_labels.astype(str).tolist()))))

    char_to_int = {char: i for i, char in enumerate(all_chars)}
    int_to_char = {i: char for i, char in enumerate(all_chars)}

    max_label_length = max(all_labels.astype(str).apply(len))

    train_dataset = create_dataset(df_train, char_to_int=char_to_int, max_label_length=max_label_length)
    test_dataset = create_dataset(df_test, char_to_int=char_to_int, max_label_length=max_label_length)
    val_dataset = create_dataset(df_val, char_to_int=char_to_int, max_label_length=max_label_length)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Training DataLoader created.")
    print("Validation DataLoader created.")
    print("Testing DataLoader created.")


    # Instantiate the CRNN model with the corrected architecture
    num_classes = len(char_to_int) + 1
    model = CRNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CTCLoss(blank=len(char_to_int)) # Use the size of the character set as the blank index
    early_stopping = EarlyStopping(patience=3, delta=0.01)

    # Implement the training loop
    num_epochs = EPOCHS # You can adjust the number of epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Get input lengths (width of features after CNN and before RNN)
            # This should be the number of time steps for the RNN, which is the width of the CNN output.
            with torch.no_grad():
                # Get the actual output width from the CNN for the current batch size
                dummy_output = model.cnn(images)
                _, _, _, output_width = dummy_output.size()
                input_lengths = torch.full((images.size(0),), output_width, dtype=torch.long).to(device)


            # Get target lengths (length of CTC labels)
            # Exclude padding value (len(char_to_int))
            target_lengths = torch.tensor([len([l for l in label if l != len(char_to_int)]) for label in labels], dtype=torch.long).to(device)


            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate the CTC loss
            # outputs: (time_steps, batch_size, num_classes)
            # labels: (batch_size, max_label_length) - need to flatten and remove padding
            # input_lengths: (batch_size,)
            # target_lengths: (batch_size,)

            # Flatten the labels and remove padding
            flat_labels = torch.cat([label[:target_lengths[i]] for i, label in enumerate(labels)])
            loss = criterion(outputs, flat_labels, input_lengths, target_lengths)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item() # Accumulate loss

        epoch_loss = running_loss / len(train_dataloader) # Calculate average loss per batch
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Validation step
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                # Get input lengths
                dummy_output = model.cnn(images)
                _, _, _, output_width = dummy_output.size()
                input_lengths = torch.full((images.size(0),), output_width, dtype=torch.long).to(device)

                # Get target lengths
                target_lengths = torch.tensor([len([l for l in label if l != len(char_to_int)]) for label in labels], dtype=torch.long).to(device)

                outputs = model(images)

                flat_labels = torch.cat([label[:target_lengths[i]] for i, label in enumerate(labels)])

                loss = criterion(outputs, flat_labels, input_lengths, target_lengths)

                val_running_loss += loss.item() # Accumulate loss

        val_epoch_loss = val_running_loss / len(val_dataloader) # Calculate average loss per batch
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}")

        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Training finished.")
    early_stopping.load_best_model(model)
    
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved")


if __name__ == "__main__":
    main()