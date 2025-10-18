import pandas as pd
import torch
import cv2
from utility import get_device
from config import *
from dataset import PrescriptionDataset
from data_utility import preprocess_data, create_dataset, create_dataloaders
from model import CRNN, EarlyStopping
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


def main():
    # Device
    device = get_device()
    print("Using device:", device)
    df_train = pd.read_csv('../dataset/Training/training_labels.csv')
    df_test = pd.read_csv('../dataset/Testing/testing_labels.csv')
    df_val = pd.read_csv('../dataset/Validation/validation_labels.csv')

    
    df_train, df_test, df_val = preprocess_data(df_train, df_test, df_val)

    all_labels = pd.concat([df_train['processed_label'], df_test['processed_label'], df_val['processed_label']])
    all_chars = sorted(list(set(''.join(all_labels.astype(str).tolist()))))
    char_to_int = {char: i for i, char in enumerate(all_chars)}
    int_to_char = {i: char for i, char in enumerate(all_chars)}
    INT_TO_CHAR = int_to_char
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
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    # Implement the training loop
    num_epochs = EPOCHS 

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