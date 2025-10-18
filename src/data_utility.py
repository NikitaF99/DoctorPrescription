import torch
from torch.utils.data import DataLoader
from preprocess import apply_preprocessing_to_row, create_ctc_labels, image_to_tensor
from dataset import PrescriptionDataset
import pandas as pd

def preprocess_data(df_train, df_test, df_val):
    print("Preprocessing data...")
    processed_train = [apply_preprocessing_to_row(row, '../dataset/Training/training_words') for _, row in df_train.iterrows()]
    processed_test = [apply_preprocessing_to_row(row, '../dataset/Testing/testing_words') for _, row in df_test.iterrows()]
    processed_val = [apply_preprocessing_to_row(row, '../dataset/Validation/validation_words') for _, row in df_val.iterrows()]

    for df, processed in zip([df_train, df_test, df_val],
                             [processed_train, processed_test, processed_val]):
        processed_df = pd.DataFrame(processed, columns=['preprocessed_image', 'processed_label'])
        df['preprocessed_image'] = processed_df['preprocessed_image']
        df['processed_label'] = processed_df['processed_label']

    all_labels = pd.concat([df_train['processed_label'], df_test['processed_label'], df_val['processed_label']])
    all_chars = sorted(list(set(''.join(all_labels.astype(str).tolist()))))
    char_to_int = {char: i for i, char in enumerate(all_chars)}
    int_to_char = {i: char for i, char in enumerate(all_chars)}
    max_label_length = max(all_labels.astype(str).apply(len))

    # Convert images to tensors and labels to CTC format
    for df in [df_train, df_test, df_val]:
        df['preprocessed_image_crnn'] = df['preprocessed_image'].apply(image_to_tensor)
        df['ctc_label'] = df['processed_label'].apply(lambda x: create_ctc_labels(x, char_to_int, max_label_length))

    print("Training DataFrame with CRNN preprocessed images and CTC labels:")
    print(df_train.head())

    print("\nValidation DataFrame with CRNN preprocessed images and CTC labels:")
    print(df_val.head())

    print("\nTesting DataFrame with CRNN preprocessed images and CTC labels:")
    print(df_test.head())

    return df_train, df_test, df_val


def create_dataloaders(dataset, batch_size, shuffle = False):
    return DataLoader(dataset, batch_size, shuffle)

def create_dataset(dataframe, char_to_int,max_label_length):
    return PrescriptionDataset(dataframe, char_to_int=char_to_int, max_label_length=max_label_length)