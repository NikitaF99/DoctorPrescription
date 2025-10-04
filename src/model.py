import torch.nn as nn
import torch.nn.functional as F
import torch

# Redefine the CRNN model with adjusted CNN layers
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        # CNN layers for feature extraction
        # Adjusted some kernel sizes and strides to prevent dimension issues
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # Output size: (batch, 32, 16, 128)

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # Output size: (batch, 64, 8, 64)

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # Output size: (batch, 128, 4, 64)

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # Output size: (batch, 256, 2, 64)

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # Output size: (batch, 512, 1, 64)

            # Last conv layer adjusted kernel size
            nn.Conv2d(512, 512, kernel_size=(1, 2), stride=1, padding=0), # Output size: (batch, 512, 1, 63)
            nn.ReLU(inplace=True),
        )

        # RNN layers (Bidirectional LSTM)
        # Input size to RNN will be 512 (channels)
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=False),
            nn.Linear(256 * 2, num_classes) # Output layer
        )


    def forward(self, x):
        # Pass through CNN
        conv_output = self.cnn(x)

        # Reshape for RNN (sequence length, batch size, feature size)
        # The height should be 1 after pooling layers
        batch_size, channels, height, width = conv_output.size()
        conv_output = conv_output.squeeze(2) # Remove height dimension (should be 1 after pooling)
        conv_output = conv_output.permute(2, 0, 1) # Change from (batch, channels, width) to (width, batch, channels)

        # Pass through RNN
        rnn_output, _ = self.rnn[0](conv_output)

        # Pass through linear layer
        output = self.rnn[1](rnn_output)

        # Apply log softmax for CTC loss
        output = F.log_softmax(output, dim=2)

        return output
    

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        print("Early Stopping created")

    def __call__(self, val_loss, model):
        score = -val_loss
        print("Early stopping call")

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
