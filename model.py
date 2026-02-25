import torch
import torch.nn as nn

class CNNLSTMSleepNet(nn.Module):
    def __init__(self, n_classes, input_channels=1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: (B, S, C, T)
        B, S, C, T = x.shape

        x = x.view(B * S, C, T)
        x = self.cnn(x)
        x = self.global_pool(x).squeeze(-1)   # (B*S, 256)

        x = x.view(B, S, 256)
        out, _ = self.lstm(x)
        last = out[:, -1, :]                  # (B, 256)

        return self.fc(last)
