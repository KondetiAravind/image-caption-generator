import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=8, embed_size=256):
        super(EncoderCNN, self).__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)

        # Use named submodules directly — avoids any Sequential index mismatch
        # across torchvision versions. This mirrors exactly what the checkpoint stores.
        self.cnn = nn.Sequential(
            inception.Conv2d_1a_3x3,    # 0
            inception.Conv2d_2a_3x3,    # 1
            inception.Conv2d_2b_3x3,    # 2
            nn.MaxPool2d(3, stride=2),  # 3  (no params, not in ckpt — skipped index 3)
            inception.Conv2d_3b_1x1,    # 4
            inception.Conv2d_4a_3x3,    # 5
            nn.MaxPool2d(3, stride=2),  # 6  (no params — skipped index 6)
            inception.Mixed_5b,         # 7
            inception.Mixed_5c,         # 8
            inception.Mixed_5d,         # 9
            inception.Mixed_6a,         # 10
            inception.Mixed_6b,         # 11
            inception.Mixed_6c,         # 12
            inception.Mixed_6d,         # 13
            inception.Mixed_6e,         # 14
            inception.Mixed_7a,         # 15
            inception.Mixed_7b,         # 16
            inception.Mixed_7c,         # 17
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.projection    = nn.Linear(2048, embed_size)
        self.relu          = nn.ReLU()
        self.dropout       = nn.Dropout(0.5)

        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, images):
        features = self.cnn(images)                            # (B, 2048, H, W)
        features = self.adaptive_pool(features)                # (B, 2048, 8, 8)
        B, C, H, W = features.size()
        features = features.view(B, C, -1).permute(0, 2, 1)   # (B, 64, 2048)
        features = self.relu(self.projection(features))        # (B, 64, 256)
        features = self.dropout(features)
        return features


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim=256, decoder_dim=512, attention_dim=256):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att    = nn.Linear(attention_dim, 1)
        self.softmax     = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1    = self.encoder_att(encoder_out)
        att2    = self.decoder_att(decoder_hidden).unsqueeze(1)
        att     = self.full_att(torch.tanh(att1 + att2)).squeeze(2)
        alpha   = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class DecoderRNN(nn.Module):
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=6988,
                 num_layers=2, dropout=0.5, encoder_dim=256, attention_dim=256):
        super(DecoderRNN, self).__init__()
        self.attention   = BahdanauAttention(encoder_dim, hidden_size, attention_dim)
        self.embedding   = nn.Embedding(vocab_size, embed_size)
        self.lstm        = nn.LSTM(embed_size + encoder_dim, hidden_size,
                                   num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

    def forward(self, encoder_out, captions):
        B          = encoder_out.size(0)
        embeddings = self.dropout(self.embedding(captions))
        h = torch.zeros(self.num_layers, B, self.hidden_size).to(encoder_out.device)
        c = torch.zeros(self.num_layers, B, self.hidden_size).to(encoder_out.device)
        outputs = []
        for t in range(embeddings.size(1)):
            context, _   = self.attention(encoder_out, h[-1])
            lstm_input   = torch.cat([embeddings[:, t, :], context], dim=1).unsqueeze(1)
            out, (h, c)  = self.lstm(lstm_input, (h, c))
            pred         = self.fc(self.dropout(out.squeeze(1)))
            outputs.append(pred)
        return torch.stack(outputs, dim=1)


class CaptionModel(nn.Module):
    def __init__(self, vocab_size=6988, embed_size=256, hidden_size=512,
                 num_layers=2, dropout=0.5, encoded_image_size=8, attention_dim=256):
        super(CaptionModel, self).__init__()
        self.encoder = EncoderCNN(encoded_image_size, embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size,
                                  num_layers, dropout, embed_size, attention_dim)

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)
