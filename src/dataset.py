import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab


def get_transform(split="test"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize(320),
            transforms.RandomCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


class FlickrDataset(Dataset):
    def __init__(self, csv_path, images_dir, vocab, split="test"):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.vocab = vocab
        self.transform = get_transform(split)
        self.word2idx = vocab["word2idx"]
        self.idx2word = vocab["idx2word"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["image_name"]
        caption = row["comment"]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        tokens = ["<SOS>"] + str(caption).split() + ["<EOS>"]
        caption_ids = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)
        return image, caption_tensor, image_name, caption


def collate_fn(batch):
    images, captions, image_names, raw_captions = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(c) for c in captions]
    max_len = max(lengths)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap
    return images, padded, image_names, raw_captions
