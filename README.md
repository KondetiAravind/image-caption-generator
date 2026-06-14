# Attention-Based Image Caption Generator

An end-to-end image captioning system built with **InceptionV3 + Bahdanau Attention + 2-layer LSTM**, trained on the **Flickr30k** dataset (31,783 images, 158,915 captions).

## Live Demo
🚀 [Try it on Vercel](#) <!-- update after deployment -->

## Results

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
|--------|--------|--------|--------|--------|--------|
| Greedy | 0.3592 | 0.1939 | 0.1091 | 0.0596 | 0.2750 |
| Beam-5 | 0.4637 | 0.2562 | 0.1450 | 0.0820 |   —    |

## Architecture
- **Encoder**: InceptionV3 (pretrained, frozen) → AdaptiveAvgPool2d(8×8) → Linear(2048→256) → 64 spatial feature vectors
- **Attention**: Bahdanau (additive) attention over 64 regions
- **Decoder**: 2-layer LSTM (hidden=512) with scheduled teacher forcing
- **Vocabulary**: 6,988 words (frequency threshold ≥ 5)
- **Best model**: Epoch 1, Val Loss 7.6367

## Features
- Greedy and Beam Search (k=5) decoding
- Attention heatmap visualization per word
- BLEU-1/2/3/4 + METEOR evaluation
- Multilingual translation (Hindi, French, Spanish, German, Tamil, Telugu, Japanese, Arabic)
- FastAPI REST backend

## Project Structure
├── src/

│   ├── model.py       # Encoder, Attention, Decoder, CaptionModel

│   ├── dataset.py     # FlickrDataset + transforms

│   ├── inference.py   # Greedy + Beam Search

│   ├── evaluate.py    # BLEU + METEOR

│   ├── visualize.py   # Attention heatmaps

│   ├── translate.py   # Multilingual translation

│   └── api.py         # FastAPI backend

├── main.py            # CLI entry point

├── data/              # vocab.pkl + test.csv

└── models/            # best_model.pth (not tracked in git)

## Setup
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
# Place best_model.pth in models/ and vocab.pkl in data/
```

## Usage
```bash
# Caption a single image
python main.py caption --image path/to/image.jpg --translate

# Attention heatmap
python main.py visualize --image path/to/image.jpg

# Full test set evaluation
python main.py evaluate --beam_size 5
```

## API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
- `POST /caption` — upload image → returns greedy + beam captions + translations
- `POST /caption_with_heatmap` — upload image → returns caption + attention heatmap (base64)

## Dataset
Flickr30k — 31,783 images with 5 captions each. Train/Val/Test split: 25,426 / 3,178 / 3,179 images.