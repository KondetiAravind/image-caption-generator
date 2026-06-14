import os
import requests

MODEL_PATH = "models/best_model.pth"
FILE_ID    = "10FGnbcARQn2xhkBhfffz3wyKaI_NkYe5"

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return
    os.makedirs("models", exist_ok=True)
    print("Downloading model from Google Drive...")
    session  = requests.Session()
    url      = "https://drive.google.com/uc?export=download"
    response = session.get(url, params={"id": FILE_ID}, stream=True)
    # Handle Google's virus scan warning for large files
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token:
        response = session.get(url, params={"id": FILE_ID, "confirm": token}, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print(f"Model saved to {MODEL_PATH} ({os.path.getsize(MODEL_PATH)//1024//1024}MB)")

if __name__ == "__main__":
    download_model()
