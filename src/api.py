import io, os, sys, torch, base64
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from src.model import CaptionModel
from src.dataset import load_vocab
from src.inference import greedy_caption, beam_search_caption
from src.translate import translate_to_all
from src.visualize import visualize_attention
import tempfile

ROOT       = os.path.join(os.path.dirname(__file__), "..")
VOCAB_PATH = os.path.join(ROOT, "data/vocab.pkl")
MODEL_PATH = os.path.join(ROOT, "models/best_model.pth")
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Image Caption Generator")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

vocab = load_vocab(VOCAB_PATH)
model = CaptionModel().to(device)
ckpt  = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"API ready | device={device}")

TRANSFORM = transforms.Compose([
    transforms.Resize(320), transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

@app.get("/")
def root():
    return {"status": "ok", "message": "Image Caption Generator API"}

@app.post("/caption")
async def caption(file: UploadFile = File(...),
                  beam_size: int = 5, translate: bool = False):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file.")
    try:
        img    = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = TRANSFORM(img)
        g, _   = greedy_caption(model, tensor, vocab, device)
        b      = beam_search_caption(model, tensor, vocab, device, beam_size=beam_size)
        result = {"greedy": g, "beam": b}
        if translate:
            result["translations"] = translate_to_all(b)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/caption_with_heatmap")
async def caption_with_heatmap(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file.")
    try:
        data = await file.read()
        img  = Image.open(io.BytesIO(data)).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            caption, heatmap_path = visualize_attention(
                model, tmp.name, vocab, device, output_dir="/tmp/heatmaps")
        with open(heatmap_path, "rb") as f:
            heatmap_b64 = base64.b64encode(f.read()).decode()
        os.unlink(tmp.name)
        return {"caption": caption, "heatmap_base64": heatmap_b64}
    except Exception as e:
        raise HTTPException(500, str(e))
