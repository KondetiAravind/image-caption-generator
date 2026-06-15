import sys, os, io, base64, torch, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from PIL import Image
from torchvision import transforms
from src.model import CaptionModel
from src.dataset import load_vocab
from src.inference import greedy_caption, beam_search_caption
from src.translate import translate_to_all
from src.visualize import visualize_attention

ROOT       = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(ROOT, "data/vocab.pkl")
MODEL_PATH = os.path.join(ROOT, "models/best_model.pth")
device     = torch.device("cpu")

print("Loading vocab...")
vocab = load_vocab(VOCAB_PATH)
print("Loading model...")
model = CaptionModel().to(device)
ckpt  = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("Model ready!")

TRANSFORM = transforms.Compose([
    transforms.Resize(320), transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def generate(image):
    if image is None:
        return "", "", {}, None
    tensor  = TRANSFORM(image)
    g, _    = greedy_caption(model, tensor, vocab, device)
    b       = beam_search_caption(model, tensor, vocab, device, beam_size=5)
    trans   = translate_to_all(b)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        _, heatmap_path = visualize_attention(model, tmp.name, vocab, device,
                                              output_dir="/tmp/heatmaps")
    os.unlink(tmp.name)
    trans_str = "\n".join([f"{k}: {v}" for k, v in trans.items()])
    return b, g, trans_str, heatmap_path

with gr.Blocks(title="Image Caption Generator", theme=gr.themes.Base()) as demo:
    gr.Markdown("""
# 🖼 Image Caption Generator
**InceptionV3 + Bahdanau Attention + LSTM** · Trained on Flickr30k  
*by [Aravind Kondeti](https://aravind-portfolio-tau.vercel.app)*
""")
    with gr.Row():
        with gr.Column():
            inp   = gr.Image(type="pil", label="Upload Image")
            btn   = gr.Button("✨ Generate Caption", variant="primary")
        with gr.Column():
            beam  = gr.Textbox(label="🎯 Beam Search Caption (k=5)")
            greedy= gr.Textbox(label="⚡ Greedy Caption")
            trans = gr.Textbox(label="🌍 Translations", lines=9)
            heat  = gr.Image(label="🔥 Attention Heatmap")
    btn.click(generate, inputs=inp, outputs=[beam, greedy, trans, heat])

demo.launch(server_name="0.0.0.0", server_port=7860)
