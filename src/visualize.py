import os, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.inference import greedy_caption


TRANSFORM = transforms.Compose([
    transforms.Resize(320), transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


def visualize_attention(model, image_path, vocab, device,
                        output_dir="outputs/heatmaps", max_len=20):
    os.makedirs(output_dir, exist_ok=True)
    raw    = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(raw)
    caption, alphas = greedy_caption(model, tensor, vocab, device, max_len=max_len)
    words  = caption.split()

    alphas = torch.stack(alphas, dim=0).numpy()   # (T, 64)
    alphas = alphas.reshape(-1, 8, 8)             # (T, 8, 8)
    n      = min(len(words), len(alphas))

    cols = 4
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = np.array(axes).flatten()
    raw_resized = raw.resize((299, 299))

    for i in range(n):
        alpha_up = np.array(Image.fromarray(alphas[i]).resize((299,299), Image.BILINEAR))
        axes[i].imshow(raw_resized)
        axes[i].imshow(alpha_up, cmap="jet", alpha=0.45)
        axes[i].set_title(words[i], fontsize=10)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Caption: {caption}", fontsize=11)
    plt.tight_layout()
    stem      = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{stem}_attention.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Caption : {caption}")
    print(f"Saved   : {save_path}")
    return caption, save_path
