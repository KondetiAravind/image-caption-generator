import os, json, torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from src.inference import greedy_caption, beam_search_caption


TRANSFORM = transforms.Compose([
    transforms.Resize(320), transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


def evaluate(model, test_csv, images_dir, vocab, device,
             output_dir="outputs/metrics", use_beam=True, beam_size=5, max_samples=None):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    df      = pd.read_csv(test_csv)
    grouped = df.groupby("image_name")["comment"].apply(list).reset_index()
    if max_samples:
        grouped = grouped.head(max_samples)

    references, hyp_greedy, hyp_beam, results = [], [], [], []

    print(f"Evaluating on {len(grouped)} images...")
    for _, row in tqdm(grouped.iterrows(), total=len(grouped)):
        img_name = row["image_name"]
        ref_caps = [str(c).split() for c in row["comment"]]
        img_path = os.path.join(images_dir, img_name)
        try:
            tensor = TRANSFORM(Image.open(img_path).convert("RGB"))
        except:
            continue

        g, _ = greedy_caption(model, tensor, vocab, device)
        references.append(ref_caps)
        hyp_greedy.append(g.split())

        b = ""
        if use_beam:
            b = beam_search_caption(model, tensor, vocab, device, beam_size=beam_size)
            hyp_beam.append(b.split())

        results.append({"image_name": img_name,
                        "references": [" ".join(r) for r in ref_caps],
                        "greedy": g, "beam": b})

    sf = SmoothingFunction().method1
    def bleu(refs, hyps):
        return {
            "BLEU-1": round(corpus_bleu(refs, hyps, weights=(1,0,0,0),         smoothing_function=sf), 4),
            "BLEU-2": round(corpus_bleu(refs, hyps, weights=(0.5,0.5,0,0),     smoothing_function=sf), 4),
            "BLEU-3": round(corpus_bleu(refs, hyps, weights=(.33,.33,.33,0),    smoothing_function=sf), 4),
            "BLEU-4": round(corpus_bleu(refs, hyps, weights=(.25,.25,.25,.25),  smoothing_function=sf), 4),
        }

    metrics = {"greedy": bleu(references, hyp_greedy)}
    if use_beam:
        metrics["beam"] = bleu(references, hyp_beam)

    # METEOR — pass tokenized lists directly
    meteor_scores = []
    for refs, hyp in zip(references[:500], hyp_greedy[:500]):
        meteor_scores.append(meteor_score(refs, hyp))   # refs=list of token lists, hyp=token list
    metrics["greedy"]["METEOR"] = round(sum(meteor_scores)/len(meteor_scores), 4) if meteor_scores else 0.0

    print("\n=== RESULTS ===")
    for method, scores in metrics.items():
        print(f"[{method.upper()}]")
        for k, v in scores.items():
            print(f"  {k}: {v}")

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(results).to_csv(f"{output_dir}/predictions.csv", index=False)
    print(f"\nSaved → {output_dir}/metrics.json and predictions.csv")
    return metrics
