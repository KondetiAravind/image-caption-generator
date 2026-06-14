import os, sys, torch, argparse
sys.path.insert(0, os.path.dirname(__file__))

from src.model   import CaptionModel
from src.dataset import load_vocab

ROOT       = os.path.dirname(__file__)
VOCAB_PATH = os.path.join(ROOT, "data/vocab.pkl")
MODEL_PATH = os.path.join(ROOT, "models/best_model.pth")
TEST_CSV   = os.path.join(ROOT, "data/test.csv")
IMAGES_DIR = os.path.join(ROOT, "images")
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all():
    vocab = load_vocab(VOCAB_PATH)
    model = CaptionModel().to(device)
    ckpt  = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded | device={device} | val_loss={round(ckpt.get('val_loss',0),4)}")
    return model, vocab


def cmd_evaluate(args):
    from src.evaluate import evaluate
    model, vocab = load_all()
    evaluate(model, TEST_CSV, IMAGES_DIR, vocab, device,
             use_beam=not args.no_beam, beam_size=args.beam_size,
             max_samples=args.max_samples)


def cmd_caption(args):
    from src.inference  import greedy_caption, beam_search_caption
    from src.translate  import translate_to_all
    from torchvision    import transforms
    from PIL            import Image

    model, vocab = load_all()
    tf = transforms.Compose([
        transforms.Resize(320), transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tensor  = tf(Image.open(args.image).convert("RGB"))
    g, _    = greedy_caption(model, tensor, vocab, device)
    print(f"\nGreedy  : {g}")
    if not args.no_beam:
        b = beam_search_caption(model, tensor, vocab, device, beam_size=args.beam_size)
        print(f"Beam({args.beam_size}) : {b}")
    if args.translate:
        print("\n── Translations ──")
        for lang, text in translate_to_all(g).items():
            print(f"  {lang:10s}: {text}")


def cmd_visualize(args):
    from src.visualize import visualize_attention
    model, vocab = load_all()
    visualize_attention(model, args.image, vocab, device)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    pe = sub.add_parser("evaluate")
    pe.add_argument("--no_beam",     action="store_true")
    pe.add_argument("--beam_size",   type=int, default=5)
    pe.add_argument("--max_samples", type=int, default=None)

    pc = sub.add_parser("caption")
    pc.add_argument("--image",     required=True)
    pc.add_argument("--no_beam",   action="store_true")
    pc.add_argument("--beam_size", type=int, default=5)
    pc.add_argument("--translate", action="store_true")

    pv = sub.add_parser("visualize")
    pv.add_argument("--image", required=True)

    args = p.parse_args()
    if   args.cmd == "evaluate":  cmd_evaluate(args)
    elif args.cmd == "caption":   cmd_caption(args)
    elif args.cmd == "visualize": cmd_visualize(args)
    else: p.print_help()
