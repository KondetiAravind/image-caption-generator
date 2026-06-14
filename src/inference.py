import torch
import torch.nn.functional as F


def greedy_caption(model, image, vocab, device, max_len=50):
    model.eval()
    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]
    sos_idx  = word2idx["<SOS>"]
    eos_idx  = word2idx["<EOS>"]

    with torch.no_grad():
        image       = image.unsqueeze(0).to(device)
        encoder_out = model.encoder(image)
        h = torch.zeros(model.decoder.num_layers, 1, model.decoder.hidden_size).to(device)
        c = torch.zeros(model.decoder.num_layers, 1, model.decoder.hidden_size).to(device)
        input_word  = torch.tensor([sos_idx], dtype=torch.long).to(device)
        caption, alphas = [], []

        for _ in range(max_len):
            emb            = model.decoder.dropout(model.decoder.embedding(input_word))
            context, alpha = model.decoder.attention(encoder_out, h[-1])
            lstm_input     = torch.cat([emb, context], dim=1).unsqueeze(1)
            out, (h, c)    = model.decoder.lstm(lstm_input, (h, c))
            pred           = model.decoder.fc(model.decoder.dropout(out.squeeze(1)))
            word_idx       = pred.argmax(dim=1).item()
            alphas.append(alpha.squeeze(0).cpu())
            if word_idx == eos_idx:
                break
            word = idx2word.get(str(word_idx), idx2word.get(word_idx, "<UNK>"))
            caption.append(word)
            input_word = torch.tensor([word_idx], dtype=torch.long).to(device)

    return " ".join(caption), alphas


def beam_search_caption(model, image, vocab, device, beam_size=5, max_len=50):
    model.eval()
    word2idx = vocab["word2idx"]
    idx2word = vocab["idx2word"]
    sos_idx  = word2idx["<SOS>"]
    eos_idx  = word2idx["<EOS>"]

    with torch.no_grad():
        image       = image.unsqueeze(0).to(device)
        encoder_out = model.encoder(image)   # (1, 64, 256)

        h0 = torch.zeros(model.decoder.num_layers, 1, model.decoder.hidden_size).to(device)
        c0 = torch.zeros(model.decoder.num_layers, 1, model.decoder.hidden_size).to(device)

        # Each beam: (score, word_list, h, c)  — h/c shape: (num_layers, 1, hidden)
        beams     = [(0.0, [sos_idx], h0, c0)]
        completed = []

        for _ in range(max_len):
            if not beams:
                break
            new_beams = []
            for score, words, bh, bc in beams:
                if words[-1] == eos_idx:
                    completed.append((score, words))
                    continue
                inp        = torch.tensor([words[-1]], dtype=torch.long).to(device)
                emb        = model.decoder.dropout(model.decoder.embedding(inp))   # (1,256)
                ctx, _     = model.decoder.attention(encoder_out, bh[-1])          # (1,256)
                li         = torch.cat([emb, ctx], dim=1).unsqueeze(1)             # (1,1,512)
                out, (nh, nc) = model.decoder.lstm(li, (bh.contiguous(), bc.contiguous()))
                logp       = F.log_softmax(model.decoder.fc(
                                 model.decoder.dropout(out.squeeze(1))), dim=1)    # (1, vocab)
                topk_scores, topk_words = logp[0].topk(beam_size)
                for k in range(beam_size):
                    new_beams.append((
                        score + topk_scores[k].item(),
                        words + [topk_words[k].item()],
                        nh, nc
                    ))

            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
            if len(completed) >= beam_size:
                break

        if not completed:
            completed = beams
        best    = sorted(completed, key=lambda x: x[0], reverse=True)[0]
        caption = []
        for w in best[1][1:]:   # skip <SOS>
            if w == eos_idx:
                break
            caption.append(idx2word.get(str(w), idx2word.get(w, "<UNK>")))

    return " ".join(caption)
