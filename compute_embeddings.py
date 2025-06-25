import os
import torch
import clip
from tqdm import tqdm
from dncbm.config import vocab_dir

def load_vocab(vocab_path):
    with open(vocab_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def compute_clip_embeddings(clip_model, device, texts):
    tokenized = clip.tokenize(texts).to(device)
    with torch.no_grad():
        embeddings = clip_model.encode_text(tokenized)
        embeddings = embeddings / torch.linalg.vector_norm(embeddings, dim=-1, keepdim=True)
    return embeddings

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model
    model, _ = clip.load("RN50", device=device)  # Change to other encoder if needed
    encoder_name = "clip_RN50"

    # Load vocabulary
    vocab_path = os.path.join(vocab_dir, "clipdissect_20k.txt")
    assert os.path.exists(vocab_path), f"Vocabulary file not found at {vocab_path}"
    vocab = load_vocab(vocab_path)

    # Compute embeddings in batches to avoid memory issues
    batch_size = 512
    all_embeddings = []

    for i in tqdm(range(0, len(vocab), batch_size)):
        batch = vocab[i:i + batch_size]
        emb = compute_clip_embeddings(model, device, batch)
        all_embeddings.append(emb.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Save embeddings
    output_path = os.path.join(vocab_dir, f"embeddings_{encoder_name}_clipdissect_20k.pth")
    torch.save(all_embeddings, output_path)
    print(f"Saved embeddings to {output_path}")

if __name__ == "__main__":
    main()
