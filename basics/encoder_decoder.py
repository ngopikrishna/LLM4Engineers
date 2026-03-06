"""
Simple character-level encoder-decoder (autoencoder) for text.
Pure numpy — no deep-learning framework required.

Architecture
------------
Encoder : one-hot chars (flattened) → linear + tanh → latent vector
Decoder : latent vector → linear + ReLU → linear → per-position logits

By flattening the full sequence into the encoder (instead of mean-pooling),
positional information is preserved and reconstruction works properly.

Usage
-----
    python encoder_decoder.py
"""

import numpy as np

RNG = np.random.default_rng(42)

# ── Vocabulary ─────────────────────────────────────────────────────────────

PAD, SOS, EOS = "<PAD>", "<SOS>", "<EOS>"

class Vocab:
    def __init__(self, texts):
        chars = sorted(set("".join(texts)))
        self.idx2ch = [PAD, SOS, EOS] + chars
        self.ch2idx = {ch: i for i, ch in enumerate(self.idx2ch)}
        self.size   = len(self.idx2ch)

    def encode(self, text, max_len):
        """Returns fixed-length int array: [SOS, chars..., EOS, PAD...]"""
        ids  = [self.ch2idx[SOS]]
        ids += [self.ch2idx.get(c, 0) for c in text[:max_len]]
        ids += [self.ch2idx[EOS]]
        ids += [self.ch2idx[PAD]] * (max_len + 2 - len(ids))
        return np.array(ids[:max_len + 2], dtype=np.int32)

    def decode(self, ids):
        out = []
        for i in ids:
            ch = self.idx2ch[int(i)]
            if ch == EOS:
                break
            if ch not in (PAD, SOS):
                out.append(ch)
        return "".join(out)


# ── Helpers ────────────────────────────────────────────────────────────────

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0.0, x)


# ── Model ──────────────────────────────────────────────────────────────────

class EncoderDecoder:
    """
    Parameters
    ----------
    vocab_size  V
    seq_len     T   (= max_len + 2, including SOS/EOS slots)
    latent_dim  H
    hidden_dim  D   (intermediate decoder hidden size)
    """
    def __init__(self, vocab_size, seq_len, latent_dim=64, hidden_dim=256):
        V, T, H, D = vocab_size, seq_len, latent_dim, hidden_dim
        sc = 0.02
        # Encoder: (T*V) → H
        self.W1 = RNG.standard_normal((T * V, H)).astype(np.float32) * sc
        self.b1 = np.zeros(H, dtype=np.float32)
        # Decoder layer 1: H → D
        self.W2 = RNG.standard_normal((H, D)).astype(np.float32) * sc
        self.b2 = np.zeros(D, dtype=np.float32)
        # Decoder layer 2: D → T*V  (logits)
        self.W3 = RNG.standard_normal((D, T * V)).astype(np.float32) * sc
        self.b3 = np.zeros(T * V, dtype=np.float32)
        self._V, self._T, self._H, self._D = V, T, H, D
        # Adam state
        self._t = 0
        self._m = {k: np.zeros_like(w) for k, w in
                   [("W1",self.W1),("b1",self.b1),("W2",self.W2),
                    ("b2",self.b2),("W3",self.W3),("b3",self.b3)]}
        self._v = {k: np.zeros_like(w) for k, w in
                   [("W1",self.W1),("b1",self.b1),("W2",self.W2),
                    ("b2",self.b2),("W3",self.W3),("b3",self.b3)]}

    def n_params(self):
        return sum(w.size for w in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, ids):
        """
        ids : (B, T) integer token ids
        Returns logits (B, T, V) and cache for backprop.
        """
        B = ids.shape[0]
        V, T, H, D = self._V, self._T, self._H, self._D

        # One-hot flatten: (B, T*V)
        oh   = np.zeros((B, T * V), dtype=np.float32)
        flat_idx = np.arange(T) * V + ids            # (B, T)
        for b in range(B):
            oh[b, flat_idx[b]] = 1.0

        # Encode
        pre1   = oh @ self.W1 + self.b1              # (B, H)
        latent = np.tanh(pre1)                        # (B, H)

        # Decode
        pre2   = latent @ self.W2 + self.b2          # (B, D)
        h2     = relu(pre2)                           # (B, D)
        pre3   = h2 @ self.W3 + self.b3              # (B, T*V)
        logits = pre3.reshape(B, T, V)               # (B, T, V)

        cache = dict(oh=oh, pre1=pre1, latent=latent,
                     pre2=pre2, h2=h2, B=B)
        return logits, cache

    # ── Backward + SGD ─────────────────────────────────────────────────────

    def step(self, ids, logits, cache, lr):
        B      = cache["B"]
        V, T, H, D = self._V, self._T, self._H, self._D
        oh     = cache["oh"]
        latent = cache["latent"]
        pre1   = cache["pre1"]
        h2     = cache["h2"]

        targets = ids                                  # (B, T)
        mask    = (targets != 0).astype(np.float32)   # ignore PAD
        N_eff   = mask.sum() + 1e-8

        # Softmax cross-entropy gradient
        probs   = softmax(logits)                      # (B, T, V)
        dlogits = probs.copy()
        dlogits[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1.0
        dlogits *= mask[:, :, None] / N_eff

        dpre3   = dlogits.reshape(B, T * V)           # (B, T*V)

        # Layer 3
        dW3  = h2.T @ dpre3                           # (D, T*V)
        db3  = dpre3.sum(0)
        dh2  = dpre3 @ self.W3.T                      # (B, D)

        # ReLU
        dpre2 = dh2 * (cache["pre2"] > 0)            # (B, D)

        # Layer 2
        dW2     = latent.T @ dpre2                    # (H, D)
        db2     = dpre2.sum(0)
        dlatent = dpre2 @ self.W2.T                   # (B, H)

        # tanh
        dpre1 = dlatent * (1 - latent**2)             # (B, H)

        # Layer 1
        dW1 = oh.T @ dpre1                            # (T*V, H)
        db1 = dpre1.sum(0)

        # Adam update
        β1, β2, ε = 0.9, 0.999, 1e-8
        self._t += 1
        for key, param, grad in [
            ("W1", self.W1, dW1), ("b1", self.b1, db1),
            ("W2", self.W2, dW2), ("b2", self.b2, db2),
            ("W3", self.W3, dW3), ("b3", self.b3, db3),
        ]:
            m = self._m[key]
            v = self._v[key]
            m *= β1;  m += (1 - β1) * grad
            v *= β2;  v += (1 - β2) * grad**2
            m_hat = m / (1 - β1**self._t)
            v_hat = v / (1 - β2**self._t)
            param -= lr * m_hat / (np.sqrt(v_hat) + ε)

        lp = -np.log(
            probs[np.arange(B)[:, None], np.arange(T)[None, :], targets] + 1e-9
        )
        return float((lp * mask).sum() / N_eff)

    # ── Inference ──────────────────────────────────────────────────────────

    def reconstruct(self, ids):
        logits, _ = self.forward(ids)
        return logits.argmax(-1)                       # (B, T)

    def encode(self, ids):
        _, cache = self.forward(ids)
        return cache["latent"]                         # (B, H)


# ── Training ───────────────────────────────────────────────────────────────

def train(model, encoded, epochs=300, batch_size=4, lr=0.03):
    N    = len(encoded)
    idxs = np.arange(N)

    for epoch in range(1, epochs + 1):
        RNG.shuffle(idxs)
        total, batches = 0.0, 0
        for s in range(0, N, batch_size):
            batch         = encoded[idxs[s:s + batch_size]]
            logits, cache = model.forward(batch)
            loss          = model.step(batch, logits, cache, lr)
            total        += loss
            batches      += 1
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d}  loss={total / batches:.4f}")


# ── Demo ───────────────────────────────────────────────────────────────────

TEXTS = [
    "hello world",
    "the quick brown fox",
    "encoder decoder model",
    "natural language processing",
    "deep learning is fun",
    "transformers changed everything",
    "attention is all you need",
    "learning representations",
    "text reconstruction task",
    "simple autoencoder demo",
]

MAX_LEN = 32   # max chars per sample (SOS + text + EOS + padding = MAX_LEN+2)

def main():
    vocab   = Vocab(TEXTS)
    encoded = np.stack([vocab.encode(t, MAX_LEN) for t in TEXTS])  # (N, T)
    T       = MAX_LEN + 2

    model = EncoderDecoder(
        vocab_size  = vocab.size,
        seq_len     = T,
        latent_dim  = 64,
        hidden_dim  = 256,
    )
    print(f"Vocab size : {vocab.size}")
    print(f"Seq len    : {T}  (max_len + SOS + EOS)")
    print(f"Parameters : {model.n_params():,}\n")

    train(model, encoded, epochs=300, batch_size=4, lr=1e-3)

    # ── Reconstruction ────────────────────────────────────────────────────
    print("\nReconstruction results:")
    print(f"{'Original':<35} {'Reconstructed'}")
    print("-" * 70)
    preds = model.reconstruct(encoded)
    for text, pred in zip(TEXTS, preds):
        print(f"{text:<35} {vocab.decode(pred)}")

    # ── Encode two phrases and show latent similarity ─────────────────────
    print("\nLatent cosine similarity (similar phrases should be closer):")
    latents = model.encode(encoded)
    # normalise
    norms   = np.linalg.norm(latents, axis=1, keepdims=True) + 1e-9
    normed  = latents / norms
    sim     = normed @ normed.T                        # (N, N) cosine sim

    labels = [t[:20] for t in TEXTS]
    print(f"{'':22}", end="")
    for lbl in labels:
        print(f"{lbl[:8]:>10}", end="")
    print()
    for i, lbl in enumerate(labels):
        print(f"{lbl[:22]:<22}", end="")
        for j in range(len(labels)):
            print(f"{sim[i,j]:10.3f}", end="")
        print()


if __name__ == "__main__":
    main()
