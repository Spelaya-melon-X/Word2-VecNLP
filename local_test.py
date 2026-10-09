import solution
import numpy as np
import time
text = open("text8").read()                      # или любой большой английский текст
data = " ".join(text.split()[:300_000])

t = time.time()
w2v = solution.train(data)
print("time:", round(time.time() - t, 1), "vocab:", len(w2v))

def nearest(word, w2v, topn=5):
    words = list(w2v.keys())
    M = np.stack([w2v[w] for w in words])
    M = M / np.linalg.norm(M, axis=1, keepdims=True)   # нормируем строки
    sims = M @ M[words.index(word)]                    # косинус со всеми словами
    best = np.argsort(-sims)[1:topn + 1]               # [0] - само слово
    return [(words[i], round(float(sims[i]), 3)) for i in best]

for w in ["king", "war", "city", "german", "music", "january"]:
    if w in w2v:
        print(w, "->", nearest(w, w2v))

def mean_random_cos(w2v, n=2000, seed=1):
    M = np.stack(list(w2v.values()))
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(M), n)
    j = rng.integers(0, len(M), n)
    return float((M[i] * M[j]).sum(axis=1).mean())

print("mean cos random pairs:", round(mean_random_cos(w2v), 3))