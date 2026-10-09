import numpy as np
import time



TIME_BUDGET = 20
D, WINDOW, K, BATCH, LR0, EPOCHS = 100, 5, 20, 512, 0.025, 50
SUBSAMPLE_T = 1e-3
def train(data: str):

    t0 = time.time()
    rng = np.random.default_rng(42)

    words = np.array(data.split(), dtype=np.str_)
    vocab, corpus, counts = np.unique(words, return_inverse=True, return_counts=True)
    corpus =   corpus.astype(np.int32)




    W_in, W_out = init_weights(len(vocab), D, rng)
    cdf = make_neg_table(counts)

    pairs_per_epoch = count_pairs(len(subsample(corpus, counts, SUBSAMPLE_T, rng)), WINDOW)
    total = EPOCHS * pairs_per_epoch
    processed = 0
    stop = False
    for ep in range(EPOCHS):
        corpus_ep = subsample(corpus, counts, SUBSAMPLE_T, rng)
        centers, contexts = make_pairs(corpus_ep, WINDOW)
        P = len(centers)
        perm = rng.permutation(P).astype(np.int32)
        for s in range(0, P, BATCH):

            if check_time(t0) :
                stop = True
                break

            idx = perm[s:s + BATCH]
            centers_b = centers[idx]
            contexts_b = contexts[idx]
            batch_size = len(idx)

            lr = learning_rate(processed, total, t0)

            negs = sample_negatives(cdf, K, rng)
            sgd_step(W_in, W_out, centers_b, contexts_b, negs, lr)
            processed += batch_size


        if stop :
            break

    W = W_in + W_out
    W = W - W.mean(axis=0)
    return {str(w): W[i] for i, w in enumerate(vocab)}

def get_random_float_numbers(count_unique_words : int , rng ) :
    return  rng.standard_normal((count_unique_words, 100), dtype=np.float32)

def learning_rate(processed, total, t0):
    progress = max(processed / total, (time.time() - t0) / TIME_BUDGET)
    return LR0 * max(1 - progress, 1e-4)


def check_time(t0) :
    return time.time() - t0 > TIME_BUDGET

def make_pairs(corpus, m):
    centers = []
    contexts = []
    for j in range(1, m + 1):
        centers.append(corpus[:-j])
        contexts.append(corpus[j:])
        centers.append(corpus[j:])
        contexts.append(corpus[:-j])
    return np.concatenate(centers), np.concatenate(contexts)

def count_pairs(N : np.int64  , m : np.int64 ) -> np.int64 :
    return 2 * ( m * N - m * ( m + 1) // 2 )


def sigmoid(z):
    z = np.clip(z, -10, 10)
    return 1.0 / (1.0 + np.exp(-z))


def init_weights(V, d, rng) :
    W_in = rng.uniform(-0.5  / d, 0.5/ d, size=(V, d)).astype(np.float32)
    W_out = np.zeros((V, d), dtype=np.float32)
    return W_in, W_out


def make_neg_table(counts) :
    p = counts ** 0.75
    res = p / p.sum()
    cdf = np.cumsum(res)
    cdf[-1] = 1.0
    return cdf


def sample_negatives(cdf, K, rng):
    r = rng.random(K)
    return np.searchsorted(cdf, r).astype(np.int32)

def sgd_step(W_in, W_out, centers, contexts, negs, lr):

    d = W_in.shape[1]

    v      = W_in[centers]       # (B, d)      центры из W_in
    u_pos  = W_out[contexts]        # (B, d)      контексты из W_out
    u_neg  = W_out[negs]      # (K, d)   негативы из W_out

    s_pos  = (v * u_pos).sum(axis=1)      # (B,)        скалярное произведение v и u_pos по оси d
    s_neg  = v @ u_neg.T

    g_pos  = sigmoid(s_pos) -1    # (B,)
    g_neg  = sigmoid(s_neg) - 0        # (B, K)

    grad_v = g_pos[:, None] * u_pos + g_neg @ u_neg  # (B, d)
    grad_u_pos = g_pos[:, None] * v  # (B, d)   без изменений
    grad_u_neg = g_neg.T @ v

    loss = -np.log(sigmoid(s_pos) + 1e-10).mean() - np.log(sigmoid(-s_neg) + 1e-10).sum(axis=1).mean()


    np.add.at(W_in , centers , -lr*grad_v)      # W_in  по centers
    np.add.at(W_out , contexts , -lr*grad_u_pos)      # W_out по contexts
    np.add.at(W_out, negs, -lr * grad_u_neg)

    return loss

def subsample(corpus, counts, t, rng) :
    f = counts / counts.sum()
    p_keep = np.minimum(1.0, np.sqrt(t / f) + t / f)  # (V,) - по слову
    mask = rng.random(len(corpus)) < p_keep[corpus]  # (N,) - fancy indexing: вероятность для каждого вхождения
    corpus_ep = corpus[mask]
    return corpus_ep