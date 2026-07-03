# PAPER: Section 5 -- donor (GLM-5.2 API output-space) & host hidden-state extractors. See RESULTS_TO_CODE.md
"""
Representation extractors for the graft-overlap analysis.

Setting (per user): the donor is GLM-5.2, available ONLY through a chat API
that does not expose hidden states. We therefore use its OUTPUT-space
representation: the next-token log-probability distribution over a fixed
probe set. This lands exactly at the logit-distillation level of the paper
(automatically basis-invariant). The host is the locally-served grafted
model, whose boundary hidden states CAN be extracted directly.

Both are reduced to a matrix H of shape (N_probes, dim):
  - donor: dim = top_k vocab logprob slots (aligned by a shared token id set)
  - host : dim = hidden width at the graft boundary layer

CKA compares them via N x N Gram matrices, so differing dim is fine.
linear_map_residual handles the unequal-dim alignment (paper Sec 4 remark).
"""
import os, json, time
import numpy as np
import requests


# ----------------------------------------------------------------------
# DONOR: GLM-5.2 via chat API, logprob distribution per probe
# ----------------------------------------------------------------------
class GLMDonor:
    """
    Pull a next-token logprob vector for each probe prompt.

    The API must support logprobs + top_logprobs (OpenAI-compatible style).
    We request the distribution at the FIRST generated token and read its
    top_logprobs list. To form a fixed-width matrix we project onto a shared
    vocabulary id set (the union of tokens seen across probes, capped at
    vocab_cap most frequent), filling unseen slots with a floor logprob.
    """
    def __init__(self, url=None, model="glm-5.2", api_key=None,
                 top_k=20, floor=-20.0, timeout=120):
        self.url = url or os.environ.get(
            "GLM_URL", "https://api.z.ai/api/paas/v4/chat/completions")
        self.model = model
        self.api_key = api_key or os.environ.get("GLM_API_KEY", "")
        self.top_k = top_k
        self.floor = floor
        self.timeout = timeout

    def _one(self, prompt):
        """Return dict token->logprob for the first generated token."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": self.top_k,
        }
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        r = requests.post(self.url, json=payload, headers=h,
                          timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        ch = data["choices"][0]
        lp = ch.get("logprobs") or {}
        content = lp.get("content") or []
        if not content:
            return {}
        top = content[0].get("top_logprobs") or []
        return {t["token"]: float(t["logprob"]) for t in top}

    def extract(self, probes, vocab_cap=512, verbose=True):
        """
        probes: list[str]. Returns H (N x V) logprob matrix and the token
        vocabulary (column order). Live per-probe output.
        """
        dists = []
        for i, p in enumerate(probes, 1):
            for attempt in range(3):
                try:
                    d = self._one(p)
                    break
                except Exception as e:
                    if attempt == 2:
                        if verbose:
                            print(f"[donor] probe {i} FAILED: {e}", flush=True)
                        d = {}
                    else:
                        time.sleep(1.0 * (attempt + 1))
            dists.append(d)
            if verbose:
                print(f"[donor] probe {i}/{len(probes)} "
                      f"got {len(d)} token logprobs", flush=True)
        # build shared vocab (most frequent tokens across probes)
        from collections import Counter
        cnt = Counter()
        for d in dists:
            cnt.update(d.keys())
        vocab = [tok for tok, _ in cnt.most_common(vocab_cap)]
        idx = {tok: j for j, tok in enumerate(vocab)}
        H = np.full((len(probes), len(vocab)), self.floor, dtype=np.float64)
        for i, d in enumerate(dists):
            for tok, val in d.items():
                if tok in idx:
                    H[i, idx[tok]] = val
        if verbose:
            print(f"[donor] H = {H.shape} over {len(vocab)} shared tokens",
                  flush=True)
        return H, vocab


# ----------------------------------------------------------------------
# HOST: locally-served grafted model, boundary hidden states
# ----------------------------------------------------------------------
class HostHidden:
    """
    Extract boundary-layer hidden states from the locally grafted model.

    Two backends:
      backend='hf'   : transformers model with output_hidden_states=True
                       (most reliable for true hidden states).
      backend='vllm' : if your vLLM build exposes a hidden-states/pooling
                       endpoint; otherwise prefer 'hf' for this analysis.

    We pool over the prompt tokens (mean of last-token is also offered) to get
    one vector per probe at the chosen layer.
    """
    def __init__(self, model_path, layer=-1, device="cuda",
                 pool="lasttok", dtype="bfloat16"):
        self.model_path = model_path
        self.layer = layer
        self.device = device
        self.pool = pool
        self.dtype = dtype
        self._model = None
        self._tok = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[self.dtype]
        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=dt, output_hidden_states=True,
            device_map=self.device)
        self._model.eval()

    def extract(self, probes, verbose=True):
        import torch
        if self._model is None:
            self._load()
        vecs = []
        for i, p in enumerate(probes, 1):
            enc = self._tok(p, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model(**enc)
            hs = out.hidden_states[self.layer][0]   # (seq, d)
            if self.pool == "lasttok":
                v = hs[-1]
            else:  # mean
                v = hs.mean(dim=0)
            vecs.append(v.float().cpu().numpy())
            if verbose:
                print(f"[host] probe {i}/{len(probes)} hidden d={v.shape[0]}",
                      flush=True)
        H = np.stack(vecs, axis=0)
        if verbose:
            print(f"[host] H = {H.shape} at layer {self.layer}", flush=True)
        return H
