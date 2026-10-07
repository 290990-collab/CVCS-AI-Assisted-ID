import torch
import numpy as np
import faiss
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.vision.models.vision_encoders import BaseVisionEncoder
from src.vision.data.preprocess import get_transform, load_image
from src.vision.data.loader import RPLANDataset


class VisionRetrievalPipeline:
    """Encoder-agnostic vision retrieval pipeline.

    Steps: extract_embeddings(), prepare_index() (optional head + PCA whitening),
    build_index() (FAISS), query() (top-k similar).
    """

    def __init__(
        self,
        encoder: BaseVisionEncoder,
        transform=None,
        device: str = "cpu"
    ):
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.device = device

        self.transform = transform if transform is not None else get_transform()

        self.raw_embeddings = None   # np.ndarray [N, D] RAW encoder embeddings
        self.embeddings  = None   # np.ndarray [N, D'] indexed gallery (after transforms)
        self.image_paths = []     # path i for row i
        self.index       = None   # FAISS index

        # transforms on top of the frozen raw: optional head + whitening
        self.head          = None   # ProjectionHead o None
        self.whiten_mean   = None   # np.ndarray [1, D]
        self.whiten_matrix = None   # np.ndarray [D, D']
        # query-only head after the whitening; None = off
        self.query_head    = None

    # --- embedding extraction ---

    def extract_embeddings(
        self,
        data_dir: str | None = None,
        batch_size: int = 128,
        save_path: str | None = None,
        image_paths: list | None = None,
    ) -> np.ndarray:
        """Extract embeddings [N, embedding_dim] for `data_dir` PNGs or an explicit `image_paths` list; saved to `save_path` if given."""
        dataset = RPLANDataset(
            data_dir=data_dir,
            image_paths=image_paths,
            transform=self.transform,
        )
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,        
            num_workers=6,
            pin_memory=(self.device != "cpu")
        )

        all_embeddings = []
        all_paths      = []

        print(f"[Retrieval] Estrazione embedding su {len(dataset)} immagini...")

        with torch.no_grad():
            for images, paths in tqdm(loader, desc="Encoding"): 
                images = images.to(self.device)
                embs   = self.encoder(images)               

                # order preserved between embeddings and paths
                all_embeddings.append(embs.cpu().numpy())
                all_paths.extend(paths)

        self.embeddings     = np.vstack(all_embeddings).astype("float32")  # [N, D]
        self.raw_embeddings = self.embeddings                               # the RAW is what gets saved
        self.image_paths    = all_paths

        print(f"[Retrieval] Embedding estratti: {self.embeddings.shape}")

        if save_path is not None:
            self._save(save_path)

        return self.embeddings

    # --- whitening (centering + PCA whitening) ---

    def prepare_index(
        self,
        head=None,
        whiten: bool = False,
        eps: float = 1e-6,
        whiten_dim: int | None = None,
        fit_rows: list[int] | np.ndarray | None = None,
    ):
        """Prepare the gallery from RAW embeddings: optional head, optional whitening, L2, then build the FAISS index.

        The same transforms apply to the query; all combinations reuse `raw_embeddings`.

        Args:
            fit_rows: rows to fit the whitening on; None = whole gallery (transductive). With
                train rows the statistics come from train only; the indexed gallery stays whole.
        """
        assert self.raw_embeddings is not None, \
            "Chiama extract_embeddings() o load() prima di prepare_index()"

        self.head = head
        gallery = self._apply_head(self.raw_embeddings)

        if whiten:
            fit_on = gallery
            if fit_rows is not None:
                rows = np.asarray(fit_rows, dtype=np.int64)
                if rows.size == 0:
                    raise ValueError(
                        "fit_rows è vuoto: il whitening non ha righe su cui stimare. "
                        "Controlla whitening.fit_split e la presenza dei metadati .mat."
                    )
                fit_on = gallery[rows]
            print(f"[Retrieval] Whitening stimato su {len(fit_on)}/{len(gallery)} "
                  f"righe della gallery")
            self._fit_whitening(fit_on, eps=eps, dim=whiten_dim)
        else:
            self.whiten_mean = self.whiten_matrix = None

        self.embeddings = self._apply_whitening(gallery)
        self.build_index()

    def _apply_head(self, embs: np.ndarray) -> np.ndarray:
        """Project embeddings with the head (batched, no_grad); identity if head is None."""
        if self.head is None:
            return embs.astype("float32")
        self.head.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(embs), 8192):
                x = torch.from_numpy(embs[i:i + 8192]).float().to(self.device)
                out.append(self.head(x).cpu().numpy())
        return np.vstack(out).astype("float32")

    def _fit_whitening(self, embs: np.ndarray, eps: float = 1e-6, dim: int | None = None):
        """PCA whitening parameters (mean + matrix) on `embs` (gallery after head); `eps` regularises near-zero variance, `dim` keeps the top components."""
        self.whiten_mean = embs.mean(axis=0, keepdims=True).astype("float32")
        centered = embs - self.whiten_mean

        # covariance eigendecomposition (D x D)
        N = centered.shape[0]
        cov = (centered.T @ centered) / max(N - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # eigh is ascending: sort by decreasing variance, keep the top `dim`
        order = np.argsort(eigvals)[::-1]
        if dim is not None:
            order = order[:int(dim)]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # project on principal components, scale by 1/sigma: unit variance
        self.whiten_matrix = (eigvecs / np.sqrt(eigvals + eps)).astype("float32")
        print(f"[Retrieval] Whitening fittato: matrix {self.whiten_matrix.shape}")

    def _apply_whitening(self, embs: np.ndarray) -> np.ndarray:
        """Centering + PCA whitening + L2 normalisation; L2 only if whitening is not set."""
        embs = embs.astype("float32")
        if self.whiten_mean is not None and self.whiten_matrix is not None:
            embs = (embs - self.whiten_mean) @ self.whiten_matrix
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return (embs / np.maximum(norms, 1e-12)).astype("float32")

    # --- FAISS index ---

    def build_index(self) -> faiss.Index:
        """Build a FAISS IndexFlatIP (inner product)."""
        assert self.embeddings is not None, \
            "Chiama extract_embeddings() prima di build_index()"

        dim         = self.embeddings.shape[1]
        self.index  = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        print(f"[Retrieval] Indice FAISS costruito: {self.index.ntotal} vettori, dim={dim}")
        return self.index

    # --- query ---

    def load_query(self, path: str) -> torch.Tensor:
        """Load a query PNG with the transform: tensor [3, H, W]."""
        return load_image(path, self.transform)

    def query(
        self,
        query_image: torch.Tensor,
        top_k: int = 5,
        return_embeddings: bool = False,
    ) -> list[dict]:
        """Top-k most similar plans for a query tensor [3, H, W] or [1, 3, H, W].

        Returns a list of {'path', 'score' (cosine similarity)}; with `return_embeddings`,
        (results, query_raw [1, D], query_final [1, D']) from this forward.
        """
        assert self.index is not None, \
            "Chiama build_index() prima di query()"

        if query_image.dim() == 3:
            query_image = query_image.unsqueeze(0)

        with torch.no_grad():
            query_emb = self.encoder(query_image.to(self.device))
            query_np  = query_emb.cpu().numpy().astype("float32")

        query_raw = query_np   # [1, D] RAW encoder output

        # same transforms as the gallery
        query_np = self._apply_head(query_np)
        query_np = self._apply_whitening(query_np)
        if getattr(self, "query_head", None) is not None:   # the gallery is not transformed
            with torch.no_grad():
                query_np = self.query_head(torch.from_numpy(query_np).float().to(self.device)) \
                    .cpu().numpy().astype("float32")

        scores, indices = self.index.search(query_np, top_k)     

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "path":  self.image_paths[idx],
                "score": float(score)       
            })

        if return_embeddings:
            return results, query_raw, query_np
        return results

    # --- disk save/load ---

    def _save(self, save_dir: str):
        """Save RAW embeddings and paths; whitening and head are not persisted here (applied by prepare_index / trained apart)."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / "embeddings.npy", self.raw_embeddings)
        with open(save_dir / "image_paths.json", "w") as f:
            json.dump(self.image_paths, f)

        print(f"[Retrieval] Salvato in {save_dir}")


    def load(self, save_dir: str):
        """Load RAW embeddings and paths; the index is built by the caller via prepare_index."""
        save_dir = Path(save_dir)

        self.raw_embeddings = np.load(save_dir / "embeddings.npy")
        self.embeddings     = self.raw_embeddings
        with open(save_dir / "image_paths.json") as f:
            self.image_paths = json.load(f)

        print(f"[Retrieval] Caricato da {save_dir}: {len(self.image_paths)} floor plan "
              f"(raw {self.raw_embeddings.shape})")
