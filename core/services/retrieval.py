import math
import re
from typing import Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from core.models import Product


VEHICLE_SYNONYMS = {
    "pulsar": "Bajaj Pulsar",
    "pulsar 150": "Bajaj Pulsar 150",
    "dominar": "Bajaj Dominar 400",
    "dominar 400": "Bajaj Dominar 400",
    "duke": "KTM Duke 390",
    "duke 390": "KTM Duke 390",
    "rc 390": "KTM RC 390",
    "apache": "TVS Apache RTR 160",
    "apache 160": "TVS Apache RTR 160",
    "apache 200": "TVS Apache RTR 200",
    "gixxer": "Suzuki Gixxer SF",
    "gixxer sf": "Suzuki Gixxer SF",
    "splendor": "Hero Splendor",
    "activa": "Honda Activa",
    "classic 350": "Royal Enfield Classic 350",
    "bullet": "Royal Enfield Bullet",
}

HINDI_ENGLISH_MAP = {
    "ke liye": "for",
    "ka": "",
    "ki": "",
    "chahiye": "need",
    "hai kya": "is available",
    "dikhao": "show",
    "kitna hai": "how much",
    "batao": "tell",
}


def normalize_query(text: str) -> str:
    """Preprocess query text, expanding vehicle nicknames and Hinglish terms."""
    q = text.lower()
    for h, repl in HINDI_ENGLISH_MAP.items():
        q = q.replace(h, repl)
    return q.strip()


class BM25Index:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc.split()) for doc in corpus]
        self.avg_doc_len = sum(self.doc_lens) / (self.corpus_size or 1)
        self.doc_freqs: dict[str, int] = {}
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]

        for doc_tokens in self.tokenized_corpus:
            unique_tokens = set(doc_tokens)
            for t in unique_tokens:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.idf: dict[str, float] = {}
        for token, df in self.doc_freqs.items():
            # Standard Lucene/BM25 IDF
            self.idf[token] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=float)
        for t in query_tokens:
            if t not in self.idf:
                continue
            idf_val = self.idf[t]
            for i, doc_tokens in enumerate(self.tokenized_corpus):
                tf = doc_tokens.count(t)
                if tf > 0:
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (self.doc_lens[i] / self.avg_doc_len))
                    scores[i] += idf_val * (numerator / denominator)
        return scores


class HybridProductRetriever:
    _instance = None

    def __init__(self):
        self.products: list[Product] = []
        self.sku_to_idx: dict[str, int] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.dense_embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Index] = None
        self._build_index()

    def _build_index(self):
        self.products = list(Product.objects.all().order_by("sku"))
        if not self.products:
            return

        self.sku_to_idx = {p.sku: i for i, p in enumerate(self.products)}

        # Build combined text per product
        docs = []
        for p in self.products:
            doc = (
                f"{p.sku} {p.name} {p.name} "
                f"{p.brand} {p.brand} "
                f"{p.category} "
                f"{p.vehicle_fitment} {p.vehicle_fitment} "
                f"{p.description}"
            )
            docs.append(doc)

        # 1. BM25 keyword index
        self.bm25 = BM25Index(docs)

        # 2. Dense semantic embedding via Sublinear TF-IDF + 64-dim TruncatedSVD
        max_df = 0.95 if len(docs) > 10 else 1.0
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=max_df,
        )
        tfidf_mat = self.vectorizer.fit_transform(docs)

        n_components = min(64, max(1, tfidf_mat.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense_vecs = self.svd.fit_transform(tfidf_mat)
        self.dense_embeddings = normalize(dense_vecs, norm="l2")

        # Optionally store in DB if missing
        missing_skus = [p.sku for p in self.products if p.embedding is None]
        if missing_skus:
            to_update = []
            for i, p in enumerate(self.products):
                if p.embedding is None:
                    p.embedding = self.dense_embeddings[i].tolist()
                    to_update.append(p)
            if to_update:
                Product.objects.bulk_update(to_update, ["embedding"], batch_size=100)

    def search(
        self,
        query: str,
        vehicle: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> list[Product]:
        """
        Executes hybrid retrieval combining dense semantic similarity and BM25 token matching.
        """
        if not self.products:
            self._build_index()
            if not self.products:
                return []

        norm_q = normalize_query(query)
        if vehicle:
            norm_q += f" {vehicle}"
        if category:
            norm_q += f" {category}"

        # 1. BM25 score
        q_tokens = re.findall(r"\w+", norm_q)
        bm25_scores = self.bm25.score(q_tokens) if self.bm25 else np.zeros(len(self.products))
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # 2. Dense semantic score
        dense_scores = np.zeros(len(self.products))
        if self.vectorizer and self.svd and self.dense_embeddings is not None:
            q_tfidf = self.vectorizer.transform([norm_q])
            q_dense = self.svd.transform(q_tfidf)
            q_dense = normalize(q_dense, norm="l2")
            dense_scores = np.dot(self.dense_embeddings, q_dense.T).flatten()

        # 3. Hybrid fusion (0.5 * BM25 + 0.5 * Dense)
        hybrid_scores = 0.5 * bm25_scores + 0.5 * dense_scores

        # 4. Filters & Vehicle bonus
        for i, p in enumerate(self.products):
            # Direct SKU match receives highest boost
            if p.sku.lower() in norm_q:
                hybrid_scores[i] += 2.0

            # Vehicle fitment match
            if vehicle and vehicle.lower() in p.vehicle_fitment.lower():
                hybrid_scores[i] += 0.5
            elif p.vehicle_fitment.lower() != "universal":
                # Check if query mentions this product's vehicle
                v_words = [w for w in p.vehicle_fitment.lower().split() if len(w) > 2]
                if v_words and any(vw in norm_q for vw in v_words):
                    hybrid_scores[i] += 0.3

            # Category filter
            if category and category.lower() not in p.category.lower():
                hybrid_scores[i] -= 1.0

        # Rank indices
        ranked_indices = np.argsort(-hybrid_scores)[:top_k]
        results = [self.products[idx] for idx in ranked_indices if hybrid_scores[idx] > 0.05]
        return results

    def is_ambiguous_part_query(self, query: str) -> bool:
        """
        Detects if query asks for a part without specifying a motorcycle make/model.
        """
        q = query.lower()
        part_keywords = ["brake pad", "brake shoe", "chain", "sprocket", "clutch", "oil filter", "air filter", "mirror"]
        has_part = any(pk in q for pk in part_keywords)

        has_vehicle = False
        for v in VEHICLE_SYNONYMS.keys():
            if v in q:
                has_vehicle = True
                break

        # Check for universal parts or explicit vehicle indicators
        if has_part and not has_vehicle and "universal" not in q:
            return True
        return False


_retriever = None


def get_retriever(force_refresh: bool = False) -> HybridProductRetriever:
    global _retriever
    curr_count = Product.objects.count()
    if _retriever is None or force_refresh or len(_retriever.products) != curr_count:
        _retriever = HybridProductRetriever()
    return _retriever
