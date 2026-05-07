from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from agent.config import SearchConfig
from agent.search.schema import SearchHit


class RerankerBackendUnavailable(RuntimeError):
    """Raised when the optional real reranker backend cannot be loaded."""


class CrossEncoderModel(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> object:
        ...


class _CrossEncoderBackend:
    backend_name = "sentence_transformers_cross_encoder"

    def __init__(self, model: CrossEncoderModel):
        self.model = model

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        pairs = [(query, hit.text) for hit in hits]
        scores = _coerce_scores(self.model.predict(pairs), expected=len(pairs))
        ranked = sorted(
            zip(scores, hits, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )
        return [hit for _score, hit in ranked[:top_k]]


class LocalReranker:
    """Local reranker with optional CrossEncoder and deterministic fallback."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.backend = (
            "disabled" if not config.reranker_enabled else "deterministic_fallback"
        )
        self._backend: _CrossEncoderBackend | None = None
        self._warning: str | None = None
        self._fallback_reason: str | None = None
        if not config.reranker_enabled:
            return
        try:
            self._backend = _CrossEncoderBackend(_load_cross_encoder_model(config))
            self.backend = self._backend.backend_name
        except RerankerBackendUnavailable as exc:
            self._set_fallback(str(exc))
        except Exception as exc:
            self._set_fallback(f"{type(exc).__name__}: {exc}")

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not self.config.reranker_enabled:
            return hits[: self.config.reranker_top_k]
        limited = hits[: self.config.reranker_top_n_candidates]
        if self._backend is not None:
            try:
                return self._backend.rerank(
                    query,
                    limited,
                    top_k=self.config.reranker_top_k,
                )
            except Exception as exc:
                self._backend = None
                self._set_fallback(f"{type(exc).__name__}: {exc}")
        return self._deterministic_rerank(query, limited)

    def status(self) -> dict[str, object]:
        status = {
            "enabled": self.config.reranker_enabled,
            "configured_model": self.config.reranker_model,
            "backend": self.backend if self.config.reranker_enabled else "disabled",
            "top_n_candidates": self.config.reranker_top_n_candidates,
            "top_k": self.config.reranker_top_k,
            "max_length": self.config.reranker_max_length,
        }
        if self._warning:
            status["warning"] = self._warning
        if self._fallback_reason:
            status["fallback_reason"] = self._fallback_reason
        return status

    def _deterministic_rerank(
        self,
        query: str,
        hits: list[SearchHit],
    ) -> list[SearchHit]:
        query_terms = {term.strip(".,:;()[]").lower() for term in query.split() if term}

        def score(hit: SearchHit) -> float:
            text_terms = {
                term.strip(".,:;()[]").lower()
                for term in hit.text.split()
                if term.strip(".,:;()[]")
            }
            overlap = len(query_terms & text_terms)
            exact_id = 5 if hit.doc_id.lower() in query.lower() else 0
            return hit.score + overlap * 0.05 + exact_id

        return sorted(hits, key=score, reverse=True)[: self.config.reranker_top_k]

    def _set_fallback(self, reason: str) -> None:
        self.backend = "deterministic_fallback"
        self._warning = "real_reranker_unavailable"
        self._fallback_reason = reason


def _load_cross_encoder_model(config: SearchConfig) -> CrossEncoderModel:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RerankerBackendUnavailable(
            "sentence_transformers is not installed"
        ) from exc

    model_ref = _resolve_local_model_ref(config.reranker_model)
    kwargs = _cross_encoder_kwargs(CrossEncoder, config)
    try:
        return CrossEncoder(model_ref, **kwargs)
    except Exception as exc:
        raise RerankerBackendUnavailable(
            f"failed to load CrossEncoder model {config.reranker_model!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _resolve_local_model_ref(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RerankerBackendUnavailable(
            "huggingface_hub is not installed, so the local reranker model cache "
            "cannot be inspected"
        ) from exc
    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        raise RerankerBackendUnavailable(
            f"model {model_name!r} is not available in the local Hugging Face cache"
        )


def _cross_encoder_kwargs(
    cross_encoder: object,
    config: SearchConfig,
) -> dict[str, object]:
    try:
        parameters = inspect.signature(cross_encoder).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs: dict[str, object] = {}
    if "max_length" in parameters:
        kwargs["max_length"] = config.reranker_max_length
    if "trust_remote_code" in parameters:
        kwargs["trust_remote_code"] = True
    return kwargs


def _coerce_scores(raw_scores: object, *, expected: int) -> list[float]:
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if isinstance(raw_scores, int | float):
        scores = [float(raw_scores)]
    elif isinstance(raw_scores, str | bytes):
        raise ValueError("CrossEncoder returned a non-numeric score result")
    elif isinstance(raw_scores, Iterable):
        scores = [_coerce_single_score(score) for score in raw_scores]
    else:
        raise ValueError("CrossEncoder returned a non-iterable score result")
    if len(scores) != expected:
        raise ValueError(
            f"CrossEncoder returned {len(scores)} scores for {expected} candidates"
        )
    return scores


def _coerce_single_score(score: object) -> float:
    if hasattr(score, "tolist"):
        score = score.tolist()
    if isinstance(score, int | float):
        return float(score)
    if isinstance(score, str | bytes):
        raise ValueError(f"CrossEncoder returned an unsupported score value: {score!r}")
    if isinstance(score, Sequence) and len(score) == 1:
        return _coerce_single_score(score[0])
    raise ValueError(f"CrossEncoder returned an unsupported score value: {score!r}")
