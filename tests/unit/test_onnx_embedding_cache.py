from __future__ import annotations

import threading
from pathlib import Path

from app.core.embedding.providers.onnx_fastembed import OnnxFastEmbedProvider


class CountingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]):
        self.calls.append(list(texts))
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


def loaded_provider(tmp_path: Path) -> tuple[OnnxFastEmbedProvider, CountingModel]:
    provider = OnnxFastEmbedProvider("unit-model", str(tmp_path), lazy_load=True)
    model = CountingModel()
    provider._model = model
    provider._dimensions = 2
    return provider, model


def test_repeat_text_reuses_the_same_vector(tmp_path: Path) -> None:
    provider, model = loaded_provider(tmp_path)
    first = provider.embed_batch(["túi vải", "canvas"])
    second = provider.embed_batch(["túi vải", "canvas"])

    assert [row.vector for row in second] == [row.vector for row in first]
    assert len(model.calls) == 1
    assert provider.cache_hits == 2
    assert provider.cache_misses == 2


def test_changed_text_is_a_cache_miss(tmp_path: Path) -> None:
    provider, model = loaded_provider(tmp_path)
    provider.embed("túi vải")
    provider.embed("túi vải mới")

    assert len(model.calls) == 2
    assert provider.cache_misses == 2


def test_disk_cache_survives_a_new_provider(tmp_path: Path) -> None:
    provider, _model = loaded_provider(tmp_path)
    original = provider.embed("túi vải").vector

    restarted, model = loaded_provider(tmp_path)
    restored = restarted.embed("túi vải").vector

    assert restored == original
    assert model.calls == []


def test_concurrent_calls_return_the_same_vector(tmp_path: Path) -> None:
    provider, model = loaded_provider(tmp_path)
    results: list[tuple[float, ...]] = []
    barrier = threading.Barrier(2)

    def run() -> None:
        barrier.wait()
        results.append(provider.embed("túi vải").vector)

    threads = [threading.Thread(target=run), threading.Thread(target=run)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results[0] == results[1]
    assert len(model.calls) == 1
