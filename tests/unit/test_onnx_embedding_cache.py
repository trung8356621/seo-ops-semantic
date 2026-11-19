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


def loaded_provider(tmp_path: Path, *, max_items: int = 4096) -> tuple[OnnxFastEmbedProvider, CountingModel]:
    provider = OnnxFastEmbedProvider(
        "unit-model",
        str(tmp_path),
        lazy_load=True,
        memory_cache_max_items=max_items,
    )
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


def test_lru_cache_eviction_preserves_embedding_results(tmp_path: Path) -> None:
    provider, model = loaded_provider(tmp_path, max_items=3)
    expected = {text: provider.embed(text).vector for text in ["kw1", "kw2", "kw3"]}
    # Touch kw1 so kw2 becomes the least recently used.
    assert provider.embed("kw1").vector == expected["kw1"]

    expected["kw4"] = provider.embed("kw4").vector
    stats = provider.cache_stats()
    assert stats["size"] == 3
    assert stats["max_items"] == 3
    assert stats["evictions"] == 1

    # kw2 was evicted from memory, but disk restore returns identical vector without re-embedding.
    calls_before = len(model.calls)
    assert provider.embed("kw2").vector == expected["kw2"]
    assert len(model.calls) == calls_before
    assert provider.cache_stats()["size"] == 3


def test_zero_memory_cache_disables_in_memory_retention(tmp_path: Path) -> None:
    provider, _model = loaded_provider(tmp_path, max_items=0)
    v1 = provider.embed("alpha").vector
    v2 = provider.embed("beta").vector

    stats = provider.cache_stats()
    assert stats["size"] == 0
    assert stats["evictions"] == 0
    assert provider.embed("alpha").vector == v1
    assert provider.embed("beta").vector == v2


def test_concurrent_eviction_keeps_bound_and_exact_results(tmp_path: Path) -> None:
    provider, _model = loaded_provider(tmp_path, max_items=16)
    inputs = [f"keyword-{i % 40}" for i in range(160)]
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker(chunk: list[str]) -> None:
        try:
            barrier.wait()
            for item in chunk:
                vec = provider.embed(item).vector
                assert len(vec) == 2
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(inputs[i::8],)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    stats = provider.cache_stats()
    assert stats["size"] <= 16
    assert stats["evictions"] > 0


def test_concurrent_load_initializes_model_once_and_recovers_from_failure(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    import types

    created = 0
    should_fail = True

    class FakeTextEmbedding:
        def __init__(self, model_name: str, cache_dir: str) -> None:
            nonlocal created
            if should_fail:
                raise RuntimeError("transient model load failure")
            created += 1

        def embed(self, texts: list[str]):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    fake_mod = types.ModuleType("fastembed")
    fake_mod.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fake_mod)

    provider = OnnxFastEmbedProvider("unit-model", str(tmp_path), lazy_load=True)
    try:
        provider.load()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert provider.is_loaded is False
    assert provider.model_load_count == 0
    assert provider.last_load_error is not None

    should_fail = False
    barrier = threading.Barrier(4)
    threads = [
        threading.Thread(target=lambda: (barrier.wait(), provider.load()))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert provider.is_loaded is True
    assert provider.model_load_count == 1
    assert provider.last_load_error is None
    assert created == 1
