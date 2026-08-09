from types import SimpleNamespace

import pytest

import rag_pipeline.store as store


class NamedCollection:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeClient:
    def __init__(self, collections, delete_error: Exception | None = None) -> None:
        self.collections = collections
        self.delete_error = delete_error
        self.deleted: list[str] = []

    def list_collections(self):
        return self.collections

    def delete_collection(self, name: str) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append(name)


def _config() -> SimpleNamespace:
    return SimpleNamespace(collection_name="grounded_pdf_chunks")


@pytest.mark.parametrize(
    "collections",
    [["grounded_pdf_chunks"], [NamedCollection("grounded_pdf_chunks")]],
)
def test_reset_collection_deletes_existing_collection(monkeypatch, collections):
    client = FakeClient(collections)
    monkeypatch.setattr(store, "get_client", lambda config: client)

    store.reset_collection(_config())

    assert client.deleted == ["grounded_pdf_chunks"]


def test_reset_collection_allows_already_absent_collection(monkeypatch):
    client = FakeClient(["another_collection"])
    monkeypatch.setattr(store, "get_client", lambda config: client)

    store.reset_collection(_config())

    assert client.deleted == []


def test_reset_collection_surfaces_deletion_failure(monkeypatch):
    client = FakeClient(
        ["grounded_pdf_chunks"], delete_error=RuntimeError("database is locked")
    )
    monkeypatch.setattr(store, "get_client", lambda config: client)

    with pytest.raises(RuntimeError, match="database is locked"):
        store.reset_collection(_config())


def test_reset_collection_with_persistent_chroma_client(tmp_path):
    config = SimpleNamespace(
        collection_name="grounded_pdf_chunks",
        chroma_dir=tmp_path / "chroma",
    )
    store.get_collection(config)

    store.reset_collection(config)

    remaining = {
        collection if isinstance(collection, str) else collection.name
        for collection in store.get_client(config).list_collections()
    }
    assert config.collection_name not in remaining
