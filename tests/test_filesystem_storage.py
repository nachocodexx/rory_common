import json
from pathlib import Path

import numpy as np
import pytest

from rory.core.enums.algorithms import Algorithm
from rory.core.security.dataowner import DataOwner
from rorycommon import LiuParams, Scheme, SourceType, StorageBuilder, StorageParams


def filesystem_backend(tmp_path: Path, *, num_chunks: int = 2):
    return (
        StorageBuilder(storage_path=str(tmp_path))
        .with_storage_params(StorageParams(num_chunks=num_chunks))
        .build()
    )


def test_none_client_selects_default_filesystem():
    backend = StorageBuilder().build()

    assert backend.client is None
    assert backend.is_filesystem
    assert backend.filesystem.root_path == Path("/rory/data")


@pytest.mark.asyncio
async def test_plaintext_round_trip_and_manifest(tmp_path):
    backend = filesystem_backend(tmp_path)
    matrix = np.arange(12, dtype=np.float32).reshape(6, 2)

    put = await backend.put("analytics", "matrix", matrix, tags={"owner": "rory"})
    get = await backend.get("analytics", "matrix")

    assert put.is_ok, put.unwrap_err()
    assert get.is_ok, get.unwrap_err()
    assert get.unwrap().source is SourceType.FILE
    np.testing.assert_array_equal(get.unwrap().raw_value, matrix)
    object_path = tmp_path / "analytics" / "matrix"
    assert put.unwrap().path == str(object_path)
    manifest = json.loads((object_path / "manifest.json").read_text())
    assert manifest["kind"] == "ndarray"
    assert manifest["tags"] == {"owner": "rory"}


@pytest.mark.asyncio
async def test_segmented_plaintext_round_trip(tmp_path):
    backend = filesystem_backend(tmp_path, num_chunks=3)
    matrix = np.arange(30, dtype=np.float64).reshape(10, 3)

    put = await backend.put("analytics", "segments", matrix, segment=True)
    get = await backend.get("analytics", "segments", segment=True)

    assert put.is_ok, put.unwrap_err()
    assert get.is_ok, get.unwrap_err()
    np.testing.assert_array_equal(get.unwrap().raw_value, matrix)
    manifest = json.loads((tmp_path / "analytics" / "segments" / "manifest.json").read_text())
    assert manifest["kind"] == "chunks"
    assert len(manifest["chunks"]) == 3


@pytest.mark.asyncio
async def test_get_rejects_incorrect_flags(tmp_path):
    backend = filesystem_backend(tmp_path)
    matrix = np.ones((4, 2))
    assert (await backend.put("analytics", "flags", matrix, segment=True)).is_ok

    result = await backend.get("analytics", "flags")

    assert result.is_err
    assert "Storage flags do not match" in str(result.unwrap_err())


@pytest.mark.asyncio
async def test_missing_ball_returns_file_not_found(tmp_path):
    backend = filesystem_backend(tmp_path)

    result = await backend.get("analytics", "missing")

    assert result.is_err
    assert isinstance(result.unwrap_err(), FileNotFoundError)


@pytest.mark.asyncio
async def test_put_atomically_replaces_existing_ball(tmp_path, monkeypatch):
    backend = filesystem_backend(tmp_path)
    original = np.arange(6).reshape(3, 2)
    replacement = np.arange(8).reshape(4, 2)
    assert (await backend.put("analytics", "replace", original)).is_ok

    write_bytes = backend.filesystem._write_bytes

    def fail_manifest(path, data):
        if ".manifest.json." in path.name:
            raise OSError("simulated manifest failure")
        write_bytes(path, data)

    monkeypatch.setattr(backend.filesystem, "_write_bytes", fail_manifest)
    failed = await backend.put("analytics", "replace", replacement)
    monkeypatch.setattr(backend.filesystem, "_write_bytes", write_bytes)

    assert failed.is_err
    current = await backend.get("analytics", "replace")
    assert current.is_ok, current.unwrap_err()
    np.testing.assert_array_equal(current.unwrap().raw_value, original)

    replaced = await backend.put("analytics", "replace", replacement)
    assert replaced.is_ok, replaced.unwrap_err()
    current = await backend.get("analytics", "replace")
    np.testing.assert_array_equal(current.unwrap().raw_value, replacement)
    generations = tmp_path / "analytics" / "replace" / "generations"
    assert len(list(generations.iterdir())) == 1


@pytest.mark.asyncio
async def test_delete_removes_ball_before_replacement(tmp_path):
    backend = filesystem_backend(tmp_path)
    first = np.zeros((2, 2))
    second = np.ones((2, 2))
    assert (await backend.put("analytics", "delete", first)).is_ok

    result = await backend.put("analytics", "delete", second, delete=True)

    assert result.is_ok, result.unwrap_err()
    restored = await backend.get("analytics", "delete")
    np.testing.assert_array_equal(restored.unwrap().raw_value, second)


@pytest.mark.asyncio
async def test_checksum_corruption_returns_error(tmp_path):
    backend = filesystem_backend(tmp_path)
    matrix = np.arange(4).reshape(2, 2)
    assert (await backend.put("analytics", "corrupt", matrix)).is_ok
    object_path = tmp_path / "analytics" / "corrupt"
    manifest = json.loads((object_path / "manifest.json").read_text())
    payload = object_path / "generations" / manifest["generation"] / "payload.bin"
    payload.write_bytes(b"corrupt")

    result = await backend.get("analytics", "corrupt")

    assert result.is_err
    assert "Checksum mismatch" in str(result.unwrap_err())


@pytest.mark.asyncio
@pytest.mark.parametrize("bucket_id,ball_id", [("../escape", "ball"), ("bucket", "../escape")])
async def test_path_traversal_is_rejected(tmp_path, bucket_id, ball_id):
    backend = filesystem_backend(tmp_path)

    result = await backend.put(bucket_id, ball_id, np.ones((2, 2)))

    assert result.is_err
    assert "Invalid" in str(result.unwrap_err())


@pytest.mark.asyncio
async def test_bucket_symlink_cannot_escape_storage_root(tmp_path):
    root = tmp_path / "storage"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "analytics").symlink_to(outside, target_is_directory=True)
    backend = filesystem_backend(root)

    result = await backend.put("analytics", "escape", np.ones((2, 2)))

    assert result.is_err
    assert "escapes the configured root" in str(result.unwrap_err())


@pytest.mark.asyncio
async def test_file_input_round_trip(tmp_path):
    source = tmp_path / "source.npy"
    matrix = np.arange(15, dtype=np.float64).reshape(5, 3)
    np.save(source, matrix)
    backend = filesystem_backend(tmp_path / "storage")

    put = await backend.put("analytics", "from-file", str(source))
    get = await backend.get("analytics", "from-file")

    assert put.is_ok, put.unwrap_err()
    assert get.is_ok, get.unwrap_err()
    np.testing.assert_array_equal(get.unwrap().raw_value, matrix)


@pytest.mark.asyncio
async def test_segmented_liu_encryption_uses_filesystem(tmp_path):
    owner = (
        DataOwner.with_scheme(Scheme.LIU)
        .with_scheme_params(LiuParams(seed=7, use_np_random=True))
        .build()
    )
    backend = (
        StorageBuilder(dataowner=owner, storage_path=str(tmp_path))
        .with_storage_params(StorageParams(num_chunks=2))
        .build()
    )
    matrix = np.arange(16, dtype=np.float64).reshape(8, 2)

    put = await backend.put("analytics", "liu", matrix, segment=True, encrypt=True)
    get = await backend.get("analytics", "liu", segment=True, encrypt=True)

    assert owner.algorithm is Algorithm.NONE
    assert put.is_ok, put.unwrap_err()
    assert get.is_ok, get.unwrap_err()
    decrypted = owner.primary_scheme.decrypt_matrix(get.unwrap().raw_value).data
    np.testing.assert_allclose(decrypted, matrix, atol=1e-12)


@pytest.mark.asyncio
async def test_prepared_ckks_round_trip(tmp_path, ckks, ckks_params):
    ciphertext = ckks.encrypt_vector(
        plaintext_vector=np.arange(4, dtype=np.float64)
    ).data
    backend = StorageBuilder(
        scheme=Scheme.CKKS,
        scheme_params=ckks_params,
        storage_path=str(tmp_path),
    ).build()

    put = await backend.put("analytics", "ckks", ciphertext, encrypt=True)
    get = await backend.get("analytics", "ckks", encrypt=True)

    assert put.is_ok, put.unwrap_err()
    assert get.is_ok, get.unwrap_err()
    assert len(get.unwrap().raw_value) == 1
