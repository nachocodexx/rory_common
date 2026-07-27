import pytest
import numpy as np
import pandas as pd
from uuid import uuid4
from Pyfhel import PyCtxt
from rorycommon import Common, StorageBuilder, StorageBackend, StorageParams, Scheme, CkksParams, LiuParams
from rory.core.enums import Algorithm
from rory.core.security.dataowner import DataOwner
from rory.core.utils.utils import Utils
from option import Ok

import os
RORY_KEYS_PATH             = os.environ.get("RORY_TEST_KEYS_PATH", "/tmp/rory/keys/test2")
RORY_COMMON_CTX_FILENAME   = os.environ.get("RORY_COMMON_CTX_FILENAME", "ctx")
RORY_COMMON_PUBKEY_FILENAME = os.environ.get("RORY_COMMON_PUBKEY_FILENAME", "pubkey")
RORY_COMMON_SECRETKEY_FILENAME = os.environ.get("RORY_COMMON_SECRETKEY_FILENAME", "secretkey")
RORY_COMMON_RELINKEY_FILENAME  = os.environ.get("RORY_COMMON_RELINKEY_FILENAME", "relinkey")
RORY_COMMON_ROTATEKEY_FILENAME = os.environ.get("RORY_COMMON_ROTATEKEY_FILENAME", "rotatekey")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_matrix():
    return np.random.random((4, 4)).astype(np.float64)


@pytest.fixture
def small_vector():
    return np.random.random((16,)).astype(np.float64)


@pytest.fixture
def tmp_npy_vector_file(tmp_path, small_vector):
    path = tmp_path / "vector.npy"
    np.save(path, small_vector)
    return str(path), "npy"


@pytest.fixture
def storage_ids():
    iid = uuid4().hex[:6]
    return {"bucket_id": f"test{iid}", "ball_id": f"ball{iid}"}


# @pytest.fixture
# def paillier_do():
#     do = DataOwnerPHE(securitylevel=128)
#     do.generate_keys()
#     return do


@pytest.fixture
def tmp_npy_file(tmp_path, small_matrix):
    path = tmp_path / "matrix.npy"
    np.save(path, small_matrix)
    return str(path), "npy"


@pytest.fixture
def tmp_csv_file(tmp_path, small_matrix):
    path = tmp_path / "matrix.csv"
    pd.DataFrame(small_matrix).to_csv(path, header=False, index=False)
    return str(path), "csv"


def ckks_builder(client, ckks, ckks_params):
    """StorageBuilder wired for CKKS with full key config."""
    return StorageBuilder(
        storage_client = client,
        scheme         = Scheme.CKKS,
        scheme_params  = ckks_params,
    ).build()


def liu_builder(client, liu_params):
    return StorageBuilder(storage_client=client, scheme=Scheme.LIU, scheme_params=liu_params).build()


# ---------------------------------------------------------------------------
# StorageParams — unit tests (no network)
# ---------------------------------------------------------------------------

def test_storage_params_defaults():
    p = StorageParams()
    assert p.backoff_factor == 0.5
    assert p.num_chunks == 2
    assert p.chunk_index == 0
    assert p.chunk_size == "256kb"
    assert p.delay == 1
    assert p.force is True
    assert p.headers == {}
    assert p.http2 is False
    assert p.max_attempts == 5
    assert p.max_parallel_gets == 10
    assert p.timeout == 300


def test_storage_params_custom():
    p = StorageParams(num_chunks=4, timeout=60, chunk_size="128kb")
    assert p.num_chunks == 4
    assert p.timeout == 60
    assert p.chunk_size == "128kb"


# ---------------------------------------------------------------------------
# StorageBuilder — unit tests (no network)
# ---------------------------------------------------------------------------

async def test_builder_with_scheme(client):
    backend = (
        StorageBuilder(storage_client=client, scheme=Scheme.CKKS)
        .with_scheme(Scheme.LIU)
        .with_scheme_params(LiuParams(seed=7))
        .build()
    )
    assert backend.scheme == Scheme.LIU


async def test_builder_with_dataowner(client):
    owner = (
        DataOwner.with_scheme(Scheme.LIU)
        .with_scheme_params(LiuParams(seed=7))
        .build()
    )
    backend = StorageBuilder(storage_client=client).with_dataowner(owner).build()
    assert backend.dataowner is owner
    assert backend.scheme == Scheme.LIU


async def test_storage_params_applied(client):
    params = StorageParams(backoff_factor=1.5, num_chunks=4, timeout=60)
    backend = (
        StorageBuilder(storage_client=client, scheme=Scheme.CKKS)
        .with_storage_params(params)
        .build()
    )
    assert backend.params.backoff_factor == 1.5
    assert backend.params.num_chunks == 4
    assert backend.params.timeout == 60


async def test_builder_defaults_to_storage_params(client):
    backend = StorageBuilder(storage_client=client, scheme=Scheme.CKKS).build()
    assert isinstance(backend.params, StorageParams)


async def test_builder_ckks_key_config(client, ckks, ckks_params):
    backend = ckks_builder(client, ckks, ckks_params)
    assert backend.scheme_params.keys_path == RORY_KEYS_PATH
    assert backend.scheme_params is ckks_params


async def test_builder_rejects_mixed_configuration(client):
    owner = DataOwner.with_scheme(Scheme.LIU).build()
    with pytest.raises(ValueError, match="mutually exclusive"):
        StorageBuilder(storage_client=client, dataowner=owner, scheme=Scheme.LIU).build()


async def test_builder_dataowner_round_trip(client):
    owner = DataOwner.with_scheme(Scheme.LIU).build()
    backend = StorageBuilder(storage_client=client, dataowner=owner).build()
    cloned = backend.as_builder().build()
    assert cloned.scheme == Scheme.LIU
    assert cloned.dataowner is owner


@pytest.mark.asyncio
async def test_segment_encrypt_put_uses_parallel_dataowner_pipeline(
    client, liu_params, small_matrix, storage_ids, monkeypatch
):
    uploaded = {}

    async def fake_put_chunks(**kwargs):
        uploaded["chunks"] = kwargs["chunks"]
        return Ok(True)

    monkeypatch.setattr(Common, "put_chunks_no_delete", fake_put_chunks)
    backend = (
        StorageBuilder(
            storage_client=client,
            scheme=Scheme.LIU,
            scheme_params=liu_params,
        )
        .with_storage_params(StorageParams(num_chunks=2))
        .build()
    )

    result = await backend.put(
        **storage_ids,
        data=small_matrix,
        segment=True,
        encrypt=True,
    )

    assert result.is_ok, result.unwrap_err()
    assert len(uploaded["chunks"]) == 2


@pytest.mark.asyncio
async def test_encrypt_without_segment_uses_one_logical_chunk(
    client, liu_params, small_matrix, storage_ids, monkeypatch
):
    uploaded = {}

    async def fake_put_chunks(**kwargs):
        uploaded["chunks"] = kwargs["chunks"]
        return Ok(True)

    monkeypatch.setattr(Common, "put_chunks_no_delete", fake_put_chunks)
    backend = StorageBuilder(
        storage_client=client,
        scheme=Scheme.LIU,
        scheme_params=liu_params,
    ).build()

    result = await backend.put(**storage_ids, data=small_matrix, encrypt=True)

    assert result.is_ok, result.unwrap_err()
    assert len(uploaded["chunks"]) == 1


@pytest.mark.asyncio
async def test_prepared_ckks_is_uploaded_without_reencryption(
    client, ckks, ckks_params, small_vector, storage_ids, monkeypatch
):
    ciphertext = ckks.encrypt_vector(plaintext_vector=small_vector).data
    owner = (
        DataOwner.with_scheme(Scheme.CKKS)
        .with_scheme_params(ckks_params)
        .build()
    )

    def unexpected_encryption(*args, **kwargs):
        raise AssertionError("prepared PyCtxt must not be encrypted again")

    async def fake_put_chunks(**kwargs):
        return Ok(True)

    monkeypatch.setattr(owner, "outsourcedData", unexpected_encryption)
    monkeypatch.setattr(Common, "put_chunks_no_delete", fake_put_chunks)
    backend = StorageBuilder(storage_client=client, dataowner=owner).build()

    result = await backend.put(
        **storage_ids,
        data=ciphertext,
        segment=True,
        encrypt=True,
    )

    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().encrypt_time == 0.0


@pytest.mark.asyncio
async def test_storage_uses_algorithm_dataowner_scheme_only(
    client,
    small_matrix,
    storage_ids,
    monkeypatch,
):
    uploaded = {}

    async def fake_put_chunks(**kwargs):
        uploaded["chunks"] = kwargs["chunks"]
        return Ok(True)

    monkeypatch.setattr(Common, "put_chunks_no_delete", fake_put_chunks)
    owner = (
        DataOwner.with_algorithm(Algorithm.SKMEANS)
        .with_scheme(Scheme.LIU)
        .with_scheme_params(LiuParams(seed=7))
        .build()
    )
    backend = StorageBuilder(storage_client=client, dataowner=owner).build()

    result = await backend.put(**storage_ids, data=small_matrix, encrypt=True)

    assert result.is_ok, result.unwrap_err()
    encrypted = np.concatenate(
        [chunk.to_ndarray().unwrap() for chunk in uploaded["chunks"]],
        axis=0,
    )
    decrypted = owner.primary_scheme.decrypt_matrix(encrypted).data
    np.testing.assert_allclose(decrypted, small_matrix, atol=1e-12)


@pytest.mark.asyncio
async def test_storage_does_not_implement_paillier(client, small_matrix, storage_ids):
    backend = StorageBuilder(storage_client=client, scheme=Scheme.PAILLIER).build()

    result = await backend.put(**storage_ids, data=small_matrix, encrypt=True)

    assert result.is_err
    assert isinstance(result.unwrap_err(), NotImplementedError)


@pytest.mark.asyncio
async def test_delete_runs_before_put(client, small_matrix, storage_ids, monkeypatch):
    calls = []

    async def fake_delete(**kwargs):
        calls.append("delete")

    async def fake_put(**kwargs):
        calls.append("put")
        return Ok(True)

    monkeypatch.setattr(Common, "while_not_delete_ball_id", fake_delete)
    monkeypatch.setattr(Common, "put_ndarray_no_delete", fake_put)
    backend = StorageBuilder(storage_client=client).build()

    result = await backend.put(**storage_ids, data=small_matrix, delete=True)

    assert result.is_ok, result.unwrap_err()
    assert calls == ["delete", "put"]


# ---------------------------------------------------------------------------
# Default put/get (no segment, no encrypt) — single blob
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_default_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=small_matrix)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_default_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    await backend.put(**storage_ids, data=small_matrix)
    result = await backend.get(**storage_ids)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None
    assert isinstance(result.unwrap().raw_value, np.ndarray)


@pytest.mark.asyncio
async def test_put_default_liu(client, liu_params, small_matrix, storage_ids):
    backend = liu_builder(client, liu_params)
    result = await backend.put(**storage_ids, data=small_matrix)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_default_liu(client, liu_params, small_matrix, storage_ids):
    backend = liu_builder(client, liu_params)
    await backend.put(**storage_ids, data=small_matrix)
    result = await backend.get(**storage_ids)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None
    assert isinstance(result.unwrap().raw_value, np.ndarray)


# ---------------------------------------------------------------------------
# Segment only (segment=True, encrypt=False) — chunked, unencrypted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_segment_only_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=small_matrix, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_segment_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend: StorageBackend = ckks_builder(client, ckks, ckks_params)
    bucket_id = storage_ids["bucket_id"]
    ball_id = storage_ids["ball_id"]
    res = await backend.put(bucket_id=bucket_id, ball_id=ball_id, data=small_matrix, segment=True)
    assert res.is_ok, res.unwrap_err()
    result = await backend.get(bucket_id=bucket_id, ball_id=ball_id, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None


@pytest.mark.asyncio
async def test_put_segment_only_liu(client, liu_params, small_matrix, storage_ids):
    backend = liu_builder(client, liu_params)
    result = await backend.put(**storage_ids, data=small_matrix, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_segment_liu(client, liu_params, small_matrix, storage_ids):
    backend = liu_builder(client, liu_params)
    await backend.put(**storage_ids, data=small_matrix, segment=True)
    result = await backend.get(**storage_ids, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None


# ---------------------------------------------------------------------------
# Segment + encrypt — CKKS and LIU
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_encrypt_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_get_encrypt_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    put = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert put.is_ok, put.unwrap_err()
    result = await backend.get(**storage_ids, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    value = result.unwrap()
    assert value.raw_value is not None
    assert isinstance(value.raw_value, list)
    assert all(isinstance(x, PyCtxt) for x in value.raw_value)


@pytest.mark.asyncio
async def test_put_encrypt_liu(client, liu_params, small_matrix, storage_ids):
    backend = liu_builder(client, liu_params)
    result = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_get_encrypt_liu(client, liu_params, small_matrix, storage_ids):
    backend = liu_builder(client, liu_params)
    put = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert put.is_ok, put.unwrap_err()
    result = await backend.get(**storage_ids, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None
    assert isinstance(result.unwrap().raw_value, np.ndarray)


@pytest.mark.asyncio
async def test_put_prepared_udm(client, small_matrix, storage_ids):
    udm = Utils.calculate_UDM(plaintext_matrix=small_matrix)
    backend = StorageBuilder(storage_client=client).build()
    result = await backend.put(**storage_ids, data=udm, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == udm.shape


@pytest.mark.asyncio
async def test_put_get_prepared_udm(client, small_matrix, storage_ids):
    udm = Utils.calculate_UDM(plaintext_matrix=small_matrix)
    backend = StorageBuilder(storage_client=client).build()
    put = await backend.put(**storage_ids, data=udm, segment=True)
    assert put.is_ok, put.unwrap_err()
    result = await backend.get(**storage_ids, segment=True)
    assert result.is_ok, result.unwrap_err()
    value = result.unwrap()
    assert value.raw_value is not None
    assert isinstance(value.raw_value, np.ndarray)
    assert value.raw_value.shape == udm.shape


@pytest.mark.asyncio
async def test_get_prepared_udm_uses_get_and_merge(client, monkeypatch):
    backend = StorageBuilder(storage_client=client).build()
    expected = np.arange(4, dtype=np.float64).reshape(2, 2)
    calls = {"get_and_merge": 0, "get_pyctxt": 0}

    async def fake_get_and_merge(**kwargs):
        calls["get_and_merge"] += 1
        assert kwargs["bucket_id"] == "bucket"
        assert kwargs["key"] == "ball"
        return expected

    async def fake_get_pyctxt(**kwargs):
        calls["get_pyctxt"] += 1
        raise AssertionError("Prepared UDM get should not use CKKS retrieval")

    monkeypatch.setattr("rorycommon.Common.get_and_merge", fake_get_and_merge)
    monkeypatch.setattr("rorycommon.Common.get_pyctxt", fake_get_pyctxt)

    result = await backend.get(bucket_id="bucket", ball_id="ball", segment=True)

    assert result.is_ok, result.unwrap_err()
    assert calls == {"get_and_merge": 1, "get_pyctxt": 0}
    assert np.array_equal(result.unwrap().raw_value, expected)


# ---------------------------------------------------------------------------
# Pre-processed TList — CKKS List[PyCtxt]
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_tlist_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    ciphertexts = ckks.encrypt_matrix(plaintext_matrix=small_matrix).data
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=ciphertexts, encrypt=True)
    assert result.is_ok, result.unwrap_err()


# ---------------------------------------------------------------------------
# put_from_file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_from_file_default_ckks(client, ckks, ckks_params, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_encrypt_ckks(client, ckks, ckks_params, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_default_liu(client, liu_params, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = liu_builder(client, liu_params)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_encrypt_liu(client, liu_params, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = liu_builder(client, liu_params)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_csv_default_ckks(client, ckks, ckks_params, storage_ids, tmp_csv_file):
    path, ext = tmp_csv_file
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext)
    assert result.is_ok, result.unwrap_err()


# ---------------------------------------------------------------------------
# Vector (1-D ndarray) — CKKS encrypt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_encrypt_ckks_vector(client, ckks, ckks_params, small_vector, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=small_vector, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_get_encrypt_ckks_vector(client, ckks, ckks_params, small_vector, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    put = await backend.put(**storage_ids, data=small_vector, encrypt=True)
    assert put.is_ok, put.unwrap_err()
    result = await backend.get(**storage_ids, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    value = result.unwrap()
    assert value.raw_value is not None
    assert isinstance(value.raw_value, list)
    assert all(isinstance(x, PyCtxt) for x in value.raw_value)


@pytest.mark.asyncio
async def test_put_from_file_encrypt_ckks_vector(client, ckks, ckks_params, storage_ids, tmp_npy_vector_file):
    path, ext = tmp_npy_vector_file
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext, encrypt=True)
    assert result.is_ok, result.unwrap_err()


# ---------------------------------------------------------------------------
# delete flag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_delete_flag_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    # First put to create the object, then overwrite with delete=True
    first = await backend.put(**storage_ids, data=small_matrix)
    assert first.is_ok, first.unwrap_err()
    result = await backend.put(**storage_ids, data=small_matrix, delete=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_delete_flag_on_nonexistent_key(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    # delete=True on a key that doesn't exist yet must not error
    result = await backend.put(**storage_ids, data=small_matrix, delete=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_delete_flag(client, ckks, ckks_params, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = ckks_builder(client, ckks, ckks_params)
    first = await backend.put_from_file(**storage_ids, path=path, extension=ext)
    assert first.is_ok, first.unwrap_err()
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext, delete=True)
    assert result.is_ok, result.unwrap_err()

@pytest.mark.asyncio
async def test_segmented_ckks_vector_prepared_ciphertext_round_trip(
    client,
    ckks,
    ckks_params,
    storage_ids,
    small_vector,
):
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(
        bucket_id = storage_ids["bucket_id"],
        ball_id   = storage_ids["ball_id"],
        data      = small_vector,
        encrypt   = True,
        segment   = True,
        delete    = True
    )
    assert result.is_ok, result.unwrap_err()

    result = await backend.get(
        bucket_id = storage_ids["bucket_id"],
        ball_id   = storage_ids["ball_id"],
        encrypt   = True,
        segment   = True,
    )
    assert result.is_ok, result.unwrap_err()
    raw_value = result.unwrap().raw_value
    expected_segments = np.array_split(small_vector, backend.params.num_chunks)
    assert len(raw_value) == len(expected_segments)
    assert all(isinstance(value, PyCtxt) for value in raw_value)
    decrypted = np.concatenate([
        ckks.decrypt_list([ciphertext], take=len(expected))[0]
        for ciphertext, expected in zip(raw_value, expected_segments)
    ])
    np.testing.assert_allclose(decrypted, small_vector, atol=1e-2)

    result = await backend.put(
        bucket_id = storage_ids["bucket_id"],
        ball_id   = storage_ids["ball_id"],
        data      = raw_value,
        delete    = True,
        segment   = True,
        encrypt   = True,
    )
    assert result.is_ok, result.unwrap_err()

    result = await backend.get(
        bucket_id = storage_ids["bucket_id"],
        ball_id   = storage_ids["ball_id"],
        encrypt   = True,
        segment   = True
    )
    assert result.is_ok, result.unwrap_err()
    raw_value2 = result.unwrap().raw_value
    assert len(raw_value) == len(raw_value2)
    assert all(isinstance(value, PyCtxt) for value in raw_value2)
    decrypted2 = np.concatenate([
        ckks.decrypt_list([ciphertext], take=len(expected))[0]
        for ciphertext, expected in zip(raw_value2, expected_segments)
    ])
    np.testing.assert_allclose(decrypted2, small_vector, atol=1e-2)

# ---------------------------------------------------------------------------
# put with string path — auto-delegates to put_from_file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_string_path_default_ckks(client, ckks, ckks_params, storage_ids, tmp_npy_file):
    path, _ = tmp_npy_file
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=path)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_string_path_encrypt_ckks(client, ckks, ckks_params, storage_ids, tmp_npy_file):
    path, _ = tmp_npy_file
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=path, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_string_path_default_liu(client, liu_params, storage_ids, tmp_npy_file):
    path, _ = tmp_npy_file
    backend = liu_builder(client, liu_params)
    result = await backend.put(**storage_ids, data=path)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_string_path_delete_flag(client, ckks, ckks_params, storage_ids, tmp_npy_file):
    path, _ = tmp_npy_file
    backend = ckks_builder(client, ckks, ckks_params)
    first = await backend.put(**storage_ids, data=path)
    assert first.is_ok, first.unwrap_err()
    result = await backend.put(**storage_ids, data=path, delete=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_a_list_with_one_element_ckks(client, ckks, ckks_params, storage_ids):
    backend = StorageBuilder(storage_client=client, scheme=Scheme.CKKS, scheme_params=ckks_params)\
    .with_storage_params(params=StorageParams(num_chunks=2))\
    .build()
    # ckks_builder(client, ckks, ckks_params)
    data = np.array([1])
    result = await backend.put(**storage_ids, data=data, delete=True, segment=True, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    get_result = await backend.get(**storage_ids, segment=True, encrypt=True)
    assert get_result.is_ok, get_result.unwrap_err()
    value = get_result.unwrap()
    # print(value.raw_value)
    result2 = await backend.put(**storage_ids, data=value.raw_value, delete=True, segment=True, encrypt=True)
    assert result2.is_ok, result2.unwrap_err()


# ---------------------------------------------------------------------------
# List[int] / List[float] auto-conversion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_list_int_plaintext(client, liu_params, storage_ids):
    """List[int] is accepted and stored as a float64 plaintext blob."""
    backend = StorageBuilder(storage_client=client).build()
    data = [1, 2, 3, 4, 5]
    result = await backend.put(**storage_ids, data=data)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().dtype == np.float64


@pytest.mark.asyncio
async def test_put_list_float_plaintext(client, liu_params, storage_ids):
    """List[float] is accepted and stored as a float64 plaintext blob."""
    backend = StorageBuilder(storage_client=client, scheme=Scheme.LIU, scheme_params=liu_params).build()
    data = [0.1, 0.2, 0.3, 0.4]
    result = await backend.put(**storage_ids, data=data)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().dtype == np.float64


@pytest.mark.asyncio
async def test_put_list_mixed_int_float_plaintext(client, liu_params, storage_ids):
    """A mixed List[int | float] is cast to float64."""
    backend = StorageBuilder(storage_client=client, scheme=Scheme.LIU, scheme_params=liu_params).build()
    data = [1, 2.5, 3, 4.0]
    result = await backend.put(**storage_ids, data=data)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().dtype == np.float64


@pytest.mark.asyncio
async def test_put_list_empty_returns_err(client, liu_params, storage_ids):
    """An empty list returns Err, not an exception."""
    backend = StorageBuilder(storage_client=client, scheme=Scheme.LIU, scheme_params=liu_params).build()
    result = await backend.put(**storage_ids, data=[])
    assert result.is_err


@pytest.mark.asyncio
async def test_put_get_list_float_round_trip(client, liu_params, storage_ids):
    """Round-trip: List[float] put then get returns the same values."""
    backend = StorageBuilder(storage_client=client, scheme=Scheme.LIU, scheme_params=liu_params).build()
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = await backend.put(**storage_ids, data=data)
    assert result.is_ok, result.unwrap_err()
    get_result = await backend.get(**storage_ids)
    assert get_result.is_ok, get_result.unwrap_err()
    retrieved = get_result.unwrap().raw_value
    np.testing.assert_array_almost_equal(retrieved.flatten(), np.array(data, dtype=np.float64))


@pytest.mark.asyncio
async def test_put_list_int_segment(client, liu_params, storage_ids):
    """List[int] with segment=True splits into plaintext chunks correctly."""
    backend = StorageBuilder(
        storage_client = client,
        scheme         = Scheme.LIU,
        scheme_params  = liu_params,
    ).with_storage_params(StorageParams(num_chunks=2)).build()
    data = [10, 20, 30, 40, 50, 60, 70, 80]
    result = await backend.put(**storage_ids, data=data, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().dtype == np.float64


@pytest.mark.asyncio
async def test_put_list_int_encrypt_ckks(client, ckks, ckks_params, storage_ids):
    """List[int] with encrypt=True on a CKKS backend uses the vector CKKS path."""
    backend = ckks_builder(client, ckks, ckks_params)
    data = [1, 2, 3, 4]
    result = await backend.put(**storage_ids, data=data, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    get_result = await backend.get(**storage_ids, encrypt=True)
    assert get_result.is_ok, get_result.unwrap_err()
    assert len(get_result.unwrap().raw_value) > 0


# ---------------------------------------------------------------------------
# scheme=None (plaintext-only backend) — unit tests (no network)
# ---------------------------------------------------------------------------

def test_builder_no_scheme(client):
    """StorageBuilder with no scheme uses Rory's explicit NONE value."""
    backend = StorageBuilder(storage_client=client).build()
    assert backend.scheme == Scheme.NONE


def test_builder_no_scheme_round_trip(client):
    """as_builder() preserves Scheme.NONE through a round-trip."""
    backend = StorageBuilder(storage_client=client).build()
    cloned = backend.as_builder().build()
    assert cloned.scheme == Scheme.NONE


# ---------------------------------------------------------------------------
# scheme=None — integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_encrypt_no_scheme_returns_err(client, small_matrix, storage_ids):
    """encrypt=True with scheme=None returns Err instead of silently storing plaintext."""
    backend = StorageBuilder(storage_client=client).build()
    result = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert result.is_err
    assert "scheme" in str(result.unwrap_err()).lower()


@pytest.mark.asyncio
async def test_put_no_scheme_plaintext(client, small_matrix, storage_ids):
    """scheme=None stores a plaintext ndarray without errors."""
    backend = StorageBuilder(storage_client=client).build()
    result = await backend.put(**storage_ids, data=small_matrix)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_no_scheme_plaintext(client, small_matrix, storage_ids):
    """scheme=None round-trip: put then get returns the original matrix."""
    backend = StorageBuilder(storage_client=client).build()
    await backend.put(**storage_ids, data=small_matrix)
    get_result = await backend.get(**storage_ids)
    assert get_result.is_ok, get_result.unwrap_err()
    np.testing.assert_array_almost_equal(get_result.unwrap().raw_value, small_matrix)


@pytest.mark.asyncio
async def test_put_get_no_scheme_segment(client, small_matrix, storage_ids):
    """scheme=None with segment=True stores and retrieves chunked plaintext."""
    backend = (
        StorageBuilder(storage_client=client)
        .with_storage_params(StorageParams(num_chunks=2))
        .build()
    )
    result = await backend.put(**storage_ids, data=small_matrix, segment=True)
    assert result.is_ok, result.unwrap_err()
    get_result = await backend.get(**storage_ids, segment=True)
    assert get_result.is_ok, get_result.unwrap_err()
    assert isinstance(get_result.unwrap().raw_value, np.ndarray)


@pytest.mark.asyncio
async def test_put_get_skmeans_liu_udm(client, liu_params, small_matrix, storage_ids):
    """Test put/get of a Liu UDM object."""
    backend = StorageBuilder(storage_client=client, scheme=Scheme.LIU, scheme_params=liu_params).build()
    udm = Utils.calculate_UDM(plaintext_matrix=small_matrix)
    print(udm.shape)
    result = await backend.put(
        bucket_id = storage_ids['bucket_id'],
        ball_id   = storage_ids['ball_id'],
        data      = udm,
        delete    = True,
        encrypt   = False,
        segment   = True,
        tags={}
    )
    assert result.is_ok, f"Error: {result.unwrap_err()}"
    result = await backend.get(
        bucket_id = storage_ids['bucket_id'],
        ball_id   = storage_ids['ball_id'],
        encrypt   = False,
        segment   = True,
    )
    assert result.is_ok, f"Error: {result.unwrap_err()}"

@pytest.mark.asyncio
async def test_put_get_dbskmeans_udm(client, small_matrix, storage_ids):
    """Test put/get of a FDHOPE UDM object."""
    backend = StorageBuilder(storage_client=client).build()
    udm = Utils.calculate_UDM(plaintext_matrix=small_matrix)
    print(udm.shape)
    result = await backend.put(
        bucket_id = storage_ids['bucket_id'],
        ball_id   = storage_ids['ball_id'],
        data      = udm,
        delete    = True,
        encrypt   = False,
        segment   = True,
        tags={}
    )
    assert result.is_ok, f"Error: {result.unwrap_err()}"
    result = await backend.get(
        bucket_id = storage_ids['bucket_id'],
        ball_id   = storage_ids['ball_id'],
        encrypt   = False,
        segment   = True,
    )
    assert result.is_ok, f"Error: {result.unwrap_err()}"
