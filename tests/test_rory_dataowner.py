import pickle
import numpy as np
import pytest
from Pyfhel import PyCtxt

from rory.core.enums.algorithms import Algorithm
from rory.core.enums.schemes import Scheme as RoryScheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import (
    CkksParams as RoryCkksParams,
    LiuParams as RoryLiuParams,
    SchemeParams as RorySchemeParams,
)
from rorycommon import CkksParams, Common, Scheme, SchemeParams, LiuParams


def _liu_params() -> RoryLiuParams:
    return RoryLiuParams(
        security_level=128,
        seed=7,
        use_np_random=True,
    )


def test_rorycommon_reexports_exact_rory_types():
    assert Scheme is RoryScheme
    assert SchemeParams is RorySchemeParams
    assert CkksParams is RoryCkksParams
    assert LiuParams is RoryLiuParams


def test_dataowner_scheme_only_encrypts_with_unified_api():
    plaintext = np.arange(6, dtype=np.float64).reshape(3, 2)
    dataowner = (
        DataOwner.with_scheme(RoryScheme.LIU)
        .with_scheme_params(_liu_params())
        .build()
    )

    result = dataowner.outsourcedData(plaintext)

    assert result.encrypted_matrix.shape[:2] == plaintext.shape
    assert result.encrypted_matrix.ndim == 3
    assert result.UDM.size == 0


def test_dataowner_algorithm_recipe_returns_udm():
    plaintext = np.arange(6, dtype=np.float64).reshape(3, 2)
    dataowner = (
        DataOwner.with_algorithm(Algorithm.SKMEANS)
        .with_scheme(RoryScheme.LIU)
        .with_scheme_params(_liu_params())
        .build()
    )

    result = dataowner.outsourcedData(plaintext)

    assert result.encrypted_matrix.shape[:2] == plaintext.shape
    assert result.UDM.shape == (3, 3, 2)


def test_ckks_dataowner_replaces_removed_pqc_dataowner(dataowner_pqc):
    plaintext = np.arange(6, dtype=np.float64).reshape(3, 2)

    result = dataowner_pqc.outsourcedData(plaintext)

    assert len(result.encrypted_matrix) == len(plaintext)
    assert all(isinstance(value, PyCtxt) for value in result.encrypted_matrix)


def test_dataowner_builder_rejects_invalid_combination():
    with pytest.raises(ValueError, match="Invalid combination"):
        DataOwner.with_algorithm(Algorithm.SKMEANS).with_scheme(
            RoryScheme.CKKS
        ).build()


def test_parallel_liu_uses_one_key_and_distinct_worker_randomness():
    half = np.arange(8, dtype=np.float64).reshape(4, 2)
    plaintext = np.concatenate([half, half], axis=0)
    dataowner = (
        DataOwner.with_scheme(RoryScheme.LIU)
        .with_scheme_params(_liu_params())
        .build()
    )

    chunks, _, _ = Common.segment_and_encrypt_with_dataowner(
        key="parallel-liu",
        plaintext=plaintext,
        dataowner=dataowner,
        num_chunks=2,
    )
    encrypted_parts = [chunk.to_ndarray().unwrap() for chunk in chunks]

    assert not np.array_equal(encrypted_parts[0], encrypted_parts[1])
    decrypted = dataowner.primary_scheme.decrypt_matrix(
        np.concatenate(encrypted_parts, axis=0)
    ).data
    np.testing.assert_allclose(decrypted, plaintext, atol=1e-12)


def test_parallel_ckks_rebuilds_workers_from_shared_keys(dataowner_pqc, ckks):
    plaintext = np.arange(16, dtype=np.float64).reshape(8, 2)

    chunks, _, _ = Common.segment_and_encrypt_with_dataowner(
        key="parallel-ckks",
        plaintext=plaintext,
        dataowner=dataowner_pqc,
        num_chunks=2,
    )
    ciphertexts = []
    for chunk in chunks:
        ciphertexts.extend(
            Common.from_bytes_to_pyctxt_list(ckks, pickle.loads(chunk.data))
        )

    assert len(ciphertexts) == 8


def test_parallel_storage_rejects_algorithm_dataowner():
    dataowner = (
        DataOwner.with_algorithm(Algorithm.SKMEANS)
        .with_scheme(RoryScheme.LIU)
        .with_scheme_params(_liu_params())
        .build()
    )

    with pytest.raises(ValueError, match="Algorithm.NONE"):
        Common.segment_and_encrypt_with_dataowner(
            key="algorithm-owner",
            plaintext=np.ones((4, 2)),
            dataowner=dataowner,
            num_chunks=2,
        )


def test_parallel_storage_caps_chunks_to_axis_zero_length():
    dataowner = (
        DataOwner.with_scheme(RoryScheme.LIU)
        .with_scheme_params(_liu_params())
        .build()
    )

    chunks, _, _ = Common.segment_and_encrypt_with_dataowner(
        key="one-value",
        plaintext=np.array([1.0]),
        dataowner=dataowner,
        num_chunks=4,
    )

    assert len(chunks) == 1
