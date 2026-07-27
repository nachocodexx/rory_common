---
icon: lucide/table-properties
---

# Storage Compatibility

`StorageBackend` keeps storage choices separate from cryptographic choices. Rory's
`DataOwner` owns encryption; the backend owns segmentation, parallel workers, and
storage I/O.

## Storage targets

| Builder configuration | Target | Result source |
|---|---|---|
| `StorageBuilder()` | Filesystem at `/rory/data` | `SourceType.FILE` |
| `StorageBuilder(storage_path="/custom/path")` | Filesystem at the custom root | `SourceType.FILE` |
| `StorageBuilder(storage_client=client)` | Mictlan | `SourceType.CLOUD` |

Filesystem buckets and balls use `<root>/<bucket_id>/<ball_id>/`. IDs must be
single safe path components. Writes atomically replace the ball manifest, and
reads verify payload checksums before deserializing data.

## Flag behavior

Use the same `segment` and `encrypt` flags for the matching `put()` and `get()`.

| `segment` | `encrypt` | `put()` behavior | `get()` behavior |
|---|---|---|---|
| `False` | `False` | Store one plaintext object | Read one plaintext `np.ndarray` |
| `True` | `False` | Split and store plaintext chunks | Merge chunks into an `np.ndarray` |
| `False` | `True` | Encrypt as one logical chunk through `DataOwner` | Read CKKS objects or merged Liu ciphertext |
| `True` | `True` | Split, encrypt chunks in parallel, then upload | Read CKKS objects or merged Liu ciphertext |

`delete=True` is independent of those combinations. It deletes the existing object
before uploading the replacement.

## Accepted inputs

| Input | Constraints | Behavior |
|---|---|---|
| Numeric `np.ndarray` | Non-empty vector or matrix | Plain storage, or DataOwner encryption with CKKS/Liu |
| `List[int]` / `List[float]` | Non-empty | Converted to a 1-D `float64` array |
| `PyCtxt` or a sequence/array of `PyCtxt` | CKKS backend and `encrypt=True` | Recognized as prepared ciphertext and serialized without re-encryption |
| `Chunks` | `encrypt=False` | Uploaded directly |
| File path string | Supported extension | Loaded and delegated to the same `put()` behavior |

## DataOwner configuration

For direct storage encryption, provide either a scheme-only or algorithm-configured
owner. Storage uses only its primary Liu/CKKS scheme and does not execute algorithm
recipes while encrypting chunks:

```python
from rory.core.security.dataowner import DataOwner
from rorycommon import LiuParams, Scheme, StorageBuilder

owner = (
    DataOwner.with_scheme(Scheme.LIU)
    .with_scheme_params(LiuParams(security_level=128))
    .build()
)
backend = StorageBuilder(storage_client=client, dataowner=owner).build()
```

Or let the storage builder construct that owner from Rory's exact parameter types:

```python
from rorycommon import CkksParams, Scheme, StorageBuilder, StorageParams

backend = (
    StorageBuilder(storage_client=client)
    .with_scheme(Scheme.CKKS)
    .with_scheme_params(CkksParams(keys_path="/rory/keys"))
    .with_storage_params(StorageParams(num_chunks=4))
    .build()
)
```

## Scheme behavior

| Scheme | Encrypted input | Encrypted `get()` value | Notes |
|---|---|---|---|
| CKKS | Numeric vector/matrix or prepared `PyCtxt` values | `List[PyCtxt]` | Parallel workers reload the shared key directory from `CkksParams.keys_path` |
| Liu | Numeric vector/matrix | merged `np.ndarray` | Workers share the key but reseed their random generators independently |
| Paillier | — | — | Storage integration is intentionally not implemented |

Algorithm recipes remain Rory concerns. The backend uses the owner's primary
encryption scheme but does not reproduce UDM/DM or FDHOPE recipe steps
chunk-by-chunk, because those artifacts require the complete dataset.

## Algorithm-prepared data

When algorithm artifacts are required, process the entire dataset once with the
algorithm-configured `DataOwner`. Select the artifact required by the remote
algorithm and store that prepared value without a second encryption pass:

```python
prepared = owner.outsourcedData(matrix)

# For example, store an algorithm-prepared UDM as chunked prepared data.
result = await StorageBuilder(storage_client=client).build().put(
    bucket_id="rory",
    ball_id="udm_v1",
    data=prepared.UDM,
    segment=True,
    encrypt=False,
)
```

For an algorithm-produced CKKS `PyCtxt` artifact, use a CKKS-configured backend and
pass `encrypt=True`. The backend detects that encryption is already complete and does
not encrypt it again.
