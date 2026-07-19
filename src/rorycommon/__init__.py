import logging
import time as T
import asyncio
import warnings
from rorycommon.utils import Utils as RoryCommonUtils
import mictlanx.interfaces as InterfaceX
from option import Option,Result,Ok,Err,Some
import numpy as np
import numpy.typing as npt
import pandas as pd
import os
from typing import Self, Tuple, Generator, Dict, AsyncGenerator, Optional, Union, List, Awaitable,Any
from dataclasses import dataclass, field, replace
from rory.core.enums.algorithms import Algorithm
from rory.core.enums.schemes import Scheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import (
    SchemeParams,
    CkksParams,
    LiuParams,
    PaillierParams,
    LiuAndFdhopeParams,
    CkksAndFdhopeParams,
)
from rory.core.security.dataowner_paillier import DataOwner as DataOwnerPHE
from rory.core.security.cryptosystem.liu import Liu
from rory.core.security.cryptosystem.fdhope import Fdhope
from concurrent.futures import ProcessPoolExecutor
from Pyfhel import PyCtxt
import pickle
from rory.core.security.cryptosystem.pqc.ckks import Ckks
import hashlib as H
from rorycommon.models import PutPlaintextResult, PutCiphertextResult,GetResult,TList,SourceType
from rorycommon.storage import FileSystemStorage

from mictlanx.logger.log import Log
from dotenv import load_dotenv

RORY_COMMON_ENV_FILE_PATH = os.environ.get("RORY_COMMON_ENV_FILE_PATH","./.env")
print(f"Loading environment variables from: {RORY_COMMON_ENV_FILE_PATH}")
if os.path.exists(RORY_COMMON_ENV_FILE_PATH):
    load_dotenv(dotenv_path=RORY_COMMON_ENV_FILE_PATH)


def _parse_log_level(value: Union[str, int], env_var_name: str) -> int:
    if isinstance(value, int):
        return value

    normalized_value = str(value).strip()
    if normalized_value.isdigit():
        return int(normalized_value)

    level = logging.getLevelNamesMapping().get(normalized_value.upper())
    if level is None:
        raise ValueError(f"Invalid log level for {env_var_name}: {value}")
    return level





# DEBUG                                 = bool(int(os.environ.get("RORY_COMMON_DEBUG","1")))
RORY_COMMON_LOG_PATH                  = os.environ.get("RORY_COMMON_LOG_PATH","./.rory/log")
RORY_COMMON_LOG_CONSOLE_HANDLER_LEVEL = os.environ.get("RORY_COMMON_LOG_CONSOLE_HANDLER_LEVEL","INFO")
RORY_COMMON_LOG_DISABLED              = bool(int(os.environ.get("RORY_COMMON_LOG_DISABLED","1")))
RORY_COMMON_LOG_ERROR_TO_FILE             = bool(int(os.environ.get("RORY_COMMON_LOG_ERROR_TO_FILE", "0")))
RORY_COMMON_LOG_INTERVAL              = int(os.environ.get("RORY_COMMON_LOG_INTERVAL", "60"))
RORY_COMMON_LOG_FILE_HANDLER_LEVEL    = os.environ.get("RORY_COMMON_LOG_FILE_HANDLER_LEVEL", "DEBUG")
# RORY_COMMON_LOG_OUTPUT_PATH           = os.environ.get("RORY_COMMON_LOG_OUTPUT_PATH", RORY_COMMON_LOG_PATH)
RORY_COMMON_LOG_TO_FILE               = bool(int(os.environ.get("RORY_COMMON_LOG_TO_FILE", "0")))
RORY_COMMON_LOG_WHEN                  = os.environ.get("RORY_COMMON_LOG_WHEN", "m")
RORY_COMMON_LOG_JSON_INDENT           = os.environ.get("RORY_COMMON_LOG_JSON_INDENT", "0")
RORY_COMMON_LOG_RICH                = bool(int(os.environ.get("RORY_COMMON_LOG_RICH", "0")))
RORY_COMMON_LOG_MICTLANX_PROPAGATE  = bool(int(os.environ.get("RORY_COMMON_LOG_MICTLANX_PROPAGATE", "1")))
if RORY_COMMON_LOG_MICTLANX_PROPAGATE:
    os.environ.setdefault("MICTLANX_LOG_DISABLED", str(int(RORY_COMMON_LOG_DISABLED)))
    os.environ.setdefault("MICTLANX_LOG_LEVEL", RORY_COMMON_LOG_CONSOLE_HANDLER_LEVEL)
    os.environ.setdefault("MICTLANX_LOG_RICH", str(int(RORY_COMMON_LOG_RICH)))
    os.environ.setdefault("MICTLANX_LOG_TO_FILE", str(int(RORY_COMMON_LOG_TO_FILE)))
    os.environ.setdefault("MICTLANX_LOG_ERROR_FILE", str(int(RORY_COMMON_LOG_ERROR_TO_FILE)))
    os.environ.setdefault("MICTLANX_LOG_ROTATION_WHEN", RORY_COMMON_LOG_WHEN)
    os.environ.setdefault("MICTLANX_LOG_ROTATION_INTERVAL", str(RORY_COMMON_LOG_INTERVAL))

from mictlanx import AsyncClient
from mictlanx.utils.segmentation import Chunks,Chunk


RORY_COMMON_LOG_CONSOLE_HANDLER_LEVEL = _parse_log_level(
    RORY_COMMON_LOG_CONSOLE_HANDLER_LEVEL,
    "RORY_COMMON_LOG_CONSOLE_HANDLER_LEVEL",
)
RORY_COMMON_LOG_FILE_HANDLER_LEVEL = _parse_log_level(
    RORY_COMMON_LOG_FILE_HANDLER_LEVEL,
    "RORY_COMMON_LOG_FILE_HANDLER_LEVEL",
)

L = Log(
    name                  = __name__,
    console_handler_level = RORY_COMMON_LOG_CONSOLE_HANDLER_LEVEL,
    disabled              = RORY_COMMON_LOG_DISABLED,
    error_log             = RORY_COMMON_LOG_ERROR_TO_FILE,
    interval              = RORY_COMMON_LOG_INTERVAL,
    file_handler_level    = RORY_COMMON_LOG_FILE_HANDLER_LEVEL,
    path                  = RORY_COMMON_LOG_PATH,
    to_file               = RORY_COMMON_LOG_TO_FILE,
    when                  = RORY_COMMON_LOG_WHEN
)

@dataclass
class StorageParams:
    """Tuning parameters applied to every `put()` / `get()` call on the backend. Pass a custom instance to `StorageBuilder.__init__` or `.with_storage_params()`.

    Attributes:
        backoff_factor: Multiplier applied to ``delay`` on each retry.
        num_chunks: Number of segments to split data into when uploading in chunks.
        chunk_index: Starting chunk index for partial retrievals.
        chunk_size: Target size per downloaded chunk (e.g. ``"256kb"``).
        delay: Base delay in seconds between retries.
        force: Pass ``force=True`` to the underlying mictlanx client.
        headers: Extra HTTP headers forwarded on every request.
        http2: Use HTTP/2 for the underlying transport.
        max_attempts: Maximum number of retry attempts before giving up.
        max_parallel_gets: Maximum concurrent chunk downloads.
        timeout: Request timeout in seconds.
    """
    backoff_factor: float = 0.5
    num_chunks: int = field(default=2)
    chunk_index: int = field(default=0)
    chunk_size: str = field(default="256kb")
    delay: int = field(default=1)
    force: bool = field(default=True)
    headers: Dict[str, str] = field(default_factory=dict)
    http2: bool = field(default=False)
    max_attempts: int = field(default=5)
    max_parallel_gets: int = field(default=10)
    timeout: int = field(default=300)

class StorageBuilder:
    """Fluent builder for ``StorageBackend``.

    Example:
        ```python
        backend = (
            StorageBuilder(storage_client=client, scheme=Scheme.CKKS)
            .with_scheme_params(CkksParams(keys_path="/rory/keys"))
            .with_storage_params(StorageParams(num_chunks=4))
            .build()
        )
        ```
    """

    def __init__(
        self,
        storage_client: Optional[AsyncClient] = None,
        dataowner: Optional[DataOwner] = None,
        scheme: Optional[Scheme] = None,
        scheme_params: Optional[SchemeParams] = None,
        params: Optional[StorageParams] = None,
        storage_path: str = "/rory/data",
    ):
        """
        Args:
            storage_client: Async mictlanx client used for remote I/O. When
                ``None``, use filesystem storage.
            dataowner: Caller-built Rory DataOwner. Mutually exclusive with
                ``scheme`` and ``scheme_params``.
            scheme: Rory scheme used to build a scheme-only DataOwner.
            scheme_params: Rory parameters for ``scheme``.
            params: Retrieval/upload tuning. Defaults to ``StorageParams()``.
            storage_path: Filesystem root used when ``storage_client`` is ``None``.
                Defaults to ``/rory/data``.
        """
        self.storage_client = storage_client
        self.dataowner      = dataowner
        self.scheme         = scheme
        self.scheme_params  = scheme_params
        self.params         = params or StorageParams()
        self.storage_path   = storage_path

    def with_dataowner(self, dataowner: DataOwner) -> Self:
        """Use a caller-built DataOwner and clear scheme configuration."""
        self.dataowner = dataowner
        self.scheme = None
        self.scheme_params = None
        return self

    def with_scheme_params(self, scheme_params: SchemeParams) -> Self:
        """Set Rory scheme parameters and switch to builder-owned mode."""
        self.dataowner = None
        self.scheme_params = scheme_params
        return self

    def with_scheme(self, scheme: Scheme) -> Self:
        """Replace the scheme and return ``self`` for chaining.

        Args:
            scheme: New ``Scheme`` value.

        Returns:
            StorageBuilder
        """
        self.dataowner = None
        self.scheme = scheme
        return self

    def with_storage_params(self, params: StorageParams) -> Self:
        """Replace the storage params and return ``self`` for chaining.

        Args:
            params: New ``StorageParams`` instance.

        Returns:
            StorageBuilder
        """
        self.params = params
        return self

    def with_storage_path(self, storage_path: str) -> Self:
        """Select filesystem storage rooted at ``storage_path``."""
        self.storage_client = None
        self.storage_path = storage_path
        return self

    def build(self) -> "StorageBackend":
        """Construct and return the configured ``StorageBackend``.

        Returns:
            A ready-to-use ``StorageBackend``.
        """
        return StorageBackend(
            client        = self.storage_client,
            dataowner     = self.dataowner,
            scheme        = self.scheme,
            scheme_params = self.scheme_params,
            params        = self.params,
            storage_path  = self.storage_path,
        )


class StorageBackend:
    """DataOwner-backed storage façade over ``Common``.

    Construct via ``StorageBuilder`` — do not instantiate directly.

    ``put``, ``put_from_file``, and ``get`` combine Rory ``DataOwner``
    encryption with storage segmentation and I/O. Callers may provide a
    ready ``DataOwner`` or let ``StorageBuilder`` construct a scheme-only one
    from Rory's ``Scheme`` and ``SchemeParams`` types.

    Example:
        ```python
        result = await backend.put(bucket_id="rory", ball_id="v1", data=matrix, encrypt=True)
        if result.is_ok:
            value = result.unwrap()

        result = await backend.get(bucket_id="rory", ball_id="v1", encrypt=True)
        if result.is_ok:
            matrix = result.unwrap().raw_value
        ```
    """

    def __init__(
        self,
        client: Optional[AsyncClient] = None,
        dataowner: Optional[DataOwner] = None,
        scheme: Optional[Scheme] = None,
        scheme_params: Optional[SchemeParams] = None,
        params: Optional[StorageParams] = None,
        storage_path: str = "/rory/data",
    ):
        if dataowner is not None and (scheme is not None or scheme_params is not None):
            raise ValueError("dataowner is mutually exclusive with scheme configuration")
        self.client        = client
        self.dataowner     = dataowner
        self.scheme        = dataowner.scheme if dataowner is not None else (scheme or Scheme.NONE)
        self.scheme_params = dataowner.scheme_params if dataowner is not None else scheme_params
        self.params        = params or StorageParams()
        self.storage_path  = storage_path
        self.filesystem    = FileSystemStorage(storage_path) if client is None else None

    def _require_dataowner(self) -> DataOwner:
        if self.dataowner is None:
            if self.scheme in {None, Scheme.NONE}:
                raise ValueError("A DataOwner or encryption scheme is required")
            builder = DataOwner.with_scheme(self.scheme)
            if self.scheme_params is not None:
                builder = builder.with_scheme_params(self.scheme_params)
            self.dataowner = builder.build()
        return self.dataowner

    def _ckks_context(self) -> Ckks:
        owner = self._require_dataowner().initialize()
        if owner.scheme not in {Scheme.CKKS, Scheme.CKKS_AND_FDHOPE}:
            raise ValueError("The configured DataOwner does not contain CKKS")
        return owner.primary_scheme

    @staticmethod
    def _pyctxt_values(data: Any) -> Optional[List[PyCtxt]]:
        if isinstance(data, PyCtxt):
            return [data]
        if isinstance(data, np.ndarray):
            values = list(data.reshape(-1))
        elif isinstance(data, (list, tuple)):
            values = list(data)
        else:
            return None
        if values and all(isinstance(value, PyCtxt) for value in values):
            return values
        return None

    @property
    def is_filesystem(self) -> bool:
        return self.filesystem is not None

    def _result_path(self, bucket_id: str, ball_id: str) -> Optional[str]:
        if self.filesystem is None:
            return None
        return str(self.filesystem.object_path(bucket_id, ball_id))

    async def _delete_ball(self, bucket_id: str, ball_id: str) -> None:
        if self.filesystem is not None:
            await self.filesystem.delete(bucket_id, ball_id)
            return
        await Common.while_not_delete_ball_id(
            STORAGE_CLIENT=self.client,
            bucket_id=bucket_id,
            key=ball_id,
            timeout=self.params.timeout,
            max_tries=self.params.max_attempts,
        )

    async def _put_ndarray(
        self,
        bucket_id: str,
        ball_id: str,
        matrix: npt.NDArray,
        tags: Dict[str, str],
        *,
        segment: bool,
        encrypt: bool,
    ) -> Result[Any, Exception]:
        if self.filesystem is not None:
            try:
                path = await self.filesystem.put_ndarray(
                    bucket_id,
                    ball_id,
                    matrix,
                    tags,
                    segment=segment,
                    encrypt=encrypt,
                )
                return Ok(path)
            except Exception as error:
                return Err(error)
        return await Common.put_ndarray_no_delete(
            client=self.client,
            bucket_id=bucket_id,
            key=ball_id,
            matrix=matrix,
            tags=tags,
            timeout=self.params.timeout,
            max_retries=self.params.max_attempts,
        )

    async def _put_chunks(
        self,
        bucket_id: str,
        ball_id: str,
        chunks: Chunks,
        tags: Dict[str, str],
        *,
        segment: bool,
        encrypt: bool,
    ) -> Result[Any, Exception]:
        if self.filesystem is not None:
            try:
                path = await self.filesystem.put_chunks(
                    bucket_id,
                    ball_id,
                    chunks,
                    tags,
                    segment=segment,
                    encrypt=encrypt,
                )
                return Ok(path)
            except Exception as error:
                return Err(error)
        return await Common.put_chunks_no_delete(
            client=self.client,
            bucket_id=bucket_id,
            key=ball_id,
            chunks=chunks,
            tags=tags,
            timeout=self.params.timeout,
            max_retries=self.params.max_attempts,
        )

    async def _get_from_filesystem(
        self,
        bucket_id: str,
        ball_id: str,
        *,
        segment: bool,
        encrypt: bool,
    ) -> GetResult:
        if self.filesystem is None:
            raise RuntimeError("Filesystem storage is not configured")
        started = T.monotonic()
        kind, stored_value = await self.filesystem.get_object(
            bucket_id,
            ball_id,
            segment=segment,
            encrypt=encrypt,
        )
        if kind == "ndarray":
            value = stored_value
        else:
            chunks = stored_value
            if encrypt and self.scheme in {Scheme.CKKS, Scheme.CKKS_AND_FDHOPE}:
                ckks = self._ckks_context()
                value = []
                for chunk in chunks.sorted_by():
                    serialized = pickle.loads(chunk.data)
                    value.extend(Common.from_bytes_to_pyctxt_list(ckks=ckks, xs=serialized))
            else:
                arrays = []
                for chunk in chunks.sorted_by():
                    restored = chunk.to_ndarray()
                    if restored.is_none:
                        raise ValueError(f"Chunk {chunk.index} is not an ndarray")
                    arrays.append(restored.unwrap())
                if not arrays:
                    raise ValueError("Stored object contains no chunks")
                value = np.concatenate(arrays, axis=0)
        dtype = str(np.dtype("object")) if isinstance(value, list) else str(value.dtype)
        return GetResult(
            source=SourceType.FILE,
            raw_value=value,
            read_time=T.monotonic() - started,
            dtype=dtype,
        )

    def as_builder(self) -> StorageBuilder:
        """Return a ``StorageBuilder`` pre-populated with this backend's configuration.

        Use this to fork the current backend into a new one that shares the same
        client and params but uses a different scheme or data owner — override
        only what you need via the fluent ``.with_*()`` methods, then call ``.build()``.

        Example:
            ```python
            liu_backend = (
                ckks_backend.as_builder()
                .with_scheme(Scheme.LIU)
                .with_scheme_params(LiuParams())
                .build()
            )
            ```

        Returns:
            StorageBuilder
        """
        if self.dataowner is not None:
            return StorageBuilder(
                storage_client=self.client,
                dataowner=self.dataowner,
                params=self.params,
                storage_path=self.storage_path,
            )
        return StorageBuilder(
            storage_client=self.client,
            scheme=self.scheme,
            scheme_params=self.scheme_params,
            params=self.params,
            storage_path=self.storage_path,
        )

    async def put(
        self,
        bucket_id: str,
        ball_id: str,
        data: Union[
            npt.NDArray,
            PyCtxt,
            List[PyCtxt],
            Tuple[PyCtxt, ...],
            List[int],
            List[float],
            Chunks,
            str,
        ],
        tags: Dict[str, str] = {},
        segment: bool = False,
        encrypt: bool = False,
        delete: bool = False,
    ) -> Result[Union[PutPlaintextResult,PutCiphertextResult], Exception]:
        """Upload data to the configured storage with optional segmentation and encryption.

        Dispatch table:

        | ``data type`` | ``encrypt`` | ``segment`` | ``data.ndim`` | scheme | action |
        |---|---|---|---|---|---|
        | ``str`` (file path) | any | any | — | any | delegates to ``put_from_file`` |
        | ``List[int]`` / ``List[float]`` | any | any | — | any | auto-converted to 1-D ``float64`` ndarray, then follows the ndarray rows below |
        | ``PyCtxt`` / sequence of ``PyCtxt`` | ``True`` | any | — | CKKS | detect prepared ciphertexts, serialize without re-encrypting |
        | ``Chunks`` | ``False`` | — | — | any | ``put_chunks`` directly |
        | ``ndarray`` | ``True`` | ``False`` | any | CKKS / LIU | encrypt through DataOwner as one logical chunk |
        | ``ndarray`` | ``True`` | ``True`` | any | CKKS / LIU | segment, encrypt chunks in parallel through DataOwner |
        | ``ndarray`` | ``False`` | ``True`` | any | any | ``Chunks.from_ndarray`` → ``put_chunks`` |
        | ``ndarray`` | ``False`` | ``False`` | any | any | single blob via ``put_ndarray`` |

        Args:
            bucket_id: Target bucket name.
            ball_id: Key identifying the object within the bucket.
            data: Data to store — a numpy array, a ``List[int]`` or ``List[float]``
                (auto-converted to a 1-D ``float64`` ndarray), a pre-encrypted
                ``List[PyCtxt]``, a pre-built ``Chunks`` object, or a **file path string**.
                When a string is passed the extension is derived from the path suffix
                and the call is forwarded to ``put_from_file``.
            tags: Arbitrary key/value metadata stored alongside the object.
            segment: Split into ``params.num_chunks`` chunks. With encryption enabled,
                each chunk is encrypted in parallel; without it, chunks remain plaintext.
            encrypt: Encrypt numeric input through the configured scheme-only DataOwner.
                An algorithm-configured DataOwner must first process the complete dataset;
                store its selected output artifact as prepared data instead.
            delete: Delete any existing object at ``ball_id`` before uploading.
                Safe to use even if the key does not exist yet.

        Returns:
            ``Ok(PutPlaintextResult)`` or ``Ok(PutCiphertextResult)`` on success, ``Err(Exception)`` on failure.
        """
        try:
            p = self.params
            if delete:
                await self._delete_ball(bucket_id, ball_id)
            # File path shortcut — derive extension and delegate; delete already ran above.
            if isinstance(data, str):
                ext = os.path.splitext(data)[1].lstrip(".")
                return await self.put_from_file(
                    bucket_id, ball_id,
                    path      = data,
                    extension = ext,
                    tags      = tags,
                    segment   = segment,
                    encrypt   = encrypt,
                    delete    = False,
                )
            # List[int] / List[float] — convert to 1-D float64 ndarray before dispatch.
            if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                if len(data) == 0:
                    return Err(ValueError("data list is empty."))
                data = np.array(data, dtype=np.float64)
            t0 = T.monotonic()
            _scheme = self.scheme
            pyctxts = self._pyctxt_values(data)
            if pyctxts is not None:
                if not encrypt:
                    return Err(ValueError("Prepared CKKS ciphertexts require encrypt=True"))
                if _scheme not in {Scheme.CKKS, Scheme.CKKS_AND_FDHOPE}:
                    return Err(ValueError("Prepared CKKS ciphertexts require a CKKS DataOwner"))
                
                t1 = T.monotonic()
                chunks = Common.from_pyctxts_to_chunks(
                    key=ball_id,
                    xs=pyctxts,
                    num_chunks=p.num_chunks if segment else 1,
                ).unwrap()
                segment_time = T.monotonic() - t1
                t2 = T.monotonic()
                r = await self._put_chunks(
                    bucket_id,
                    ball_id,
                    chunks,
                    tags,
                    segment=segment,
                    encrypt=True,
                )

                if r.is_err:
                    return r
                return Ok(PutCiphertextResult(
                    path         = self._result_path(bucket_id, ball_id),
                    extension    = "",
                    bucket_id    = bucket_id,
                    ball_id      = ball_id,
                    tags         = tags,
                    shape        = (len(pyctxts),),
                    dtype        = np.dtype("object"),
                    read_time    = 0.0,
                    segment_time = segment_time,
                    encrypt_time = 0.0,
                    upload_time  = T.monotonic() - t2,
                ))

            # Pre-processed Chunks — put_chunks directly
            if isinstance(data, Chunks) and not encrypt:
                r = await self._put_chunks(
                    bucket_id,
                    ball_id,
                    data,
                    tags,
                    segment=segment,
                    encrypt=False,
                )
                if r.is_err:
                    return r
                return Ok(PutPlaintextResult(
                    path         = self._result_path(bucket_id, ball_id),
                    extension    = "",
                    bucket_id    = bucket_id,
                    ball_id      = ball_id,
                    tags         = tags,
                    shape        = (0,),
                    dtype        = None,
                    read_time    = 0.0,
                    segment_time = 0.0,
                    upload_time  = T.monotonic() - t0,
                ))

            if isinstance(data, np.ndarray) and encrypt:
                owner = self._require_dataowner()
                if owner.algorithm != Algorithm.NONE:
                    return Err(ValueError(
                        "Parallel storage encryption requires a scheme-only DataOwner "
                        "(Algorithm.NONE). Run outsourcedData() on the full dataset and "
                        "store the selected result artifact instead."
                    ))
                if owner.scheme == Scheme.PAILLIER:
                    return Err(NotImplementedError("Paillier storage is not supported"))

                encrypted_chunks, segment_time, encrypt_time = (
                    Common.segment_and_encrypt_with_dataowner(
                        key        = ball_id,
                        plaintext  = data,
                        dataowner  = owner,
                        num_chunks = p.num_chunks if segment else 1,
                    )
                )
                t1 = T.monotonic()
                r = await self._put_chunks(
                    bucket_id,
                    ball_id,
                    encrypted_chunks,
                    tags,
                    segment=segment,
                    encrypt=True,
                )
                if r.is_err:
                    return r
                return Ok(PutCiphertextResult(
                    path         = self._result_path(bucket_id, ball_id),
                    extension    = "",
                    bucket_id    = bucket_id,
                    ball_id      = ball_id,
                    tags         = tags,
                    shape        = data.shape,
                    dtype        = data.dtype,
                    read_time    = 0.0,
                    segment_time = segment_time,
                    encrypt_time = encrypt_time,
                    upload_time  = T.monotonic() - t1,
                ))

            # ndarray + segment=True, encrypt=False → Chunks.from_ndarray → put_chunks
            if segment and not encrypt and isinstance(data, np.ndarray):
                # print("Segmenting plaintext ndarray into chunks for upload...")
                t0 = T.monotonic()
                plain_chunks = Chunks.from_ndarray(ndarray=data, group_id=ball_id,chunk_prefix=Some(ball_id), num_chunks=p.num_chunks).unwrap()
                segment_time = T.monotonic() - t0
                t1 = T.monotonic()
                r = await self._put_chunks(
                    bucket_id,
                    ball_id,
                    plain_chunks,
                    tags,
                    segment=True,
                    encrypt=False,
                )
                if r.is_err:
                    return r
                upload_time = T.monotonic() - t1
                return Ok(PutPlaintextResult(
                    path         = self._result_path(bucket_id, ball_id),
                    extension    = "",
                    bucket_id    = bucket_id,
                    ball_id      = ball_id,
                    tags         = tags,
                    shape        = data.shape,
                    dtype        = data.dtype,
                    read_time    = 0.0,
                    segment_time = segment_time,
                    upload_time  = upload_time,
                ))

            # Default: single blob, no segmentation, no encryption
            t0 = T.monotonic()
            r = await self._put_ndarray(
                bucket_id,
                ball_id,
                data,
                tags,
                segment=False,
                encrypt=False,
            )
            if r.is_err:
                return r
            return Ok(PutPlaintextResult(
                path         = self._result_path(bucket_id, ball_id),
                extension    = "",
                bucket_id    = bucket_id,
                ball_id      = ball_id,
                tags         = tags,
                shape        = data.shape,
                dtype        = data.dtype,
                read_time    = 0.0,
                segment_time = 0.0,
                upload_time  = T.monotonic() - t0,
            ))

        except Exception as e:
            return Err(e)

    async def put_from_file(
        self,
        bucket_id: str,
        ball_id: str,
        path: str,
        extension: str,
        tags: Dict[str, str] = {},
        segment: bool = False,
        encrypt: bool = False,
        delete: bool = False,
    ) -> Result[Union[PutPlaintextResult, PutCiphertextResult], Exception]:
        """Read an array from disk and upload it to the configured storage.

        For encrypted uploads the file is read first so that ``put`` can dispatch
        on ``data.ndim`` (1-D vector vs 2-D matrix).  All combinations delegate to
        ``put``, which applies the ``delete`` pre-step when requested.

        Args:
            bucket_id: Target bucket name.
            ball_id: Key identifying the object within the bucket.
            path: Path to the file on disk (include extension in the path string).
            extension: File format — ``"npy"`` or ``"csv"``.
            tags: Arbitrary key/value metadata stored alongside the object.
            segment: Split into plaintext chunks before uploading (no encryption).
            encrypt: Segment *and* encrypt before uploading.
            delete: Delete any existing object at ``ball_id`` before uploading.

        Returns:
            ``Ok(PutPlaintextResult)`` or ``Ok(PutCiphertextResult)`` on success,
            ``Err(Exception)`` on failure.
        """
        try:
            p = self.params
            # Default: single blob, no encryption — use the dedicated disk→put helper
            # (avoids reading the whole file into memory when it's not needed).
            if not self.is_filesystem and not segment and not encrypt:
                if delete:
                    await self._delete_ball(bucket_id, ball_id)
                return await Common.from_matrix_on_disk_to_cloud_storage(
                    path         = path,
                    extension    = extension,
                    client       = self.client,
                    bucket_id    = bucket_id,
                    ball_id      = ball_id,
                    tags         = tags,
                    timeout      = p.timeout,
                    max_attempts = p.max_attempts,
                )
            # All other combinations: read file then delegate to put.
            # put handles ndim dispatch (vector vs matrix) and the delete flag.
            t_read = T.monotonic()
            res = await Common.read_numpy_from(path=path, extension=extension)
            if res.is_err:
                return res
            read_time = T.monotonic() - t_read
            result = await self.put(bucket_id, ball_id, res.unwrap(), tags, segment=segment, encrypt=encrypt, delete=delete)
            if result.is_ok:
                result.unwrap().read_time = read_time
            return result
        except Exception as e:
            return Err(e)

    async def get(
        self,
        bucket_id: str,
        ball_id: str,
        segment: bool = False,
        encrypt: bool = False,
    ) -> Result[GetResult[TList], Exception]:
        """Download data from the configured storage.

        Mirror the ``segment`` and ``encrypt`` flags used during the corresponding
        ``put`` call so the correct retrieval method is selected.

        | ``encrypt`` | ``segment`` | scheme | ``GetResult.raw_value`` type |
        |---|---|---|---|
        | ``True`` | — | CKKS | ``List[PyCtxt]`` |
        | ``True`` | — | LIU | ``np.ndarray`` (merged ciphertext chunks) |
        | ``False`` | ``True`` | any | ``np.ndarray`` (merged chunks) |
        | ``False`` | ``False`` | any | ``np.ndarray`` (single blob) |

        Args:
            bucket_id: Target bucket name.
            ball_id: Key identifying the object within the bucket.
            segment: ``True`` when the data was stored as segmented chunks.
            encrypt: ``True`` when DataOwner ciphertext was stored.

        Returns:
            ``Ok(GetResult[T])`` on success, ``Err(Exception)`` on failure.
        """
        try:
            if self.filesystem is not None:
                return Ok(await self._get_from_filesystem(
                    bucket_id,
                    ball_id,
                    segment=segment,
                    encrypt=encrypt,
                ))
            p = self.params
            _scheme = self.scheme
            if encrypt and _scheme == Scheme.PAILLIER:
                return Err(NotImplementedError("Paillier storage is not supported"))
            # CKKS encrypted chunks → get_pyctxt
            t_read = T.monotonic()
            if encrypt and _scheme in {Scheme.CKKS, Scheme.CKKS_AND_FDHOPE}:
                pyctxts = await Common.get_pyctxt(
                    client            = self.client,
                    bucket_id         = bucket_id,
                    key               = ball_id,
                    ckks              = self._ckks_context(),
                    max_retries       = p.max_attempts,
                    delay             = p.delay,
                    backoff_factor    = p.backoff_factor,
                    max_paralell_gets = p.max_parallel_gets,
                    force             = p.force,
                    timeout           = p.timeout,
                    chunk_size        = p.chunk_size,
                    headers           = p.headers,
                    http2             = p.http2,
                    chunk_index       = p.chunk_index,
                )
                read_time = T.monotonic() - t_read
                return Ok(
                    GetResult(
                        source    = SourceType.CLOUD,
                        raw_value = pyctxts,
                        read_time = read_time,
                        dtype     = str(np.dtype("object"))
                    )
                )

            if encrypt and _scheme != Scheme.LIU:
                return Err(ValueError(
                    f"Storage retrieval does not support encrypted scheme {_scheme.value}"
                ))

            # Liu/plain segmented retrieval → get_and_merge
            if encrypt or segment:
                # print("Using get_and_merge for segmented/encrypted retrieval")
                merged = await Common.get_and_merge(
                    client            = self.client,
                    bucket_id         = bucket_id,
                    key               = ball_id,
                    max_retries       = p.max_attempts,
                    delay             = p.delay,
                    backoff_factor    = p.backoff_factor,
                    max_paralell_gets = p.max_parallel_gets,
                    force             = p.force,
                    timeout           = p.timeout,
                    chunk_size        = p.chunk_size,
                    headers           = p.headers,
                    http2             = p.http2,
                    chunk_index       = p.chunk_index,
                )
                read_time = T.monotonic() - t_read
                return Ok(
                    GetResult(
                        source    = SourceType.CLOUD,
                        raw_value = merged,
                        read_time = read_time,
                        dtype     = str(merged.dtype)
                    )
                )

            # Default: single blob → get_matrix_or_error
            matrix = await Common.get_matrix_or_error(
                
                client            = self.client,
                key               = ball_id,
                bucket_id         = bucket_id,
                max_retries       = p.max_attempts,
                delay             = p.delay,
                backoff_factor    = p.backoff_factor,
                max_paralell_gets = p.max_parallel_gets,
                force             = p.force,
                timeout           = p.timeout,
                chunk_size        = p.chunk_size,
                headers           = p.headers,
                http2             = p.http2,
                chunk_index       = p.chunk_index,
            )
            read_time = T.monotonic() - t_read
            return Ok(
                GetResult(
                    source    = SourceType.CLOUD,
                    raw_value = matrix,
                    read_time = read_time,
                    dtype= str(matrix.dtype)
                )
            )

        except Exception as e:
            return Err(e)


class Common:
    """Low-level static helpers called by ``StorageBackend``.

    Not part of the public API — use ``StorageBackend`` instead.

    Methods are grouped by concern:

    - **Plaintext** — ``from_matrix_*``, ``from_cloud_storage_to_matrix``
    - **CKKS** — ``from_matrix_*_ckks``, ``from_vector_*_ckks``, ``segment_and_encrypt_ckks_*``
    - **Liu** — ``segment_and_encrypt_liu*``
    - **Retrieval** — ``get_matrix_or_error``, ``get_and_merge``, ``get_pyctxt``, ``get_by_chunk_index``
    - **Serialization** — ``from_pyctxt_*``, ``serialize_matrix_with_pickle``
    - **Low-level I/O** — ``put_ndarray``, ``put_chunks``, ``delete_and_put_*``
    """
    
    ckks             = None
    dataowner        = None
    liu_dataowner    = None
    fdhope_dataowner = None

    @staticmethod
    def _encrypt_dataowner_chunk(dataowner: DataOwner, key: str, chunk: Chunk) -> Chunk:
        plaintext = chunk.to_ndarray().unwrap()
        encrypted = dataowner.outsourcedData(plaintext).encrypted_matrix
        if dataowner.scheme == Scheme.CKKS:
            ciphertexts = list(np.asarray(encrypted, dtype=object).reshape(-1))
            return Chunk(
                group_id=key,
                index=chunk.index,
                data=Common.from_pyctxt_list_to_bytes(ciphertexts),
                chunk_id=Some(f"{key}_{chunk.index}"),
                metadata={"cipher_format": "ckks"},
            )
        if dataowner.scheme == Scheme.LIU:
            return Chunk.from_ndarray(
                group_id=key,
                index=chunk.index,
                ndarray=encrypted,
                chunk_id=Some(f"{key}_{chunk.index}"),
            )
        raise ValueError(f"Unsupported storage encryption scheme: {dataowner.scheme.value}")

    @staticmethod
    def init_liu_dataowner_worker(serialized_dataowner: bytes):
        """Restore one Liu key context and assign fresh worker randomness."""
        Common.dataowner = pickle.loads(serialized_dataowner).reseed()

    @staticmethod
    def init_ckks_dataowner_worker(params: CkksParams):
        """Load the shared CKKS key context once in a worker process."""
        Common.dataowner = (
            DataOwner.with_scheme(Scheme.CKKS)
            .with_scheme_params(params)
            .build()
            .initialize()
        )

    @staticmethod
    def encrypt_chunk_with_initialized_dataowner(key: str, chunk: Chunk) -> Chunk:
        if Common.dataowner is None:
            raise RuntimeError("DataOwner worker context is not initialized")
        return Common._encrypt_dataowner_chunk(Common.dataowner, key, chunk)

    @staticmethod
    def segment_and_encrypt_with_dataowner(
        key: str,
        plaintext: npt.NDArray,
        dataowner: DataOwner,
        num_chunks: int,
    ) -> Tuple[Chunks, float, float]:
        """Segment plaintext and encrypt every segment through Rory DataOwner."""
        if dataowner.algorithm != Algorithm.NONE:
            raise ValueError("Segment encryption requires Algorithm.NONE")
        if dataowner.scheme not in {Scheme.CKKS, Scheme.LIU}:
            raise ValueError(f"Unsupported storage encryption scheme: {dataowner.scheme.value}")
        if num_chunks < 1:
            raise ValueError("num_chunks must be at least 1")
        if plaintext.ndim == 0 or plaintext.size == 0:
            raise ValueError("plaintext must be a non-empty vector or matrix")

        # Splitting happens on axis 0. Avoid empty segments for vectors and
        # matrices with fewer rows than the configured worker count.
        num_chunks = min(num_chunks, plaintext.shape[0])

        t0 = T.monotonic()
        chunks = Chunks.from_ndarray(
            ndarray=plaintext,
            group_id=key,
            chunk_prefix=Some(key),
            num_chunks=num_chunks,
        ).unwrap()
        segment_time = T.monotonic() - t0
        t1 = T.monotonic()

        if num_chunks == 1:
            encrypted = [
                Common._encrypt_dataowner_chunk(dataowner, key, chunk)
                for chunk in chunks.iter()
            ]
            return Chunks(chs=encrypted, n=plaintext.size), segment_time, T.monotonic() - t1

        if dataowner.scheme == Scheme.LIU:
            dataowner.initialize()
            initializer = Common.init_liu_dataowner_worker
            initargs = (pickle.dumps(dataowner, protocol=pickle.HIGHEST_PROTOCOL),)
        else:
            params = dataowner.scheme_params
            if not isinstance(params, CkksParams):
                raise ValueError("CKKS parallel encryption requires CkksParams")
            worker_params = params
            if params.keys_path is None:
                if not params.save:
                    raise ValueError(
                        "Parallel CKKS encryption requires keys_path or save=True"
                    )
                dataowner.initialize()
                worker_params = replace(
                    params,
                    keys_path=params.output_path,
                    save=False,
                )
            initializer = Common.init_ckks_dataowner_worker
            initargs = (worker_params,)

        with ProcessPoolExecutor(
            max_workers=num_chunks,
            initializer=initializer,
            initargs=initargs,
        ) as executor:
            futures = [
                executor.submit(
                    Common.encrypt_chunk_with_initialized_dataowner,
                    key=key,
                    chunk=chunk,
                )
                for chunk in chunks.iter()
            ]
            encrypted = Common.to_chunks_generator(futures)
        return Chunks(chs=encrypted, n=plaintext.size), segment_time, T.monotonic() - t1


    # Plain text
    @staticmethod 
    async def from_matrix_on_disk_to_cloud_storage(
        path:str,
        extension:str,
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        tags:Dict[str,str]={},
        timeout:int=120,
        max_attempts:int = 5,
    )-> Result[PutPlaintextResult, Exception]:
        """Read a matrix from disk and store it as a single plaintext blob.

        Args:
            path: Path to the file on disk.
            extension: File format — ``"npy"`` or ``"csv"``.
            client: mictlanx async client.
            bucket_id: Target bucket.
            ball_id: Object key.
            tags: Extra metadata tags.
            timeout: Request timeout in seconds.
            max_attempts: Maximum upload retries.

        Returns:
            ``Ok(PutPlaintextResult)`` on success, ``Err(Exception)`` on failure.
        """
        try:
            read_start_time = T.monotonic()
            res = await Common.read_numpy_from(path=path, extension=extension)
            if res.is_err:
                print(f"Failed to read matrix from disk: {res.unwrap_err()}")
                return res
            plaintext_matrix = res.unwrap()
            read_time        = T.monotonic() - read_start_time
            shape = plaintext_matrix.shape
            dtype = plaintext_matrix.dtype
            t0 = T.monotonic()
            result = await Common.put_ndarray(
                client      = client,
                bucket_id   = bucket_id,
                key         = ball_id,
                matrix      = plaintext_matrix,
                tags        = tags,
                timeout     = timeout,
                max_retries = max_attempts,
            )

            if result.is_err:
                print(f"Failed to put ndarray from disk to cloud storage: {result.unwrap_err()}")
                return result
            upload_time = T.monotonic() - t0
            response = PutPlaintextResult(
                ball_id      = ball_id,
                bucket_id    = bucket_id,
                extension    = extension,
                path         = path,
                tags         = tags,
                shape        = shape,
                dtype        = dtype,
                read_time    = read_time,
                segment_time = 0.0,
                upload_time  = upload_time
            )
            return  Ok(response)
        except Exception as e:
            return Err(e)
    
    @staticmethod
    async def from_matrix_to_cloud_storage(
        plaintext_matrix:npt.NDArray,
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        tags:Dict[str,str]={},
        timeout:int=120,
        max_attempts:int = 5,
    )->Result[PutPlaintextResult,Exception]:
        """Store an in-memory matrix as a single plaintext blob.

        Shape and dtype are saved as metadata tags so ``get_matrix_or_error``
        can reconstruct the original array without extra bookkeeping.

        Args:
            plaintext_matrix: NumPy array to store.
            client: mictlanx async client.
            bucket_id: Target bucket.
            ball_id: Object key.
            tags: Extra metadata tags.
            timeout: Request timeout in seconds.
            max_attempts: Maximum upload retries.

        Returns:
            ``Ok(PutPlaintextResult)`` on success, ``Err(Exception)`` on failure.
        """
        try:
            t0 = T.monotonic()
            result = await Common.put_ndarray(
                client      = client,
                bucket_id   = bucket_id,
                key         = ball_id,
                matrix      = plaintext_matrix,
                tags        = tags,
                timeout     = timeout,
                max_retries = max_attempts,
            )
            if result.is_err:
                print(f"Failed to put ndarray to cloud storage: {result.unwrap_err()}")
                return Err(result.unwrap_err())
            upload_time = T.monotonic() - t0
            response = PutPlaintextResult(
                ball_id      = ball_id,
                bucket_id    = bucket_id,
                extension    = "",
                path         = "",
                tags         = tags,
                shape        = plaintext_matrix.shape,
                dtype        = plaintext_matrix.dtype,
                read_time    = 0.0,
                segment_time = 0.0,
                upload_time  = upload_time
            )
            return  Ok(response)
        except Exception as e:
            return Err(e)


    @staticmethod
    async def from_cloud_storage_to_matrix(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        chunk_index:int = 0,
        backoff_factor:int = 2,
        delay:int = 1,
        max_attempts:int = 5,
        timeout:int = 120,
        force:bool = True,
        max_parallel_gets:int = 10,
        headers:Dict[str,str] = {},
        chunk_size:str = "256kb",
        http2:bool = False,
    )->Result[GetResult[npt.NDArray],Exception]:
        """Download a single plaintext blob and return it as a numpy array.

        Counterpart to ``from_matrix_to_cloud_storage``.

        Args:
            client: mictlanx async client.
            bucket_id: Source bucket.
            ball_id: Object key.
            chunk_index: Starting chunk index (0 for full objects).
            backoff_factor: Retry backoff multiplier.
            delay: Base retry delay in seconds.
            max_attempts: Maximum download retries.
            timeout: Request timeout in seconds.
            force: Pass ``force=True`` to the underlying client.
            max_parallel_gets: Maximum concurrent chunk downloads.
            headers: Extra HTTP headers.
            chunk_size: Target size per received chunk.
            http2: Use HTTP/2.

        Returns:
            ``Ok(GetResult[np.ndarray])`` on success, ``Err(Exception)`` on failure.
        """
        try:
            t0 = T.monotonic()
            res = await Common.get_matrix_or_error(
                client            = client,
                bucket_id         = bucket_id,
                key               = ball_id,
                backoff_factor    = backoff_factor,
                chunk_index       = chunk_index,
                chunk_size        = chunk_size,
                timeout           = timeout,
                delay             = delay,
                force             = force,
                headers           = headers,
                http2             = http2,
                max_paralell_gets = max_parallel_gets,
                max_retries       = max_attempts
            )
            response = GetResult(
                source= SourceType.CLOUD,
                raw_value= res,
                read_time= T.monotonic() - t0
            )
            return Ok(response)
        except Exception as e:
            return Err(e)
        
    # CKKS
    @staticmethod
    async def from_matrix_on_disk_to_cloud_storage_ckks(
        path:str,
        extension:str,
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        keys_path:str,
        ctx_filename:str,
        relinkey_filename:str,
        rotatekey_filename:str,
        secretkey_filename:str,
        decimals:int,
        num_chunks:int,
        pubkey_filename:str,
        tags:Dict[str,str]={},
        timeout:int=120,
        max_attempts:int = 5,
    )-> Result[PutCiphertextResult, Exception]:
        """Read a matrix from disk, CKKS-encrypt it, and upload the ciphertext chunks.

        Uses an initialized ``ProcessPoolExecutor`` so each worker loads CKKS
        keys from disk independently — the large PyFHEL context is never pickled.

        Args:
            path: Path to the matrix file on disk.
            extension: File format — ``"npy"`` or ``"csv"``.
            client: mictlanx async client.
            bucket_id: Target bucket.
            ball_id: Object key.
            keys_path: Directory containing the CKKS key files.
            ctx_filename: CKKS context filename.
            relinkey_filename: Relinearization key filename.
            rotatekey_filename: Rotation key filename.
            secretkey_filename: Secret key filename.
            decimals: Fixed-point precision for CKKS encoding.
            num_chunks: Number of ciphertext chunks to produce.
            pubkey_filename: Public key filename.
            tags: Extra metadata tags.
            timeout: Request timeout in seconds.
            max_attempts: Maximum upload retries.

        Returns:
            Result[PutCiphertextResult, Exception]
        """
        try:

            t0 = T.monotonic()
            res = await Common.read_numpy_from(path=path, extension=extension)
            if res.is_err:
                return Err(res.unwrap_err())
            plaintext_matrix = res.unwrap()
            read_time = T.monotonic() - t0
            (result, segment_time, encrypt_time, upload_time) = await Common.segement_and_encrypt_ckks_with_initialized_executor_put_chunks(
                ball_id            = ball_id,
                bucket_id          = bucket_id,
                client             = client,
                ctx_filename       = ctx_filename,
                key                = ball_id,
                max_retries        = max_attempts,
                n                  = plaintext_matrix.shape[0]*plaintext_matrix.shape[1],
                num_chunks         = num_chunks,
                keys_path          = keys_path,
                pubkey_filename    = pubkey_filename,
                plaintext_matrix   = plaintext_matrix,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename,
                secretkey_filename = secretkey_filename,
                tags               = tags,
                timeout            = timeout,
                decimals           = decimals,
            )
            if result.is_err:
                return Err(result.unwrap_err())
            response = PutCiphertextResult(
                ball_id      = ball_id,
                bucket_id    = bucket_id,
                extension    = extension,
                path         = path,
                tags         = tags,
                shape        = plaintext_matrix.shape,
                dtype        = plaintext_matrix.dtype,
                read_time    = read_time,
                segment_time = segment_time,
                encrypt_time = encrypt_time,
                upload_time  = upload_time,
            )
            return Ok(response)
        except Exception as e:
            return Err(e)

    @staticmethod
    async def from_vector_on_disk_to_cloud_storage_ckks(
        path:str,
        extension:str,
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        keys_path:str,
        ctx_filename:str,
        relinkey_filename:str,
        rotatekey_filename:str,
        secretkey_filename:str,
        decimals:int,
        num_chunks:int,
        pubkey_filename:str,
        tags:Dict[str,str]={},
        timeout:int=120,
        max_attempts:int = 5,
        _round:bool = False
    )-> Result[PutCiphertextResult, Exception]:
        """Read a 1-D vector from disk, CKKS-encrypt it, and upload the ciphertext chunks.

        Args:
            path: Path to the vector file on disk.
            extension: File format — ``"npy"`` or ``"csv"``.
            client: mictlanx async client.
            bucket_id: Target bucket.
            ball_id: Object key.
            keys_path: Directory containing the CKKS key files.
            ctx_filename: CKKS context filename.
            relinkey_filename: Relinearization key filename.
            rotatekey_filename: Rotation key filename.
            secretkey_filename: Secret key filename.
            decimals: Fixed-point precision for CKKS encoding.
            num_chunks: Number of ciphertext chunks to produce.
            pubkey_filename: Public key filename.
            tags: Extra metadata tags.
            timeout: Request timeout in seconds.
            max_attempts: Maximum upload retries.
            _round: Round values after CKKS decoding.

        Returns:
            ``Ok(PutCiphertextResult)`` on success, ``Err(Exception)`` on failure.
        """
        try:
            t0 = T.monotonic()
            res = await Common.read_numpy_from(path=path, extension=extension)
            if res.is_err:
                return Err(res.unwrap_err())
            plaintext_matrix = res.unwrap()
            read_time = T.monotonic() - t0
            (result,segment_time,encrypt_time,upload_time) = await Common.segment_encrypt_with_vector_ckks_and_put_chunks_with_initialized_executor(
                client             = client,
                bucket_id          = bucket_id,
                key                = ball_id,
                vector             = plaintext_matrix,
                _round             = _round,
                decimals           = decimals,
                path               = keys_path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename,
                max_attempts       = max_attempts,
                max_workers        = num_chunks,
                tags               = tags,
                timeout            = timeout,
            )
            if result.is_err:
                return Err(result.unwrap_err())
            response = PutCiphertextResult(
                ball_id      = ball_id,
                bucket_id    = bucket_id,
                extension    = extension,
                path         = path,
                tags         = tags,
                shape        = plaintext_matrix.shape,
                dtype        = plaintext_matrix.dtype,
                read_time    = read_time,
                segment_time = segment_time,
                encrypt_time = encrypt_time,
                upload_time=   upload_time,
            )

            return Ok(response) 
        except Exception as e:
            return Err(e)

    @staticmethod
    async def from_matrix_to_cloud_storage_ckks(
        plaintext_matrix:npt.NDArray,
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        keys_path:str,
        ctx_filename:str,
        relinkey_filename:str,
        rotatekey_filename:str,
        secretkey_filename:str,
        decimals:int,
        num_chunks:int,
        pubkey_filename:str,
        tags:Dict[str,str]={},
        timeout:int=120,
        max_attempts:int = 5,
        _round:bool = False
    )->Result[PutCiphertextResult, Exception]:
        """CKKS-encrypt an in-memory matrix and upload the ciphertext chunks.

        Internally calls ``segement_and_encrypt_ckks_with_initialized_executor_put_chunks``
        which creates a ``ProcessPoolExecutor`` where each worker loads the CKKS keys
        from ``keys_path`` — the large PyFHEL context is never pickled across processes.

        Args:
            plaintext_matrix: 2-D numpy array to encrypt and store.
            client: mictlanx async client.
            bucket_id: Target bucket.
            ball_id: Object key.
            keys_path: Directory containing the CKKS key files.
            ctx_filename: CKKS context filename.
            relinkey_filename: Relinearization key filename.
            rotatekey_filename: Rotation key filename.
            secretkey_filename: Secret key filename.
            decimals: Fixed-point precision for CKKS encoding.
            num_chunks: Number of ciphertext chunks to produce.
            pubkey_filename: Public key filename.
            tags: Extra metadata tags.
            timeout: Request timeout in seconds.
            max_attempts: Maximum upload retries.
            _round: Round values after CKKS decoding.

        Returns:
            Result[PutCiphertextResult, Exception]
        """
        try:
            (result, segment_time, encrypt_time, upload_time) = await Common.segement_and_encrypt_ckks_with_initialized_executor_put_chunks(
                client             = client,
                ball_id            = ball_id,
                bucket_id          = bucket_id,
                ctx_filename       = ctx_filename,
                key                = ball_id,
                max_retries        = max_attempts,
                n                  = plaintext_matrix.shape[0]*plaintext_matrix.shape[1],
                num_chunks         = num_chunks,
                keys_path          = keys_path,
                pubkey_filename    = pubkey_filename,
                plaintext_matrix   = plaintext_matrix,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename,
                secretkey_filename = secretkey_filename,
                tags               = tags,
                timeout            = timeout,
                decimals           = decimals,
                _round             = _round
            )
            if result.is_err:
                return Err(result.unwrap_err())
            response = PutCiphertextResult(
                ball_id      = ball_id,
                bucket_id    = bucket_id,
                extension    = "",
                path         = "",
                tags         = tags,
                shape        = plaintext_matrix.shape,
                dtype        = plaintext_matrix.dtype,
                read_time    = 0.0,
                segment_time = segment_time,
                encrypt_time = encrypt_time,
                upload_time  = upload_time
            )
            return Ok(response)
        except Exception as e:
            return Err(e)

    
    @staticmethod
    async def from_vector_to_cloud_storage_ckks(
        vector:npt.NDArray,
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        keys_path:str,
        ctx_filename:str,
        relinkey_filename:str,
        rotatekey_filename:str,
        secretkey_filename:str,
        decimals:int,
        num_chunks:int,
        pubkey_filename:str,
        tags:Dict[str,str]={},
        timeout:int=120,
        max_attempts:int = 5,
        _round:bool = False
    )->Result[PutCiphertextResult, Exception]:
        """CKKS-encrypt an in-memory 1-D vector and upload the ciphertext chunks.

        Args:
            vector: 1-D numpy array to encrypt and store.
            client: mictlanx async client.
            bucket_id: Target bucket.
            ball_id: Object key.
            keys_path: Directory containing the CKKS key files.
            ctx_filename: CKKS context filename.
            relinkey_filename: Relinearization key filename.
            rotatekey_filename: Rotation key filename.
            secretkey_filename: Secret key filename.
            decimals: Fixed-point precision for CKKS encoding.
            num_chunks: Number of ciphertext chunks to produce.
            pubkey_filename: Public key filename.
            tags: Extra metadata tags.
            timeout: Request timeout in seconds.
            max_attempts: Maximum upload retries.
            _round: Round values after CKKS decoding.

        Returns:
            ``Ok(PutCiphertextResult)`` on success, ``Err(Exception)`` on failure.
        """
        try:
            (result, segment_time, encrypt_time, upload_time) = await Common.segment_encrypt_with_vector_ckks_and_put_chunks_with_initialized_executor(
                client             = client,
                bucket_id          = bucket_id,
                key                = ball_id,
                vector             = vector,
                _round             = _round,
                decimals           = decimals,
                path               = keys_path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename,
                max_attempts       = max_attempts,
                max_workers        = num_chunks,
                tags               = tags,
                timeout            = timeout,
            )
            if result.is_err:
                return Err(result.unwrap_err())
            response = PutCiphertextResult(
                ball_id      = ball_id,
                bucket_id    = bucket_id,
                extension    = "",
                path         = "",
                tags         = tags,
                shape        = vector.shape,
                dtype        = vector.dtype,
                read_time    = 0.0,
                segment_time = segment_time,
                encrypt_time = encrypt_time,
                upload_time  = upload_time,
            )
            return Ok(response)
            # return result
        except Exception as e:
            return Err(e)

    @staticmethod
    def init_ckks_worker_context(
        path: str,
        ctx_filename: str,
        pubkey_filename: str,
        secretkey_filename: str,
        relinkey_filename: str = "",
        rotatekey_filename: str = "",
        _round: bool = False,
        decimals: int = 2
    ):
        """Runs once per worker process to load the context into RAM."""
        try:
            L.debug({
                "message": "Initializing CKKS context in worker process",
                "path": path,
                "ctx_filename": ctx_filename,
                "pubkey_filename": pubkey_filename,
                "secretkey_filename": secretkey_filename,
                "relinkey_filename": relinkey_filename,
                "rotatekey_filename": rotatekey_filename,
            })
            Common.ckks = Ckks.from_pyfhel(
                _round             = _round,
                decimals           = decimals,
                path               = path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename 
            ) 
        except Exception as e:
            print(f"Failed to initialize CKKS context: {e}")
            raise e

    @staticmethod
    async def encrypt_and_put_chunk(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        index:int,
        dataowner: DataOwner,
        ndarray:npt.NDArray,
        full_shape:Tuple[int,int],
        num_chunks:int,
        max_backoff:int= 5, 
        max_attempts:int = 10,
        timeout:int=120
    ):
        """
        Encrypts a chunk of data using the provided data owner and puts it into the storage system.

        Arguments:
            client (AsyncClient): The asynchronous client to interact with the storage system.
            bucket_id (str): The ID of the bucket where the chunk will be stored.
            ball_id (str): The ID of the ball to which the chunk belongs.
            index (int): The index of the chunk within the ball.
            dataowner (DataOwner): The data owner responsible for encrypting the chunk.
            ndarray (npt.NDArray): The NumPy array representing the chunk data.
            full_shape (Tuple[int, int]): The full shape of the original data.
            num_chunks (int): The total number of chunks.
            max_backoff (int, optional): The maximum backoff time for retries. Defaults to 5.
            max_attempts (int, optional): The maximum number of attempts for retries. Defaults to 10.
            timeout (int, optional): The timeout for the operation in seconds. Defaults to 120.

        """ 
        res = dataowner.outsourcedData(ndarray).encrypted_matrix
        m = res.shape[2]
        new_full_shape = (full_shape[0],full_shape[1], m)
        new_c = Chunk.from_ndarray(
            group_id=ball_id,
            index=index,
            ndarray=res,
            metadata={
                "full_shape":str(new_full_shape),
                "dtype":str(res.dtype),
                "shape":str(res.shape),
                "num_chunks":str(num_chunks)
            }, 
            chunk_id=Some(f"{ball_id}_{index}")
        )
        res_dp_chunk = await Common.delete_and_put_chunk(
            client      = client,
            bucket_id   = bucket_id,
            ball_id     = ball_id,
            chunk       = new_c,
            tags        = {},
            max_backoff = max_backoff,
            max_tries   = max_attempts,
            timeout     = timeout
        )
        if res_dp_chunk.is_err:
            return res_dp_chunk
        return res_dp_chunk
                # print("FAILED TO PUT", new_c.chunk_id)
    @staticmethod
    async def encrypt_ckks_and_put_chunk(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        index:int,
        dataowner: DataOwner,
        ndarray:npt.NDArray,
        full_shape:Tuple[int,int],
        num_chunks:int,
        max_backoff:int= 5, 
        max_attempts:int = 10,
        timeout:int=120
    ):
            encyrpted_chunk = dataowner.outsourcedData(ndarray).encrypted_matrix
            data = Common.from_pyctxt_list_to_bytes(xs=encyrpted_chunk)
            new_c= Chunk(
                group_id = ball_id, 
                index = index, 
                data = data, 
                chunk_id = Some("{}_{}".format(ball_id,index)),
                metadata= {
                    "full_shape":str(full_shape),
                    "num_chunks":str(num_chunks),
                }
            )
            res_dp_chunk = await Common.delete_and_put_chunk(
                client=client, 
                bucket_id=bucket_id,
                ball_id=ball_id,
                chunk=new_c,
                tags ={},
                max_backoff=max_backoff, 
                max_tries=max_attempts,
                timeout=timeout
            )
            if res_dp_chunk.is_err:
                return res_dp_chunk
            return res_dp_chunk

    @staticmethod
    async def encrypt_paillier_and_put_chunk(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        index:int,
        dataowner: DataOwnerPHE,
        ndarray:npt.NDArray,
        full_shape:Tuple[int,int],
        num_chunks:int,
        max_backoff:int= 5, 
        max_attempts:int = 10,
        timeout:int=120
    ):      
            encrypted_chunk = dataowner.paillier_encrypt_matrix_chunk(ndarray)
            # data = Common.from_pyctxt_list_to_bytes(xs=encyrpted_chunk)
            data = pickle.dumps(encrypted_chunk)
            new_c= Chunk(
                group_id = ball_id, 
                index = index, 
                data = data, 
                chunk_id = Some("{}_{}".format(ball_id,index)),
                metadata= {
                    "full_shape":str(full_shape),
                    "num_chunks":str(num_chunks),
                    "shape":str(encrypted_chunk.shape),
                    "dtype":str(encrypted_chunk.dtype)
                }
            )
            res_dp_chunk = await Common.delete_and_put_chunk(
                client=client, 
                bucket_id=bucket_id,
                ball_id=ball_id,
                chunk=new_c,
                tags ={},
                max_backoff=max_backoff, 
                max_tries=max_attempts,
                timeout=timeout
            )
            if res_dp_chunk.is_err:
                return res_dp_chunk
            return res_dp_chunk
      

    @staticmethod
    async def get_by_chunk_index(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        index:int,
        timeout:int =120,
        max_attempts:int = 5,
        delay:int =1,
        backoff_factor:int =2,
        force:bool =True,
        max_parallel_gets:int =10,
        headers:Dict[str,str]= {},
        chunk_size:str="256kb",
        http2:bool = False,
        max_backoff:int =3
    )->Tuple[Chunk, InterfaceX.Metadata]:
        """Download a single chunk by its index, with retry and exponential backoff.

        Args:
            client: mictlanx async client.
            bucket_id: Source bucket.
            ball_id: Object key.
            index: Zero-based chunk index.
            timeout: Request timeout in seconds.
            max_attempts: Maximum download retries.
            delay: Base retry delay in seconds.
            backoff_factor: Multiplier applied to ``delay`` on each retry.
            force: Pass ``force=True`` to the underlying client.
            max_parallel_gets: Maximum concurrent sub-requests.
            headers: Extra HTTP headers.
            chunk_size: Target size per received chunk.
            http2: Use HTTP/2.
            max_backoff: Maximum delay cap in seconds.

        Returns:
            Tuple[Chunk, InterfaceX.Metadata]: ``(chunk, metadata)`` on success.
        """
        i =0
        while i <= max_attempts :
            x = await client.get_chunk(
                bucket_id         = bucket_id,
                ball_id           = ball_id,
                index             = index,
                max_parallel_gets = max_parallel_gets,
                headers           = headers,
                chunk_size        = chunk_size,
                timeout           = timeout,
                http2             = http2,
                max_retries       = max_attempts,
                delay             = delay,
                backoff_factor    = backoff_factor,
                force             = force, 
                max_backoff       = max_backoff
            )
            if x.is_err:
                e = x.unwrap_err()
                print(f"Retrying in {delay} seconds... (Attemp {i}/{max_attempts})")
                await asyncio.sleep(delay)
                i+=1
                continue
            return x.unwrap()
        
    @staticmethod
    async def segment_and_put_lazy(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        path:str,
        row_chunk_size:int =100,
        max_attempts:int = 10,
        timeout:int=120,max_backoff:int =5,tags:Dict[str,str]={}
    )->AsyncGenerator[InterfaceX.PutChunkedResponse, None]:
        chunks_generator = RoryCommonUtils.read_chunks_numpy(ball_id=ball_id,filename=path,row_chunk=row_chunk_size)
        for c in chunks_generator:
            res = await Common.delete_and_put_chunk(

                client    = client,
                bucket_id = bucket_id,
                ball_id    = ball_id,
                chunk      = c,
                tags      = {**c.metadata,**tags},
                max_tries = max_attempts,
                timeout   = timeout,
                max_backoff=max_backoff 
            )
            if res.is_err:
                raise Exception(f"Failed to put a chunk: {c.chunk_id}")
            yield res.unwrap()

    @staticmethod
    async def read_numpy_from(path:str="",extension:str="")->Result[npt.NDArray,Exception]:
        try:
            if extension == "csv":
                plaintextMatrix = pd.read_csv(
                    path, 
                    header=None
                ).values
                return Ok(plaintextMatrix)
            elif extension == "npy":
                with open(path, "rb") as f:
                    plaintextMatrix = np.load(f)
                    return Ok(plaintextMatrix.astype(np.float64))
            else:
                return Err(Exception("Either path or extension  was not provided"))
        except Exception as e:
            return Err(e)
    
    # Serializer
    @staticmethod
    def from_pyctxt_list_to_bytes(xs:List[PyCtxt]):
        serialized_ciphertexts = [ctxt.to_bytes() for ctxt in xs]
        return pickle.dumps(serialized_ciphertexts)
    
    def from_pyctxt_matrix_to_bytes(xs:List[PyCtxt]):
        serialized_ciphertexts = []
        for ctxts in xs:
            inner_sctxts = []
            for ctxt in ctxts:
                inner_sctxts.append(ctxt.to_bytes())
            serialized_ciphertexts.append(inner_sctxts)
        return pickle.dumps(serialized_ciphertexts)
    
    def from_bytes_to_pyctxt_matrix(ckks:Ckks,x:bytes):
        yss = pickle.loads(x)
        scheme = ckks.he_object
        result = []
        for ys in yss: 
            tmp_row = []
            for y in ys:
                _y = PyCtxt(None,scheme,None,y,'FRACTIONAL')
                tmp_row.append(_y)
            result.append(tmp_row)
        _res = np.vstack(result)
        return _res
    

    @staticmethod
    def from_bytes_to_pyctxt_list_v1(ckks:Ckks,x:bytes)->List[PyCtxt]:
        scheme  = ckks.he_object
        xx      = list(map(lambda x: PyCtxt(None,scheme,None,x,'FRACTIONAL'), x))
        return xx


    @staticmethod
    def from_pyctxts_to_chunks(key:str,xs:List[PyCtxt],num_chunks:int=2)->Option[Chunks]:
        """Serialize a flat list of ``PyCtxt`` objects into a ``Chunks`` object.

        Ciphertexts are pickled, then split into ``num_chunks`` chunks. Each chunk
        carries its index in the metadata so ``get_pyctxt`` can reassemble them in
        order.

        Args:
            key: Object key — used as the ``group_id`` for chunk metadata.
            xs: Flat list of ``PyCtxt`` ciphertexts.
            num_chunks: Number of chunks to split the list into.

        Returns:
            ``Some(Chunks)`` on success, ``None`` on failure.
        """
        try:
            n = len(xs)
            if n==0:
                raise ValueError("Input list of PyCtxt is empty.")
            if n == 1:
                chunk = Chunk.from_list(
                    group_id = key,
                    index    = 0,
                    chunk_id = Some(f"{key}_0"),
                    metadata = {
                        "num_chunks": "1",
                        "shape"     : str((1,)),
                    },
                    xs       = [xs[0].to_bytes()],
                )
                return Some(Chunks(chs=[chunk], n=1, strict=False))
            
            if num_chunks > n:
                num_chunks = n
            chs = Chunks._iter_to_chunks(num_chunks=num_chunks,chunk_prefix=Some(key),group_id=key,n=n,iterable=xs)
            def __inner():
                for c in chs:
                    chunk_id = Some(c.get("chunk_id",None)).filter(lambda x: not x == None)
                    data = [x.to_bytes() for x in c["data"]]
                    c_tmp = Chunk.from_list(
                        group_id = c["group_id"],
                        index    = c["index"],
                        chunk_id = chunk_id,
                        metadata = c["metadata"],
                        xs       = data,
                    )
                    yield c_tmp
            return Some(Chunks(chs= __inner(), n=n, strict=False))
        except Exception as e:
            from option import NONE
            return NONE
    

    @staticmethod
    def from_pyctxt_matrix_to_chunks(key:str,xs:List[List[PyCtxt]],num_chunks:int=2)->Option[Chunks]:
        try:
            n = len(xs)
            chs = Chunks._iter_to_chunks(num_chunks=num_chunks,chunk_prefix=Some(key),group_id=key,n=n,iterable=xs)
            def __inner():
                for c in chs:
                    chunk_id = Some(c.get("chunk_id",None)).filter(lambda x: not x == None)
                    ys    = c["data"]
                    data  = Common.from_pyctxt_matrix_to_bytes(ys)
                    c_tmp = Chunk(
                        group_id = c["group_id"],
                        index    = c["index"],
                        chunk_id = chunk_id,
                        data     = data
                    )
                    yield c_tmp
            return Some(Chunks(chs= __inner() , n = n ))
        except Exception as e:
            print(e)
            return e
        

    @staticmethod
    def from_chunks_to_pyctxts_list(ckks:Ckks, chunks:Chunks)->List[PyCtxt]:
        chunks.sort()
        xs = []
        for c in chunks:
            x = c.to_list().unwrap()
            xx = Common.from_bytes_to_pyctxt_list(ckks=ckks, xs=x)
            xs.extend(xx)
        return xs


    @staticmethod
    def from_bytes_to_pyctxt_list(ckks:Ckks,xs:List[bytes])->List[PyCtxt]:
        scheme  = ckks.he_object
        xx = []
        for x in xs:
            y = PyCtxt(None, scheme, None,x, "FRACTIONAL")
            xx.append(y)
        return xx
    

    @staticmethod
    def from_bytes_to_pyctxt_list_v2(ckks:Ckks, x:bytes):
        xs = pickle.loads(x)
        scheme = ckks.he_object
        xx = [PyCtxt(None, scheme, None, x, "FRACTIONAL") for x in xs ]
        return xx


    @staticmethod
    def encrypt_chunk_liu(key:str,dataowner:DataOwner,chunk:Chunk, np_random:bool)-> Chunk:
        ptm = chunk.to_ndarray().unwrap()
        encyrpted_chunk:npt.NDArray = dataowner.outsourcedData(ptm).encrypted_matrix
        return Chunk.from_ndarray(group_id=key, index= chunk.index, ndarray= encyrpted_chunk, chunk_id=Some("{}_{}".format(key,chunk.index)))


    @staticmethod
    def to_chunks_generator(awaitable_chunks:List[Awaitable[Chunk]]):
        xs = list(map(lambda fut: fut.result(), awaitable_chunks))
        return xs


    #  Segmentation
    @staticmethod
    def segment_and_encrypt_liu(key:str,dataowner:DataOwner,plaintext_matrix:npt.NDArray, n:int, np_random:bool, num_chunks:int=2,max_workers:int = int(os.cpu_count()/2))->Chunks:
        """Segment a matrix and Liu-encrypt each chunk using an internal process pool.

        .. deprecated::
            Use ``StorageBackend.put`` with ``Scheme.LIU`` instead.
            Will be removed in rory-common 1.0.0.

        Args:
            key: Object key — used as the ``group_id`` for chunk metadata.
            dataowner: Liu-scheme data owner that performs the per-chunk encryption.
            plaintext_matrix: Matrix to segment and encrypt.
            n: Total number of elements (``matrix.size``) — stored in ``Chunks.n``.
            np_random: Use numpy's random number generator inside the Liu scheme.
            num_chunks: Number of chunks to split the matrix into.
            max_workers: Process pool size (defaults to half the CPU count).

        Returns:
            ``Chunks`` object whose iterator yields encrypted ``Chunk`` instances.
        """
        warnings.warn(
            "segment_and_encrypt_liu is deprecated and will be removed in rory-common 1.0.0. "
            "Use StorageBackend.put with Scheme.LIU instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            awaitable_chunks:List[Awaitable[Chunk]] = []
            for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
                future = executor.submit(Common.encrypt_chunk_liu,key = key, dataowner = dataowner,chunk = plaintext_matrix_chunk, np_random = np_random)
                awaitable_chunks.append(future)
            return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )


    @staticmethod
    def segment_and_encrypt_liu_with_executor(executor:ProcessPoolExecutor,key:str,dataowner:DataOwner,plaintext_matrix:npt.NDArray, n:int, np_random:bool, num_chunks:int=2 )->Chunks:
        """Segment a matrix and Liu-encrypt each chunk using a caller-supplied executor.

        .. deprecated::
            Use ``StorageBackend.put`` with ``Scheme.LIU`` instead.
            Will be removed in rory-common 1.0.0.

        Args:
            executor: Running ``ProcessPoolExecutor`` to submit work to.
            key: Object key — used as the ``group_id`` for chunk metadata.
            dataowner: Liu-scheme data owner that performs the per-chunk encryption.
            plaintext_matrix: Matrix to segment and encrypt.
            n: Total number of elements (``matrix.size``) — stored in ``Chunks.n``.
            np_random: Use numpy's random number generator inside the Liu scheme.
            num_chunks: Number of chunks to split the matrix into.

        Returns:
            Chunks: object whose iterator yields encrypted Chunk instances.
        """
        warnings.warn(
            "segment_and_encrypt_liu_with_executor is deprecated and will be removed in rory-common 1.0.0. "
            "Use StorageBackend.put with Scheme.LIU instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        plaintext_matrix_chunks:Chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(Common.encrypt_chunk_liu,key = key, dataowner = dataowner,chunk = plaintext_matrix_chunk, np_random = np_random)
            awaitable_chunks.append(future)
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )

    @staticmethod
    def segment_and_encrypt_liu_timed(
        key: str,
        dataowner: DataOwner,
        plaintext_matrix: npt.NDArray,
        n: int,
        np_random: bool,
        num_chunks: int = 2,
        max_workers: int = int(os.cpu_count() / 2),
    ) -> Tuple[Chunks, float, float]:
        """Segment a matrix and Liu-encrypt each chunk using an internal process pool.

        Returns timing data alongside the encrypted chunks, matching the pattern of
        ``segment_and_encrypt_ckks_with_initialized_executor``.

        Args:
            key: Object key — used as the ``group_id`` for chunk metadata.
            dataowner: Liu-scheme data owner that performs the per-chunk encryption.
            plaintext_matrix: Matrix to segment and encrypt.
            n: Total number of elements (``matrix.size``) — stored in ``Chunks.n``.
            np_random: Use numpy's random number generator inside the Liu scheme.
            num_chunks: Number of chunks to split the matrix into.
            max_workers: Process pool size (defaults to half the CPU count).

        Returns:
            Tuple of ``(Chunks, segment_time, encrypt_time)`` where times are in seconds.
        """
        t0 = T.monotonic()
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray=plaintext_matrix, group_id=key, num_chunks=num_chunks).unwrap()
        t1 = T.monotonic()
        segment_time = t1 - t0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            awaitable_chunks: List[Awaitable[Chunk]] = []
            for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
                future = executor.submit(Common.encrypt_chunk_liu, key=key, dataowner=dataowner, chunk=plaintext_matrix_chunk, np_random=np_random)
                awaitable_chunks.append(future)
            chs = Common.to_chunks_generator(awaitable_chunks=awaitable_chunks)
        encrypt_time = T.monotonic() - t1
        return (Chunks(chs=chs, n=n), segment_time, encrypt_time)

    @staticmethod
    def segment_and_encrypt_liu_with_executor_timed(
        executor: ProcessPoolExecutor,
        key: str,
        dataowner: DataOwner,
        plaintext_matrix: npt.NDArray,
        n: int,
        np_random: bool,
        num_chunks: int = 2,
    ) -> Tuple[Chunks, float, float]:
        """Segment a matrix and Liu-encrypt each chunk using a caller-supplied executor.

        Prefer this over ``segment_and_encrypt_liu_timed`` when the caller already owns
        a ``ProcessPoolExecutor`` (e.g. ``StorageBackend.put``). Returns timing data
        alongside the encrypted chunks, matching the pattern of
        ``segment_and_encrypt_ckks_with_initialized_executor``.

        Args:
            executor: Running ``ProcessPoolExecutor`` to submit work to.
            key: Object key — used as the ``group_id`` for chunk metadata.
            dataowner: Liu-scheme data owner that performs the per-chunk encryption.
            plaintext_matrix: Matrix to segment and encrypt.
            n: Total number of elements (``matrix.size``) — stored in ``Chunks.n``.
            np_random: Use numpy's random number generator inside the Liu scheme.
            num_chunks: Number of chunks to split the matrix into.

        Returns:
            Tuple of ``(Chunks, segment_time, encrypt_time)`` where times are in seconds.
        """
        t0 = T.monotonic()
        plaintext_matrix_chunks: Chunks = Chunks.from_ndarray(ndarray=plaintext_matrix, group_id=key, num_chunks=num_chunks).unwrap()
        t1 = T.monotonic()
        segment_time = t1 - t0
        awaitable_chunks: List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(Common.encrypt_chunk_liu, key=key, dataowner=dataowner, chunk=plaintext_matrix_chunk, np_random=np_random)
            awaitable_chunks.append(future)
        chs = Common.to_chunks_generator(awaitable_chunks=awaitable_chunks)
        encrypt_time = T.monotonic() - t1
        return (Chunks(chs=chs, n=n), segment_time, encrypt_time)

    
    @staticmethod
    def init_liu_worker_context(liu_params: "LiuParams"):
        """Runs once per worker process to construct the Liu cipher in RAM.

        Args:
            liu_params: Liu scheme construction parameters.
        """
        try:
            _liu = Liu(
                _round         = liu_params._round,
                decimals       = liu_params.decimals,
                secure_random  = liu_params.secure_random,
                seed           = liu_params.seed,
                use_np_random  = liu_params.use_np_random,
                security_level = liu_params.security_level,
            )
            _liu.generate_keys(security_level=liu_params.security_level)
            Common.liu_dataowner = _liu
        except Exception as e:
            print(f"Failed to initialize Liu worker context: {e}")
            raise e

    @staticmethod
    def encrypt_chunk_liu_with_initialized_executor(key: str, chunk: Chunk, np_random: bool) -> Chunk:
        """Encrypt a single chunk using the pre-initialized Liu cipher in this worker process.

        Must be run inside a ``ProcessPoolExecutor`` whose ``initializer`` was
        ``init_liu_worker_context``.

        Args:
            key: Object key — used as the ``group_id`` for chunk metadata.
            chunk: Plaintext chunk to encrypt.
            np_random: Use numpy's random number generator inside the Liu scheme.

        Returns:
            Encrypted ``Chunk``.
        """
        try:
            _liu = Common.liu_dataowner
            if _liu is None:
                raise Exception("Liu scheme not initialized. Please run init_liu_worker_context first.")
            ptm = chunk.to_ndarray().unwrap()
            encrypted: npt.NDArray = _liu.encrypt_matrix(plaintext_matrix=ptm).data
            return Chunk.from_ndarray(group_id=key, index=chunk.index, ndarray=encrypted, chunk_id=Some("{}_{}".format(key, chunk.index)))
        except Exception as e:
            print("ENCRYPT_CHUNK_LIU_ERROR", e)
            raise e

    @staticmethod
    def segment_and_encrypt_liu_with_initialized_executor_timed(
        key: str,
        plaintext_matrix: npt.NDArray,
        n: int,
        np_random: bool,
        liu_params: "LiuParams",
        num_chunks: int = 2,
    ) -> Tuple[Chunks, float, float]:
        """Segment a matrix and Liu-encrypt each chunk using a process pool with a pre-initialized cipher.

        Each worker loads the Liu context once via ``init_liu_worker_context`` rather than
        rebuilding a ``Liu`` object for every task — mirrors the CKKS
        ``segment_and_encrypt_ckks_with_initialized_executor`` pattern.

        Args:
            key: Object key — used as the ``group_id`` for chunk metadata.
            plaintext_matrix: Matrix to segment and encrypt.
            n: Total number of elements (``matrix.size``) — stored in ``Chunks.n``.
            np_random: Use numpy's random number generator inside the Liu scheme.
            liu_params: Liu scheme construction parameters forwarded to workers.
            num_chunks: Number of chunks (also the process pool size).

        Returns:
            Tuple of ``(Chunks, segment_time, encrypt_time)`` where times are in seconds.
        """
        t0 = T.monotonic()
        executor = ProcessPoolExecutor(
            max_workers = num_chunks,
            initializer = Common.init_liu_worker_context,
            initargs    = (liu_params,),
        )
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray=plaintext_matrix, group_id=key, num_chunks=num_chunks).unwrap()
        t1 = T.monotonic()
        segment_time = t1 - t0
        awaitable_chunks: List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(
                Common.encrypt_chunk_liu_with_initialized_executor,
                key       = key,
                chunk     = plaintext_matrix_chunk,
                np_random = np_random,
            )
            awaitable_chunks.append(future)
        chs = Common.to_chunks_generator(awaitable_chunks=awaitable_chunks)
        encrypt_time = T.monotonic() - t1
        executor.shutdown(wait=True)
        return (Chunks(chs=chs, n=n), segment_time, encrypt_time)

    @staticmethod
    def segment_and_encrypt_fdhope(scheme:str, key:str,dataowner:Fdhope,plaintext_matrix:npt.NDArray, n:int ,num_chunks:int=2, threshold:float = 0.0, max_workers:int = int(os.cpu_count()/2) )->Chunks:
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            awaitable_chunks:List[Awaitable[Chunk]] = []
            for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
                future = executor.submit(Common.encrypt_chunk_fdhope, key = key, dataowner = dataowner, chunk = plaintext_matrix_chunk, scheme = scheme, sens = threshold)
                awaitable_chunks.append(future)
            return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )


    @staticmethod
    def segment_and_encrypt_fdhope_with_executor(executor:ProcessPoolExecutor,scheme:str, key:str,dataowner:Fdhope,matrix:npt.NDArray, n:int ,num_chunks:int=2, sens:float = 0.00001 ):
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray= matrix, group_id = key, num_chunks= num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(Common.encrypt_chunk_fdhope, key = key, dataowner = dataowner, chunk = plaintext_matrix_chunk, scheme = scheme, sens = sens)
            awaitable_chunks.append(future)
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)


    @staticmethod
    def encrypt_chunk_fdhope(key:str,dataowner:Fdhope,chunk:Chunk,scheme:str,sens:float=0.00001)-> Chunk:
        try:
            plaintext = chunk.to_ndarray().unwrap()
            if scheme == "DBSKMEANS":
                encrypted = dataowner.encrypt_tensor(plaintext_tensor=plaintext, sens=sens).data
            elif scheme == "DBSNNC":
                encrypted = dataowner.encrypt_matrix(plaintext_matrix=plaintext, sens=sens).data
            else:
                raise ValueError(f"Unknown FDHOPE algorithm: {scheme}")
            return Chunk.from_ndarray(group_id=key, index= chunk.index, ndarray=encrypted, chunk_id=Some("{}_{}".format(key,chunk.index)))
        except Exception as e:
            print("ERROR", e)
            raise e
    

    @staticmethod
    def chunks_to_bytes_gen(chs:Chunks) -> Generator[bytes,None,None]:
        for chunk in chs.iter():
            yield chunk.data


    @staticmethod
    def encrypt_vector_ckks_with_initialized_executor(
        # executor:ProcessPoolExecutor,
        key:str,
        vector:npt.NDArray,
        path: str,
        ctx_filename:str="ctx",
        pubkey_filename:str="pubkey",
        secretkey_filename:str="secretkey",
        relinkey_filename:str="",
        rotatekey_filename:str="",
        decimals:int=2,
        _round:bool = False,
        max_workers:int = int(os.cpu_count()/2)
    ) -> "tuple[Chunks, float, float]":
        """Encrypt a 1-D vector with CKKS using an internal process pool with pre-initialized context.

        A thin wrapper around ``segment_and_encrypt_ckks_with_initialized_executor`` that fixes
        ``num_chunks=1`` so the whole vector is encrypted as a single ciphertext chunk.

        Returns:
            Tuple of ``(chunks, segment_time, encrypt_time)`` where ``segment_time`` and
            ``encrypt_time`` are wall-clock durations in seconds.
        """
        return Common.segment_and_encrypt_ckks_with_initialized_executor(
            # executor           = executor,
            key                = key,
            plaintext_matrix   = vector,
            n                  = vector.size,
            num_chunks         = 1,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename,
            rotatekey_filename = rotatekey_filename,
            decimals           = decimals,
            path               = path,
            _round             = _round,
            max_workers         = max_workers
        )
    
    
    @staticmethod
    def segment_and_encrypt_ckks_with_initialized_executor(
        # executor:ProcessPoolExecutor,
        key:str, 
        plaintext_matrix:npt.NDArray,
        n:int, 
        _round:bool,
        decimals:int,
        path:str,
        ctx_filename:str,
        pubkey_filename:str,
        secretkey_filename:str,
        num_chunks:int=2,
        relinkey_filename:str="",
        rotatekey_filename:str="",
        max_workers:int = int(os.cpu_count()/2)
    ):
        """Segment a matrix and CKKS-encrypt each chunk using an internal process pool with pre-initialized context.

        Returns:
            Tuple of ``(chunks, segment_time, encrypt_time)`` where ``segment_time`` is the time
            spent splitting the matrix into chunks and ``encrypt_time`` is the time spent
            submitting encryption tasks to the pool — both in seconds.
        """
        t0 = T.monotonic()
        executor = ProcessPoolExecutor(
            max_workers = max_workers if num_chunks > max_workers else num_chunks,
            initializer = Common.init_ckks_worker_context,
            initargs    = (path, ctx_filename, pubkey_filename, secretkey_filename, relinkey_filename, rotatekey_filename, _round, decimals)
        )   
        plaintext_matrix_chunks = Chunks.from_ndarray( ndarray = plaintext_matrix, group_id = key, num_chunks = num_chunks).unwrap()
        t1 = T.monotonic()
        segment_time = t1 - t0
        awaitable_chunks:List[Awaitable[Chunk]] = []
        # encrypt_times = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(
                Common.encrypt_chunk_ckks_with_initialized_executor,
                    key      = key,
                    chunk    = plaintext_matrix_chunk,
                )
            awaitable_chunks.append(future)

        chs = Common.to_chunks_generator(awaitable_chunks=awaitable_chunks)
        encrypt_time = T.monotonic() - t1
        chunks = Chunks(chs= chs,n =n)
        executor.shutdown(wait=True)
        return (chunks, segment_time, encrypt_time)
    
    @staticmethod
    async def segement_and_encrypt_ckks_with_initialized_executor_put_chunks(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        key:str, 
        plaintext_matrix:npt.NDArray,
        n:int, 
        keys_path:str,
        num_chunks:int = 2, 
        timeout:int = 120,
        max_retries:int = 5,
        tags:Dict[str,str] = {},
        ctx_filename:str = "ctx",
        pubkey_filename:str = "pubkey",
        secretkey_filename:str = "secretkey",
        relinkey_filename:str = "",
        rotatekey_filename:str = "",
        decimals:int = 2,
        _round:bool = False,
        max_workers:int = int(os.cpu_count()/2)

    ):
        
        (encrypted_chunks,segment_time,encrypt_time) = Common.segment_and_encrypt_ckks_with_initialized_executor(
            key                = key,
            plaintext_matrix   = plaintext_matrix,
            n                  = n,
            num_chunks         = num_chunks,
            path               = keys_path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename,
            rotatekey_filename = rotatekey_filename,
            decimals           = decimals,
            _round             = _round,
            max_workers         = max_workers
        )
        t1 = T.monotonic()
        put_result = await Common.put_chunks(
            client      = client,
            bucket_id   = bucket_id,
            key         = ball_id,
            chunks      = encrypted_chunks,
            max_retries = max_retries,
            timeout     = timeout,
            tags        = tags,
        )
        upload_time = T.monotonic() - t1
        return (put_result,segment_time, encrypt_time, upload_time)
    



    @staticmethod
    def segment_and_encrypt_ckks_with_executor(
        executor:ProcessPoolExecutor,
        key:str,
        plaintext_matrix:npt.NDArray,
        n:int,
        num_chunks:int=2,
    ):
        plaintext_matrix_chunks = Chunks.from_ndarray( ndarray = plaintext_matrix, group_id = key, num_chunks = num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(
                Common.encrypt_chunk_ckks_with_initialized_executor,
                key                = key,
                chunk              = plaintext_matrix_chunk,
            )
            awaitable_chunks.append(future)
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)
    



    @staticmethod
    def encrypt_chunk_ckks(key:str, chunk:Chunk, _round:bool, decimals:int, path:str, ctx_filename:str, pubkey_filename:str, secretkey_filename:str, relinkey_filename:str = "", rotatekey_filename:str = "")-> Chunk:
        try:
            ckks = Ckks.from_pyfhel(
                    _round             = _round,
                    decimals           = decimals,
                    path               = path,
                    ctx_filename       = ctx_filename,
                    pubkey_filename    = pubkey_filename,
                    secretkey_filename = secretkey_filename,
                    relinkey_filename  = relinkey_filename,
                    rotatekey_filename =  rotatekey_filename 

                )
            plaintext_matrix = chunk.to_ndarray().unwrap().copy()
            encyrpted_chunk:List[PyCtxt] = ckks.encrypt_matrix(plaintext_matrix=plaintext_matrix).data
            data = Common.from_pyctxt_list_to_bytes(xs=encyrpted_chunk)
            c= Chunk(group_id = key, index = chunk.index, data = data, chunk_id = Some("{}_{}".format(key,chunk.index)))
            return c
        except Exception as e:
            print("ENCRYPT_CHUNK_ERROR",e)
    


    @staticmethod
    def encrypt_chunk_ckks_with_initialized_executor(key:str, chunk:Chunk)-> Chunk:
        try:
            if Common.ckks is None:
                raise Exception("CKKS context not initialized. Please run init_ckks_worker_context first.")
  
            plaintext_matrix = chunk.to_ndarray().unwrap().copy()
            encyrpted_chunk:List[PyCtxt] = Common.ckks.encrypt_matrix(plaintext_matrix=plaintext_matrix).data
            data = Common.from_pyctxt_list_to_bytes(xs=encyrpted_chunk)
            c= Chunk(group_id = key, index = chunk.index, data = data, chunk_id = Some("{}_{}".format(key,chunk.index)))
            return c
        except Exception as e:
            print("ENCRYPT_CHUNK_ERROR",e)
            raise e
    

    @staticmethod
    def segment_and_encrypt_ckks_with_executor_v2(
        executor:ProcessPoolExecutor,
        key:str,
        plaintext_matrix:npt.NDArray,
        n:int,
        _round:bool, 
        decimals:int, 
        path:str, 
        ctx_filename:str, 
        pubkey_filename:str, 
        secretkey_filename:str,
        num_chunks:int=2, 
        relinkey_filename:str="",
        rotatekey_filename:str=""
    ):
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray = plaintext_matrix, group_id = key, num_chunks = num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(
                Common.encrypt_chunk_ckks_v2,
                key                = key,
                chunk              = plaintext_matrix_chunk,
                _round             = _round,
                decimals           = decimals,
                path               = path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename
            )
            awaitable_chunks.append(future)
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)


    @staticmethod
    def encrypt_chunk_ckks_v2(
        key:str, 
        chunk:Chunk, 
        _round:bool, 
        decimals:int, 
        path:str, 
        ctx_filename:str, 
        pubkey_filename:str, 
        secretkey_filename:str, 
        relinkey_filename:str = "",
        rotatekey_filename:str = ""
        )-> Chunk:
        try:
            ckks = Ckks.from_pyfhel(
                    _round             = _round,
                    decimals           = decimals,
                    path               = path,
                    ctx_filename       = ctx_filename,
                    pubkey_filename    = pubkey_filename,
                    secretkey_filename = secretkey_filename,
                    relinkey_filename  = relinkey_filename,
                    rotatekey_filename = rotatekey_filename
                )
            plaintext_matrix = chunk.to_ndarray().unwrap().copy()
            encyrpted_chunk:List[List[PyCtxt]] = ckks.encrypt_matrix_list(plaintext_matrix=plaintext_matrix)
            data = Common.from_pyctxt_matrix_to_bytes(xs=encyrpted_chunk)
            return Chunk(group_id = key, index = chunk.index, data = data, chunk_id = Some("{}_{}".format(key,chunk.index)))
        except Exception as e:
            print("ENCRYPT_CHUNK_ERROR",e)
            raise e
    

    @staticmethod
    def from_chunks_to_pyctxt_list(chunks:Chunks, ckks:Ckks)->List[List[PyCtxt]]:
        xs = []
        for ch in chunks.iter():
            x  = pickle.loads(ch.data)
            xx = Common.from_bytes_to_pyctxt_list(ckks=ckks,xs=x)
            xs.append(xx)
        return xs
    

    def verify_mean_error(old_matrix:npt.NDArray, new_matrix:npt.NDArray, min_error:float=0.15):
        mean_error = np.mean(np.abs((old_matrix - new_matrix) / old_matrix))
        return mean_error <= min_error


    @staticmethod
    def from_chunks_to_pyctxt_matrix(chunks:Chunks, ckks:Ckks)->List[PyCtxt]:
        xs = []
        for ch in chunks.iter():
            x = ch.data
            x  = pickle.loads(x)
            xx = Common.from_list_bytes_to_pyctxt_matrix(ckks=ckks,xs=x)
            xs.extend(xx)
        return xs
    

    @staticmethod
    def from_list_bytes_to_pyctxt_matrix(ckks:Ckks,xs:List[bytes])->List[PyCtxt]:
        scheme  = ckks.he_object
        matrix = []
        for xs in xs:
            tmp_row = []
            for x in xs:
                element = PyCtxt(None, scheme, None,x, "FRACTIONAL")
                tmp_row.append(element)
            matrix.append(tmp_row)
        return matrix
    @staticmethod
    async def while_not_delete_key(client:AsyncClient ,bucket_id:str, key:str,timeout:int = 3600,max_tries:int = 5): 
        n_deletes = -1
        i = 0
        while (n_deletes ==-1 or n_deletes >0) and i <= max_tries:
            _delete_result = await client.delete_by_key(bucket_id=bucket_id,key=key,timeout=timeout,force = True)

            if _delete_result.is_ok:
                del_response = _delete_result.unwrap()
                n_deletes = del_response.n_deletes
                L.debug({
                    "event":"WHILE.NOT.DELETE.KEY.SUCCESS",
                    "bucket_id":bucket_id,
                    "key":key,
                    "n_deletes":n_deletes,
                    "i":i, 
                    "max_tries":max_tries,
                    "ok":_delete_result.is_ok
                 })
                if n_deletes == 0:
                    return n_deletes
                else:
                    i+=1
                    L.debug({
                        "event":"WHILE.NOT.DELETE.KEY.RETRY",
                        "bucket_id":bucket_id,
                        "key":key,
                        "n_deletes":n_deletes,
                        "i":i,
                        "max_tries":max_tries,
                    })
            else:
                L.error({
                    "error":str(_delete_result.unwrap_err()),
                    "bucket_id":bucket_id,
                    "key":key,
                    "i":i,
                    "max_tries":max_tries,
                    "n_deletes":n_deletes,

                })
                i+=1
        return n_deletes

    @staticmethod
    async def while_not_delete_ball_id(STORAGE_CLIENT:AsyncClient ,bucket_id:str, key:str,timeout:int = 3600,max_tries:int = 5): 
        n_deletes = -1
        i = 0
        while (n_deletes ==-1 or n_deletes >0) and i <= max_tries:
            _delete_result = await STORAGE_CLIENT.delete(bucket_id=bucket_id,ball_id=key,timeout=timeout,force = True)
            # print(_delete_result)
            # T.sleep(100)
            if _delete_result.is_ok:
                del_response = _delete_result.unwrap()
                n_deletes = del_response.n_deletes
                L.debug({
                    "event":"WHILE.NOT.DELETE.BALL_ID.SUCCESS",
                    "bucket_id":bucket_id,
                    "ball_id":key,
                    "n_deletes":n_deletes,
                    "i":i, 
                    "max_tries":max_tries,
                    "ok":_delete_result.is_ok
                 })
                if n_deletes == 0:
                    return n_deletes
                else:
                    L.debug({
                        "event":"WHILE.NOT.DELETE.BALL_ID.RETRY",
                        "bucket_id":bucket_id,
                        "ball_id":key,
                        "n_deletes":n_deletes,
                        "i":i,
                        "max_tries":max_tries,
                    })
                    i+=1
            else:
                L.warning({
                    "error":str(_delete_result.unwrap_err()),
                    "bucket_id":bucket_id,
                    "ball_id":key,
                    "i":i,
                    "max_tries":max_tries,
                    "n_deletes":n_deletes,

                })
                i+=1
        return n_deletes
    

    @staticmethod
    async def delete_and_put_bytes(
        client:AsyncClient,
        bucket_id:str,
        key:str,
        data:bytes, 
        chunk_size:str="128kb",
        tags:Dict[str,str]={},
        timeout:int = 3600,
        max_tries:int =5
    )->Result[bool,Exception]:
        condition = True
        put_res   = None
        i         = 0
        while  i < max_tries: 
            _delete_result = await Common.while_not_delete_ball_id( STORAGE_CLIENT = client, bucket_id = bucket_id, key = key,max_tries=max_tries)
            
            put_res = await client.put(bucket_id = bucket_id, key = key, value = data, chunk_size = chunk_size, tags = tags, timeout = timeout)
            if put_res.is_ok:
                return put_res
            else:
                L.error({
                    "event":"PUT.BYTES.FAILED",
                    "error":str(put_res.unwrap_err())
                })
                i+=1            


            condition = not (_delete_result == 0)
            if condition:
                L.error({
                    "event":"DELETE.BALL_ID.FAILED.RETRY",
                    "error":str(put_res.unwrap_err())
                })
                print(f"Put failed reytring in 1 second... Attemp {i+1}/{max_tries}")
                i+=1
                await asyncio.sleep(1)
        return put_res
   
    @staticmethod
    async def delete_and_put_chunk(
        client:AsyncClient,
        bucket_id:str,
        ball_id:str,
        chunk:Chunk, 
        tags:Dict[str,str]={},
        timeout:int = 3600,
        max_tries:int =5,
        max_backoff:int =5
    )->Result[bool,Exception]:
        condition = True
        put_res = None
        i = 0
        while  i < max_tries: 
            _delete_result = await Common.while_not_delete_key(client = client, bucket_id = bucket_id, key = chunk.chunk_id,timeout=timeout,max_tries=max_tries)
            
            put_res = await client.put_single_chunk(
                bucket_id   = bucket_id,
                ball_id     = ball_id,
                chunk       = chunk,
                tags        = tags,
                timeout     = timeout,
                max_tries   = max_tries,
                max_backoff = max_backoff
            )

            L.debug({
                "event":"DELETE.COMPLETED",
                "bucket_id":bucket_id,
                "key":ball_id,
                "n_deletes":_delete_result,
                "put_ok": put_res.is_ok
            })
            if put_res.is_ok:
                return put_res
            else:
                L.error({
                    "event":"PUT.CHUNK.FAILED",
                    "error":str(put_res.unwrap_err()),
                    "i":i
                })
                i+=1
            
            # condition = put_res.is_err or _delete_result >0
            condition =  not (_delete_result == 0)
            # and not (_delete_result == 0)
            if condition:
                L.error({
                    "event":"DELETE.FAILED.RETRY",
                    "error":str(put_res.unwrap_err()),
                    "i":i
                })
                print(f"Put failed reytring in 1 second... Attemp {i+1}/{max_tries}")
                i+=1
                await asyncio.sleep(1)
        L.debug(
            {
                "event":"DELETE.PUT.CHUNKS",
                "bucket_id":bucket_id,
                "key":ball_id,
                "ok":put_res.is_ok
            }
        )
        return put_res

    @staticmethod
    async def delete_and_put_chunks(
        client:AsyncClient,
        bucket_id:str,
        key:str,
        chunks:Chunk, 
        tags:Dict[str,str]={},
        timeout:int = 3600,
        max_tries:int =5
    )->Result[InterfaceX.PutChunkedResponse,Exception]:
        condition = True
        put_res = None
        i = 0
        while  i < max_tries: 
            _delete_result = await Common.while_not_delete_ball_id(STORAGE_CLIENT = client, bucket_id = bucket_id, key = key,timeout=timeout,max_tries=max_tries)

            put_res = await client.put_chunks(bucket_id = bucket_id, key = key, chunks = chunks, tags = tags, timeout = timeout)
            L.debug({
                "event":"DELETE.COMPLETED",
                "bucket_id":bucket_id,
                "key":key,
                "n_deletes":_delete_result,
                "put_ok": put_res.is_ok
            })
            if put_res.is_ok:
                return put_res
            
            # condition = put_res.is_err or _delete_result >0
            condition = put_res.is_err and not (_delete_result <=0)
            # and not (_delete_result == 0)
            if condition:
                L.error({
                    "error":str(put_res.unwrap_err()),
                    "i":i
                })
                print(f"Put failed reytring in 1 second... Attemp {i+1}/{max_tries}")
                i+=1
                await asyncio.sleep(1)
                continue

        L.debug(
            {
                "event":"DELETE.PUT.CHUNKS",
                "bucket_id":bucket_id,
                "key":key,
                "ok":put_res.is_ok
            }
        )
        return put_res


    @staticmethod
    async def put_ndarray(client:AsyncClient,key:str,matrix:npt.NDArray,timeout:int =300,max_retries:int=5,tags:Dict[str,str]={},bucket_id:str= "rory")->Result[bool, Exception]:
        """Serialize a numpy array to bytes and store it as a single blob.

        Shape and dtype are embedded as metadata tags so retrieval can
        reconstruct the original array without side-channel information.

        Args:
            client: mictlanx async client.
            key: Object key.
            matrix: Array to store.
            timeout: Request timeout in seconds.
            max_retries: Maximum upload retries.
            tags: Extra metadata tags (merged with shape/dtype tags).
            bucket_id: Target bucket.

        Returns:
            Result[bool, Exception]: The raw put result from the underlying client.
        """
        put_chunks_generator_results = await Common.delete_and_put_bytes(
            client    = client,
            bucket_id = bucket_id,
            key       = key,
            data      = matrix.tobytes(),
            tags      = {
                "shape": str(matrix.shape),
                "dtype": str(matrix.dtype),
                **tags
            },
            timeout=timeout,
            max_tries=max_retries
        )
        return put_chunks_generator_results

    @staticmethod
    async def put_ndarray_no_delete(
        client:AsyncClient,
        key:str,
        matrix:npt.NDArray,
        timeout:int = 300,
        max_retries:int = 5,
        tags:Dict[str,str] = {},
        bucket_id:str = "rory",
    ) -> Result[bool, Exception]:
        """Serialize a numpy array to bytes and store it as a single blob without deleting first.

        Same serialization as ``put_ndarray`` (shape/dtype embedded as tags) but skips
        the internal delete step. Use this when the caller has already handled deletion
        or when no deletion is desired.

        Args:
            client: mictlanx async client.
            key: Object key.
            matrix: Array to store.
            timeout: Request timeout in seconds.
            max_retries: Maximum upload retries.
            tags: Extra metadata tags (merged with shape/dtype tags).
            bucket_id: Target bucket.

        Returns:
            Result[bool, Exception]: The raw put result from the underlying client.
        """
        merged_tags = {"shape": str(matrix.shape), "dtype": str(matrix.dtype), **tags}
        put_res = None
        for _ in range(max_retries):
            put_res = await client.put(
                bucket_id  = bucket_id,
                key        = key,
                value      = matrix.tobytes(),
                tags       = merged_tags,
                timeout    = timeout,
            )
            if put_res.is_ok:
                return put_res
            await asyncio.sleep(1)
        return put_res

    @staticmethod
    async def segment_and_encrypt_liu_and_put_chunks(
        executor:ProcessPoolExecutor,
        dataowner:DataOwner,
        n:int,
        np_random:bool,
        client:AsyncClient,
        key:str,matrix:npt.NDArray,
        num_chunks:int=2,
        tags:Dict[str,str]={},
        bucket_id:str= "rory"
    )->Tuple[Result[InterfaceX.PutChunkedResponse,Exception],float,float]:
        """This is only a convenience method that runs the segment_encrypt_and_putchunks and then puts the chunks. You can run them separately if you want more control over the process or want to do some processing in between."""
        return await Common.segment_encrypt_and_put_chunks(
            bucket_id=bucket_id,
            client=client,
            dataowner=dataowner,
            key=key,
            matrix=matrix,
            n=n,
            np_random=np_random,
            num_chunks=num_chunks,
            tags=tags,
            executor=executor
        )


    @staticmethod
    async def segment_encrypt_and_put_chunks(
        executor:ProcessPoolExecutor,
        dataowner:DataOwner,
        n:int,
        np_random:bool,
        client:AsyncClient,
        key:str,matrix:npt.NDArray,
        num_chunks:int=2,
        tags:Dict[str,str]={},
        bucket_id:str= "rory"
    )->Tuple[Result[InterfaceX.PutChunkedResponse,Exception],float,float]:
        t1     = T.time()
        chunks = Common.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = key,
            plaintext_matrix = matrix,
            dataowner        = dataowner,
            n                = n,
            num_chunks       = num_chunks,
            np_random        = np_random
        )
        seg_encrypt_rt = T.time() - t1
        put_chunks_generator_results = await Common.put_chunks(client = client, bucket_id = bucket_id, key = key, chunks = chunks, tags = tags)
        return put_chunks_generator_results,seg_encrypt_rt,T.time()-t1
    

    # @staticmethod
    # async def segment_encrypt_with_vector_ckks_and_put_chunks_with_executor(
    #     client: AsyncClient,
    #     bucket_id:str,
    #     executor:ProcessPoolExecutor, 
    #     key:str, 
    #     vector:npt.NDArray, 
    #     _round:bool,
    #     decimals:int,
    #     path:str,
    #     ctx_filename:str,
    #     pubkey_filename:str,
    #     secretkey_filename:str,
    #     relinkey_filename:str="",
    #     rotatekey_filename:str="",
    #     tags:Dict[str,str]={},
    #     timeout:int =300,
    #     max_attempts:int =5
    # ) :
    #     t1     = T.monotonic()
        
    #     chunks = Common.encrypt_vector_ckks_with_initialized_executor(
    #         executor           = executor,
    #         key                = key,
    #         vector             = vector,
    #         _round             = _round,
    #         decimals           = decimals,
    #         path               = path,
    #         ctx_filename       = ctx_filename,
    #         pubkey_filename    = pubkey_filename,
    #         secretkey_filename = secretkey_filename,
    #         relinkey_filename  = relinkey_filename,
    #         rotatekey_filename = rotatekey_filename 
    #     )
    #     encrypt_time = T.monotonic() - t1
    #     t1 = T.monotonic()
    #     put_result = await Common.delete_and_put_chunks(
    #         client    = client,
    #         bucket_id = bucket_id,
    #         key       = key,
    #         chunks    = chunks,
    #         timeout   = timeout,
    #         max_tries = max_attempts,
    #         tags      = tags
            
    #     )
    #     upload_time = T.monotonic() - t1
    #     return (put_result, encrypt_time, upload_time)
    
    # @staticmethod
    # async def segment_encrypt_vector_ckks_put_chunks_with_executor(
    #     client: AsyncClient,
    #     bucket_id: str,
    #     executor: ProcessPoolExecutor,
    #     key: str,
    #     vector: npt.NDArray,
    #     _round: bool,
    #     decimals: int,
    #     path: str,
    #     ctx_filename: str,
    #     pubkey_filename: str,
    #     secretkey_filename: str,
    #     relinkey_filename: str = "",
    #     rotatekey_filename: str = "",
    #     tags: Dict[str, str] = {},
    #     timeout: int = 300,
    #     max_attempts: int = 5,
    # ):
    #     """Encrypt a 1-D vector with a caller-supplied executor and upload with ``put_chunks``.

    #     Unlike ``segment_encrypt_with_vector_ckks_and_put_chunks_with_executor``, this
    #     method does **not** delete any existing object before uploading.  Pre-deletion is
    #     the caller's responsibility (e.g. via the ``delete`` flag on ``StorageBackend.put``).
    #     """
    #     t1 = T.monotonic()
    #     chunks = Common.encrypt_vector_ckks_with_initialized_executor(
    #         executor           = executor,
    #         key                = key,
    #         vector             = vector,
    #         _round             = _round,
    #         decimals           = decimals,
    #         path               = path,
    #         ctx_filename       = ctx_filename,
    #         pubkey_filename    = pubkey_filename,
    #         secretkey_filename = secretkey_filename,
    #         relinkey_filename  = relinkey_filename,
    #         rotatekey_filename = rotatekey_filename,
    #     )
    #     encrypt_time = T.monotonic() - t1
    #     t1 = T.monotonic()
    #     put_result = await Common.put_chunks(
    #         client      = client,
    #         bucket_id   = bucket_id,
    #         key         = key,
    #         chunks      = chunks,
    #         timeout     = timeout,
    #         max_retries = max_attempts,
    #         tags        = tags,
    #     )
    #     upload_time = T.monotonic() - t1
    #     return (put_result, encrypt_time, upload_time)

    @staticmethod
    async def segment_encrypt_with_vector_ckks_and_put_chunks_with_initialized_executor(
        client: AsyncClient,
        bucket_id:str,
        key:str,
        vector:npt.NDArray,
        _round:bool,
        decimals:int,
        path:str,
        ctx_filename:str,
        pubkey_filename:str,
        secretkey_filename:str,
        relinkey_filename:str="",
        rotatekey_filename:str="",
        tags:Dict[str,str]={},
        timeout:int =300,
        max_workers:int = 2,
        max_attempts:int =5
    ):
        try:
            # t1 = T.monotonic()
            (chunks,segement_time, encrypt_time) = Common.encrypt_vector_ckks_with_initialized_executor(
                # executor           = executor,
                key                = key,
                vector             = vector,
                _round             = _round,
                decimals           = decimals,
                path               = path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename,
                max_workers         = max_workers
                # max_workers
            )
            # encrypt_time = T.monotonic() - t1
            t1 = T.monotonic()
            put_result = await Common.put_chunks(
                client      = client,
                bucket_id   = bucket_id,
                key         = key,
                chunks      = chunks,
                timeout     = timeout,
                max_retries = max_attempts,
                tags        = tags,
            )
            upload_time = T.monotonic() - t1
            return (put_result, segement_time,  encrypt_time, upload_time)
            # executor = ProcessPoolExecutor(
            #     max_workers = max_workers,
            #     initializer = Common.init_ckks_worker_context,
            #     initargs    = (path, ctx_filename, pubkey_filename, secretkey_filename, relinkey_filename, rotatekey_filename, _round, decimals)
            # )
            
            
            # res = await Common.segment_encrypt_vector_ckks_put_chunks_with_executor(
            #     client             = client,
            #     bucket_id          = bucket_id,
            #     # executor           = executor,
            #     key                = key,
            #     vector             = vector,
            #     _round             = _round,
            #     decimals           = decimals,
            #     path               = path,
            #     ctx_filename       = ctx_filename,
            #     pubkey_filename    = pubkey_filename,
            #     secretkey_filename = secretkey_filename,
            #     relinkey_filename  = relinkey_filename,
            #     rotatekey_filename = rotatekey_filename,
            #     tags               = tags,
            #     timeout            = timeout,
            #     max_attempts       = max_attempts,
            # )
            # return res
        except Exception as e:
            raise e

    @staticmethod
    async def put_chunks(client:AsyncClient,key:str,chunks:Chunks,timeout:int=300,max_retries:int=5,tags:Dict[str,str]={},bucket_id:str= "rory")->Result[InterfaceX.PutChunkedResponse,Exception]:
        """Sort and upload a ``Chunks`` object to cloud storage.

        Sorts chunks by index before uploading to guarantee consistent ordering
        regardless of how the ``Chunks`` generator produced them.

        Args:
            client: mictlanx async client.
            key: Object key.
            chunks: Pre-built ``Chunks`` (plaintext or encrypted).
            timeout: Request timeout in seconds.
            max_retries: Maximum upload retries.
            tags: Extra metadata tags applied to each chunk.
            bucket_id: Target bucket.

        Returns:
            Result from ``delete_and_put_chunks``.
        """
        chunks.sort()
        put_chunks_generator_results = await Common.delete_and_put_chunks(
            client    = client,
            bucket_id = bucket_id,
            key       = key,
            chunks    = chunks,
            tags      = tags,
             timeout=timeout,
             max_tries=max_retries
        )
        return put_chunks_generator_results

    @staticmethod
    async def put_chunks_no_delete(
        client:AsyncClient,
        key:str,
        chunks:Chunks,
        timeout:int = 300,
        max_retries:int = 5,
        tags:Dict[str,str] = {},
        bucket_id:str = "rory",
    ) -> Result[InterfaceX.PutChunkedResponse, Exception]:
        """Sort and upload a ``Chunks`` object without deleting first.

        Same as ``put_chunks`` (sorts by index before uploading) but skips the
        internal delete step. Use this when the caller has already handled deletion
        or when no deletion is desired.

        Args:
            client: mictlanx async client.
            key: Object key.
            chunks: Pre-built ``Chunks`` (plaintext or encrypted).
            timeout: Request timeout in seconds.
            max_retries: Maximum upload retries.
            tags: Extra metadata tags applied to each chunk.
            bucket_id: Target bucket.

        Returns:
            Result from the underlying ``client.put_chunks`` call.
        """
        chunks.sort()
        put_res = None
        for _ in range(max_retries):
            put_res = await client.put_chunks(
                bucket_id = bucket_id,
                key       = key,
                chunks    = chunks,
                tags      = tags,
                timeout   = timeout,
            )
            if put_res.is_ok:
                return put_res
            await asyncio.sleep(1)
        return put_res

    @staticmethod
    async def get_matrix_chunk_or_error(
        client:AsyncClient,
        ball_id:str, 
        bucket_id:str,
        index:int,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        max_backoff:int = 5
    )->Tuple[npt.NDArray,InterfaceX.Metadata]:
        i =0
        while i <= max_retries :
            x = await client.get_chunk(
                bucket_id         = bucket_id,
                ball_id           = ball_id,
                index             = index,
                max_parallel_gets = max_paralell_gets,
                headers           = headers,
                chunk_size        = chunk_size,
                timeout           = timeout,
                http2             = http2,
                max_retries       = max_retries,
                delay             = delay,
                backoff_factor    = backoff_factor,
                force             = force, 
                max_backoff       = max_backoff
            )
            if x.is_err:
                e = x.unwrap_err()
                print(f"Retrying in {delay} seconds... (Attemp {i}/{max_retries})")
                await asyncio.sleep(delay)
                i+=1
                continue
            (data,metadata) = x.unwrap()
            maybe_ndarray = data.to_ndarray()
            if maybe_ndarray.is_none:
                raise Exception("Failed to convert chunk into a numpy array")
            return (maybe_ndarray.unwrap(), metadata)
            # dtype = metadata.tags.get("dtype","float32")
            # raw_ndarray = np.frombuffer(buffer=data.to,dtype=dtype)
            # shape = eval(metadata.tags.get("shape",str(raw_ndarray.shape)))
            # return raw_ndarray.reshape(shape)
        raise Exception(f"Get {bucket_id}@{ball_id} failed: Max tries reached")
    
    @staticmethod
    async def get_matrix_or_error(
        client:AsyncClient,
        key:str,
        bucket_id:str,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10,
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        chunk_index:int = 0,
    )->npt.NDArray:
        """Download a single plaintext blob and return the reconstructed numpy array.

        Shape and dtype are recovered from the object's metadata tags (set by
        ``put_ndarray``). Raises after ``max_retries`` failed attempts.

        Args:
            client: mictlanx async client.
            key: Object key.
            bucket_id: Source bucket.
            max_retries: Maximum download retries.
            delay: Base retry delay in seconds.
            backoff_factor: Retry backoff multiplier.
            max_paralell_gets: Maximum concurrent chunk downloads.
            force: Pass ``force=True`` to the underlying client.
            timeout: Request timeout in seconds.
            chunk_size: Target size per received chunk.
            headers: Extra HTTP headers.
            http2: Use HTTP/2.
            chunk_index: Starting chunk index.

        Returns:
            Reconstructed ``np.ndarray`` with the original shape and dtype.

        Raises:
            Exception: If all retries are exhausted.
        """
        i =0
        while i <= max_retries :
            x = await client.get(
                bucket_id         = bucket_id,
                key               = key,
                backoff_factor    = backoff_factor,
                max_paralell_gets = max_paralell_gets,
                chunk_size        = chunk_size,
                delay             = delay,
                force             = force, 
                headers           = headers,
                http2             = http2,
                max_retries       = max_retries,
                chunk_index       = chunk_index
            )
            if x.is_err:
                e = x.unwrap_err()
                print(f"Retrying in {delay} seconds... (Attemp {i}/{max_retries})")
                await asyncio.sleep(delay)
                i+=1
                continue
            get_response = x.unwrap()
            dtype = get_response.metadatas[0].tags.get("dtype","float32")
            raw_ndarray = np.frombuffer(buffer=get_response.data.tobytes(),dtype=dtype)
            shape = eval(get_response.metadatas[0].tags.get("shape",str(raw_ndarray.shape)))
            return raw_ndarray.reshape(shape)
        raise Exception(f"Get {bucket_id}@{key} failed: Max tries reached")
    

    @staticmethod
    async def get_and_merge(
        client:AsyncClient,
        key:str,
        bucket_id:str="rory",
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10,
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        chunk_index:int = 0,
    )->npt.NDArray:
        """Download all chunks, verify integrity, merge, and return the ndarray.

        Used after a segmented (plaintext or Liu-encrypted) ``put``. Chunks are
        sorted by index before merging so out-of-order delivery is handled safely.

        Args:
            client: mictlanx async client.
            key: Object key.
            bucket_id: Source bucket.
            max_retries: Maximum download retries.
            delay: Base retry delay in seconds.
            backoff_factor: Retry backoff multiplier.
            max_paralell_gets: Maximum concurrent chunk downloads.
            force: Pass ``force=True`` to the underlying client.
            timeout: Request timeout in seconds.
            chunk_size: Target size per received chunk.
            headers: Extra HTTP headers.
            http2: Use HTTP/2.
            chunk_index: Starting chunk index.

        Returns:
            Merged ``np.ndarray`` reconstructed from all chunks.
        """
        try:
            i= 0
            while i < max_retries:
                x_result = client.get_chunks(
                    bucket_id         = bucket_id,
                    key               = key,
                    timeout           = timeout,
                    max_retries       = max_retries,
                    delay             = delay,
                    backoff_factor    = backoff_factor,
                    force             = force,
                    max_parallel_gets = max_paralell_gets,
                    chunk_size        = chunk_size,
                    headers           = headers,
                    http2             = http2,
                    chunk_index       = chunk_index
                )
                # print("Downloading chunks...",x_result)
                ms:List[InterfaceX.Metadata] = []
                xs:List[Tuple[int, npt.NDArray,bytes]] = []
                h = H.sha256()
                async for (m,data) in x_result:
                    data_bytes = data.tobytes()
                    ms.append(m)
                    shape = eval(m.tags.get("shape"))
                    dtype = m.tags.get("dtype","float64")
                    x = np.frombuffer(data_bytes,dtype= dtype ).reshape(shape)
                    index = int(m.tags.get("index","-1"))
                    xs.append((index,x,data_bytes))
                if len(ms) >0:
                    m = ms[0]
                    num_chunks = int(m.tags.get("num_chunks","-1"))
                    if num_chunks == -1 or num_chunks != len(ms):
                        raise Exception("Faile to get the chunks")
                    else:
                        full_shape  = m.tags.get("full_shape")
                        full_dtype  = m.tags.get("full_dtype","float64")
                        xs_sorted = sorted(xs, key=lambda t: t[0])  # Sort by index value
                        ordered_chunks = []
                        for (i,nd, data) in xs_sorted:
                            h2 = H.sha256()
                            h.update(data)
                            h2.update(data)
                            ordered_chunks.append(nd)
                        checksum = h.hexdigest()
                        return np.concatenate(ordered_chunks, axis=0)
        except Exception as e:
            raise e
    
    @staticmethod
    async def get_and_merge_safe(
             client:AsyncClient,
        key:str,
        bucket_id:str="rory",
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        chunk_index:int = 0,
    ):
        try:
            matrix = await Common.get_and_merge(
                client = client,
                key = key,
                bucket_id = bucket_id,
                max_retries = max_retries,
                delay = delay,
                backoff_factor = backoff_factor,
                max_paralell_gets = max_paralell_gets,
                force = force,
                timeout = timeout,
                chunk_size = chunk_size,
                headers = headers,
                http2 = http2,
                chunk_index = chunk_index
            )
            return Ok(matrix)
        except Exception as e:
            return Err(e) 

  
    @staticmethod
    async def iterate_matrix_chunks(
        client:AsyncClient,
        ball_id:str,
        bucket_id:str="rory",
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        chunk_index:int = 0
    ):

            x_result = client.get_chunks(
                bucket_id         = bucket_id,
                key               = ball_id,
                timeout           = timeout,
                max_retries       = max_retries,
                delay             = delay,
                backoff_factor    = backoff_factor,
                force             = force,
                max_parallel_gets = max_paralell_gets,
                chunk_size        = chunk_size,
                headers           = headers,
                http2             = http2,
                chunk_index       = chunk_index
            )
            async for (m,data) in x_result:
                shape = eval(m.tags.get("shape"))
                dtype = m.tags.get("dtype","float64")
                index = int(m.tags.get("index","-1"))
                # print(shape,dtype,index)
                yield index,m,np.frombuffer(data.tobytes(),dtype=dtype).reshape(shape)

    @staticmethod
    async def get_pyctxt_chunk_or_error(
        client:AsyncClient,
        ckks:Ckks,
        ball_id:str, 
        bucket_id:str,
        index:int,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        max_backoff:int = 5
    )->Tuple[List[PyCtxt],InterfaceX.Metadata]:
        i =0
        while i <= max_retries :
            x = await client.get_chunk(
                bucket_id         = bucket_id,
                ball_id           = ball_id,
                index             = index,
                max_parallel_gets = max_paralell_gets,
                headers           = headers,
                chunk_size        = chunk_size,
                timeout           = timeout,
                http2             = http2,
                max_retries       = max_retries,
                delay             = delay,
                backoff_factor    = backoff_factor,
                force             = force, 
                max_backoff       = max_backoff
            )
            if x.is_err:
                e = x.unwrap_err()
                print(f"Retrying in {delay} seconds... (Attemp {i}/{max_retries})")
                await asyncio.sleep(delay)
                i+=1
                continue
            (data,m) = x.unwrap()
            data_bytes = data.data
            x          = pickle.loads(data_bytes)
            xx         = Common.from_bytes_to_pyctxt_list(ckks=ckks, xs=x)
            return xx,m
    
        raise Exception(f"Get {bucket_id}@{ball_id} failed: Max tries reached")
    
    @staticmethod
    async def get_pyctxt_chunk(
        client:AsyncClient,
        ckks:Ckks,
        ball_id:str, 
        bucket_id:str,
        index:int,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        max_backoff:int = 5
    )->Result[Tuple[List[PyCtxt],InterfaceX.Metadata],Exception]:
        try:
            x = await Common.get_pyctxt_chunk_or_error(
                client = client,
                ckks   = ckks,
                ball_id = ball_id,
                bucket_id = bucket_id,
                index = index,
                max_retries = max_retries,
                delay = delay,
                backoff_factor = backoff_factor,
                max_paralell_gets = max_paralell_gets,
                force = force,
                timeout = timeout,
                chunk_size = chunk_size,
                headers = headers,
                http2 = http2,
                max_backoff = max_backoff
            )
            return Ok(x)
        except Exception as e:
            return Err(e)
        
    
    
    @staticmethod
    async def get_paillier_chunk_or_error(
        client:AsyncClient,
        ball_id:str, 
        bucket_id:str,
        index:int,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        max_backoff:int = 5
    )->Tuple[List[PyCtxt],InterfaceX.Metadata]:
        i =0
        while i <= max_retries :
            x = await client.get_chunk(
                bucket_id         = bucket_id,
                ball_id           = ball_id,
                index             = index,
                max_parallel_gets = max_paralell_gets,
                headers           = headers,
                chunk_size        = chunk_size,
                timeout           = timeout,
                http2             = http2,
                max_retries       = max_retries,
                delay             = delay,
                backoff_factor    = backoff_factor,
                force             = force, 
                max_backoff       = max_backoff
            )
            if x.is_err:
                e = x.unwrap_err()
                print(f"Retrying in {delay} seconds... (Attemp {i}/{max_retries})")
                await asyncio.sleep(delay)
                i+=1
                continue
            (data,m) = x.unwrap()
            data_bytes = data.data
            x          = pickle.loads(data_bytes)
            # xx         = Common.from_bytes_to_pyctxt_list(ckks=ckks, xs=x)
            return x,m
    
        raise Exception(f"Get {bucket_id}@{ball_id} failed: Max tries reached")
    @staticmethod
    async def get_paillier_matrix(
        client:AsyncClient,
        ball_id:str, 
        bucket_id:str,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] ={},
        http2:bool = False,
        chunk_index:int = 0
    ):
        try:
            i= 0
            while i < max_retries:
                x_result = client.get_chunks(
                    bucket_id         = bucket_id,
                    key               = ball_id,
                    timeout           = timeout,
                    max_retries       = max_retries,
                    delay             = delay,
                    backoff_factor    = backoff_factor,
                    force             = force,
                    max_parallel_gets = max_paralell_gets,
                    chunk_size        = chunk_size,
                    headers           = headers,
                    http2             = http2,
                    chunk_index       = chunk_index
                )
                ms:List[InterfaceX.Metadata] = []
                xs:List[Tuple[int, npt.NDArray]] = []
                async for (m,data) in x_result:
                    ms.append(m)
                    data_bytes = data.tobytes()
                    shape      = eval(m.tags.get("shape"))
                    dtype      = m.tags.get("dtype","float64")
                    x          = pickle.loads(data_bytes)
                    index = int(m.tags.get("index","-1"))
                    xs.append((index,x))
                if len(ms) >0:
                    m = ms[0]
                    num_chunks = int(m.tags.get("num_chunks","-1"))
                    if num_chunks == -1 or num_chunks != len(ms):
                        return Err(Exception("Faile to get the chunks"))
                    else:
                        xs_sorted = sorted(xs, key=lambda t: t[0])  # Sort by index value
                        ordered_chunks = []
                        for (i,nd) in xs_sorted:
                            ordered_chunks.append(nd)
                        return Ok(np.concatenate(ordered_chunks, axis=0))
                return Err(Exception("Failed to get chunks. "))
        except Exception as e:
            return Err(e)
        

    @staticmethod
    async def get_pyctxt(
            client:AsyncClient,
            bucket_id:str,
            key:str,
            ckks:Ckks,
            max_retries:int = 5,
            delay:float = 1,
            backoff_factor:float =.5,
            max_paralell_gets:int = 10,
            force:bool = False,
            timeout:int = 120,
            chunk_size:str="256kb",
            headers:Dict[str,str] ={},
            http2:bool = False,
            chunk_index:int = 0
    )-> List[PyCtxt]:
        """Download CKKS ciphertext chunks and return a flat ordered ``List[PyCtxt]``.

        Each chunk is deserialized from pickle bytes and decoded back to
        ``PyCtxt`` using the provided ``Ckks`` context. Chunks are sorted
        by index before flattening.

        Args:
            client: mictlanx async client.
            bucket_id: Source bucket.
            key: Object key.
            ckks: ``Ckks`` context used for ``PyCtxt`` deserialization.
            max_retries: Maximum download retries.
            delay: Base retry delay in seconds.
            backoff_factor: Retry backoff multiplier.
            max_paralell_gets: Maximum concurrent chunk downloads.
            force: Pass ``force=True`` to the underlying client.
            timeout: Request timeout in seconds.
            chunk_size: Target size per received chunk.
            headers: Extra HTTP headers.
            http2: Use HTTP/2.
            chunk_index: Starting chunk index.

        Returns:
            Flat, index-ordered ``List[PyCtxt]``.
        """
        get_chunks_generator = client.get_chunks(
            key               = key,
            bucket_id         = bucket_id,
            max_retries       = max_retries,
            delay             = delay,
            backoff_factor    = backoff_factor,
            max_parallel_gets=  max_paralell_gets,
            force             = force,
            timeout           = timeout, 
            chunk_size        = chunk_size,
            headers           = headers,
            http2             = http2,
            chunk_index       = chunk_index
        )
        xs:List[Tuple[int, List[PyCtxt]]] = []
        async for (m,data) in get_chunks_generator:
            data_bytes = data.tobytes()
            index = int(m.tags.get("index","-1"))
            h = H.sha256()
            h.update(data_bytes)
            x = pickle.loads(data_bytes)
            xx = Common.from_bytes_to_pyctxt_list(ckks=ckks, xs=x)
            xs.append((index, xx))
        xs_sorted = sorted(xs, key=lambda t: t[0])  # Sort by index value
        ordered_xs:List[PyCtxt] = []
        for i in xs_sorted:
            ordered_xs.extend(i[1])
        return ordered_xs


    @staticmethod
    async def get_pyctxt_matrix(
        client:AsyncClient,
        bucket_id:str,
        key:str,
        ckks:Ckks,
        max_retries:int = 5,
        delay:float = 1,
        backoff_factor:float =.5,
        max_paralell_gets:int = 10, 
        force:bool = False,
        timeout:int = 120,
        chunk_size:str="256kb",
        headers:Dict[str,str] = {},
        http2:bool = False, 
        chunk_index:int =0
    ):
        res = client.get_chunks(
            bucket_id         = bucket_id,
            key               = key,
            timeout           = timeout,
            backoff_factor    = backoff_factor,
            chunk_size        = chunk_size,
            delay             = delay,
            force             = force,
            headers           = headers,
            max_retries       = max_retries,
            http2             = http2,
            max_parallel_gets = max_paralell_gets,
            chunk_index       = chunk_index
        )
        xs = []
        async for (m,c) in res:
            x = Common.from_bytes_to_pyctxt_matrix(ckks= ckks, x = c)
            xs.append(x)
        res = np.vstack(xs)
        return res
    def serialize_matrix_with_pickle(enc_matrix:Any)->bytes:
        """Pickle ``enc_matrix`` using the highest available protocol.

        Args:
        enc_matrix(Any): Any pickle-able object (typically an encrypted matrix).

        Returns:
            bytes: Pickle bytes.
        """
        return pickle.dumps(enc_matrix, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize_matrix_with_pickle(serialized_bytes: bytes) -> Any:
        """Unpickle bytes produced by ``serialize_matrix_with_pickle``.

        Args:
            serialized_bytes (bytes): Raw pickle bytes.

        Returns:
            Any: The deserialized object.
        """
        return pickle.loads(serialized_bytes)
    
    @staticmethod
    def segment_and_encrypt_paillier_with_executor(executor:ProcessPoolExecutor,key:str,dataowner:DataOwner,plaintext_matrix:npt.NDArray, n:int, num_chunks:int=2 ):
        plaintext_matrix_chunks:Chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(Common.encrypt_chunk_paillier,key = key, dataowner = dataowner,chunk = plaintext_matrix_chunk)
            awaitable_chunks.append(future)
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )
    
    @staticmethod
    def encrypt_chunk_paillier(key:str,dataowner:DataOwnerPHE,chunk:Chunk)-> Chunk:
        ptm = chunk.to_ndarray().unwrap()
        encyrpted_chunk = Common.serialize_matrix_with_pickle(dataowner.paillier_encrypt_matrix_chunk(plaintext_matrix = ptm))
        # return Chunk.from_bytes()
        # print("HERE!", encyrpted_chunk.shape,encyrpted_chunk.dtype)
        return Chunk.from_bytes(group_id=key, index= chunk.index, data= encyrpted_chunk, chunk_id=Some("{}_{}".format(key,chunk.index)),metadata={
            "shape":str(ptm.shape),
            "dtype":str(ptm.dtype)
        } )
