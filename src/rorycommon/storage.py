from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import numpy.typing as npt
from mictlanx.utils.segmentation import Chunk, Chunks
from option import Some


class FileSystemStorage:
    """Durable local storage using bucket and ball directories."""

    FORMAT_VERSION = 1
    MANIFEST_NAME = "manifest.json"

    def __init__(self, root_path: str | os.PathLike[str] = "/rory/data"):
        self.root_path = Path(root_path).expanduser().resolve()

    @staticmethod
    async def _run_sync(function, *args):
        return function(*args)

    @staticmethod
    def _validate_component(value: str, label: str) -> str:
        if not isinstance(value, str) or not value or value in {".", ".."}:
            raise ValueError(f"{label} must be a non-empty path component")
        if value == ".locks" or Path(value).name != value:
            raise ValueError(f"Invalid {label}: {value!r}")
        if os.sep in value or (os.altsep and os.altsep in value):
            raise ValueError(f"Invalid {label}: {value!r}")
        return value

    def object_path(self, bucket_id: str, ball_id: str) -> Path:
        bucket = self._validate_component(bucket_id, "bucket_id")
        ball = self._validate_component(ball_id, "ball_id")
        candidate = (self.root_path / bucket / ball).resolve(strict=False)
        try:
            candidate.relative_to(self.root_path)
        except ValueError as error:
            raise ValueError("Storage object escapes the configured root") from error
        return candidate

    def _lock_path(self, bucket_id: str, ball_id: str) -> Path:
        bucket = self._validate_component(bucket_id, "bucket_id")
        ball = self._validate_component(ball_id, "ball_id")
        digest = hashlib.sha256(ball.encode("utf-8")).hexdigest()
        candidate = (self.root_path / bucket / ".locks" / f"{digest}.lock").resolve(
            strict=False
        )
        try:
            candidate.relative_to(self.root_path)
        except ValueError as error:
            raise ValueError("Storage lock escapes the configured root") from error
        return candidate

    @contextmanager
    def _lock(self, bucket_id: str, ball_id: str, *, exclusive: bool) -> Iterator[None]:
        lock_path = self._lock_path(bucket_id, ball_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), operation)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _checksum(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _write_bytes(path: Path, data: bytes) -> None:
        with path.open("wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _commit(
        self,
        bucket_id: str,
        ball_id: str,
        manifest: Dict[str, Any],
        payload_writer,
    ) -> Path:
        object_path = self.object_path(bucket_id, ball_id)
        with self._lock(bucket_id, ball_id, exclusive=True):
            generations_path = object_path / "generations"
            generation = uuid.uuid4().hex
            generation_path = generations_path / generation
            generation_path.mkdir(parents=True, exist_ok=False)
            manifest_tmp = object_path / f".{self.MANIFEST_NAME}.{generation}.tmp"
            committed = False
            try:
                payload_writer(generation_path)
                manifest = {
                    "format_version": self.FORMAT_VERSION,
                    "bucket_id": bucket_id,
                    "ball_id": ball_id,
                    "generation": generation,
                    **manifest,
                }
                encoded = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
                self._write_bytes(manifest_tmp, encoded)
                os.replace(manifest_tmp, object_path / self.MANIFEST_NAME)
                committed = True
                self._fsync_directory(object_path)

                for candidate in generations_path.iterdir():
                    if candidate.name != generation:
                        shutil.rmtree(candidate, ignore_errors=True)
                return object_path
            except Exception:
                manifest_tmp.unlink(missing_ok=True)
                if not committed:
                    shutil.rmtree(generation_path, ignore_errors=True)
                raise

    @staticmethod
    def _string_dict(values: Optional[Dict[str, Any]]) -> Dict[str, str]:
        return {str(key): str(value) for key, value in (values or {}).items()}

    def _put_ndarray_sync(
        self,
        bucket_id: str,
        ball_id: str,
        array: npt.NDArray,
        tags: Dict[str, str],
        segment: bool,
        encrypt: bool,
    ) -> Path:
        data = array.tobytes(order="C")

        def write_payload(generation_path: Path) -> None:
            self._write_bytes(generation_path / "payload.bin", data)

        return self._commit(
            bucket_id,
            ball_id,
            {
                "kind": "ndarray",
                "segment": bool(segment),
                "encrypt": bool(encrypt),
                "tags": self._string_dict(tags),
                "payload": {
                    "file": "payload.bin",
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "checksum": self._checksum(data),
                },
            },
            write_payload,
        )

    async def put_ndarray(
        self,
        bucket_id: str,
        ball_id: str,
        array: npt.NDArray,
        tags: Optional[Dict[str, str]] = None,
        *,
        segment: bool = False,
        encrypt: bool = False,
    ) -> Path:
        return await self._run_sync(
            self._put_ndarray_sync,
            bucket_id,
            ball_id,
            array,
            tags or {},
            segment,
            encrypt,
        )

    def _put_chunks_sync(
        self,
        bucket_id: str,
        ball_id: str,
        chunks: Chunks,
        tags: Dict[str, str],
        segment: bool,
        encrypt: bool,
    ) -> Path:
        ordered = sorted(chunks.iter(), key=lambda chunk: chunk.index)
        indexes = [chunk.index for chunk in ordered]
        if len(indexes) != len(set(indexes)):
            raise ValueError("Chunk indexes must be unique")

        entries = []
        for chunk in ordered:
            filename = f"{chunk.index:08d}.bin"
            entries.append(
                {
                    "index": chunk.index,
                    "chunk_id": chunk.chunk_id,
                    "file": filename,
                    "size": len(chunk.data),
                    "checksum": self._checksum(chunk.data),
                    "metadata": self._string_dict(chunk.metadata),
                }
            )

        def write_payload(generation_path: Path) -> None:
            chunks_path = generation_path / "chunks"
            chunks_path.mkdir()
            for chunk, entry in zip(ordered, entries):
                self._write_bytes(chunks_path / entry["file"], chunk.data)

        return self._commit(
            bucket_id,
            ball_id,
            {
                "kind": "chunks",
                "segment": bool(segment),
                "encrypt": bool(encrypt),
                "tags": self._string_dict(tags),
                "n": int(chunks.n),
                "chunks": entries,
            },
            write_payload,
        )

    async def put_chunks(
        self,
        bucket_id: str,
        ball_id: str,
        chunks: Chunks,
        tags: Optional[Dict[str, str]] = None,
        *,
        segment: bool,
        encrypt: bool,
    ) -> Path:
        return await self._run_sync(
            self._put_chunks_sync,
            bucket_id,
            ball_id,
            chunks,
            tags or {},
            segment,
            encrypt,
        )

    def _read_manifest(self, object_path: Path) -> Dict[str, Any]:
        manifest_path = object_path / self.MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Stored object not found: {object_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("Unsupported filesystem storage format")
        return manifest

    @staticmethod
    def _validate_flags(manifest: Dict[str, Any], segment: bool, encrypt: bool) -> None:
        stored = (bool(manifest.get("segment")), bool(manifest.get("encrypt")))
        requested = (bool(segment), bool(encrypt))
        if stored != requested:
            raise ValueError(
                "Storage flags do not match the stored object: "
                f"stored segment={stored[0]}, encrypt={stored[1]}"
            )

    def _get_ndarray_sync(
        self,
        bucket_id: str,
        ball_id: str,
        segment: bool,
        encrypt: bool,
    ) -> npt.NDArray:
        object_path = self.object_path(bucket_id, ball_id)
        with self._lock(bucket_id, ball_id, exclusive=False):
            manifest = self._read_manifest(object_path)
            self._validate_flags(manifest, segment, encrypt)
            if manifest.get("kind") != "ndarray":
                raise ValueError("Stored object is chunked")
            payload = manifest["payload"]
            path = object_path / "generations" / manifest["generation"] / payload["file"]
            data = path.read_bytes()
            if self._checksum(data) != payload["checksum"]:
                raise ValueError(f"Checksum mismatch for {path}")
            array = np.frombuffer(data, dtype=payload["dtype"])
            return array.reshape(tuple(payload["shape"])).copy()

    async def get_ndarray(
        self,
        bucket_id: str,
        ball_id: str,
        *,
        segment: bool = False,
        encrypt: bool = False,
    ) -> npt.NDArray:
        return await self._run_sync(
            self._get_ndarray_sync,
            bucket_id,
            ball_id,
            segment,
            encrypt,
        )

    def _get_chunks_sync(
        self,
        bucket_id: str,
        ball_id: str,
        segment: bool,
        encrypt: bool,
    ) -> Chunks:
        object_path = self.object_path(bucket_id, ball_id)
        with self._lock(bucket_id, ball_id, exclusive=False):
            manifest = self._read_manifest(object_path)
            self._validate_flags(manifest, segment, encrypt)
            if manifest.get("kind") != "chunks":
                raise ValueError("Stored object is not chunked")
            generation_path = object_path / "generations" / manifest["generation"] / "chunks"
            restored = []
            for entry in sorted(manifest["chunks"], key=lambda value: value["index"]):
                path = generation_path / entry["file"]
                data = path.read_bytes()
                if len(data) != entry["size"] or self._checksum(data) != entry["checksum"]:
                    raise ValueError(f"Checksum mismatch for {path}")
                restored.append(
                    Chunk(
                        group_id=ball_id,
                        index=int(entry["index"]),
                        data=data,
                        chunk_id=Some(entry["chunk_id"]),
                        metadata=self._string_dict(entry.get("metadata")),
                    )
                )
            return Chunks(chs=restored, n=int(manifest.get("n", 0)))

    async def get_chunks(
        self,
        bucket_id: str,
        ball_id: str,
        *,
        segment: bool,
        encrypt: bool,
    ) -> Chunks:
        return await self._run_sync(
            self._get_chunks_sync,
            bucket_id,
            ball_id,
            segment,
            encrypt,
        )

    def _get_object_sync(
        self,
        bucket_id: str,
        ball_id: str,
        segment: bool,
        encrypt: bool,
    ):
        object_path = self.object_path(bucket_id, ball_id)
        with self._lock(bucket_id, ball_id, exclusive=False):
            manifest = self._read_manifest(object_path)
            self._validate_flags(manifest, segment, encrypt)
            kind = str(manifest["kind"])
            if kind == "ndarray":
                payload = manifest["payload"]
                path = object_path / "generations" / manifest["generation"] / payload["file"]
                data = path.read_bytes()
                if self._checksum(data) != payload["checksum"]:
                    raise ValueError(f"Checksum mismatch for {path}")
                value = np.frombuffer(data, dtype=payload["dtype"])
                return kind, value.reshape(tuple(payload["shape"])).copy()
            if kind != "chunks":
                raise ValueError(f"Unsupported stored object kind: {kind}")

            generation_path = object_path / "generations" / manifest["generation"] / "chunks"
            restored = []
            for entry in sorted(manifest["chunks"], key=lambda value: value["index"]):
                path = generation_path / entry["file"]
                data = path.read_bytes()
                if len(data) != entry["size"] or self._checksum(data) != entry["checksum"]:
                    raise ValueError(f"Checksum mismatch for {path}")
                restored.append(
                    Chunk(
                        group_id=ball_id,
                        index=int(entry["index"]),
                        data=data,
                        chunk_id=Some(entry["chunk_id"]),
                        metadata=self._string_dict(entry.get("metadata")),
                    )
                )
            return kind, Chunks(chs=restored, n=int(manifest.get("n", 0)))

    async def get_object(
        self,
        bucket_id: str,
        ball_id: str,
        *,
        segment: bool,
        encrypt: bool,
    ):
        return await self._run_sync(
            self._get_object_sync,
            bucket_id,
            ball_id,
            segment,
            encrypt,
        )

    def _delete_sync(self, bucket_id: str, ball_id: str) -> int:
        object_path = self.object_path(bucket_id, ball_id)
        with self._lock(bucket_id, ball_id, exclusive=True):
            if not object_path.exists():
                return 0
            shutil.rmtree(object_path)
            return 1

    async def delete(self, bucket_id: str, ball_id: str) -> int:
        return await self._run_sync(self._delete_sync, bucket_id, ball_id)
