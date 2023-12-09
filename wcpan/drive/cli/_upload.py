from concurrent.futures import Executor
from collections.abc import Iterable, Awaitable
from pathlib import Path
from typing import override

from wcpan.drive.core.types import Node, Drive
from wcpan.drive.core.lib import upload_file_from_local

from ._queue import AbstractHandler, walk_list
from .lib import get_media_info


class UploadHandler(AbstractHandler[Path, Node]):
    def __init__(self, *, drive: Drive, pool: Executor) -> None:
        super().__init__(drive=drive, pool=pool)

    @override
    async def count_all(self, src: Path) -> int:
        total = 1
        for _root, folders, files in src.walk():
            total = total + len(folders) + len(files)
        return total

    @override
    def source_is_trashed(self, src: Path) -> bool:
        return False

    @override
    def source_is_directory(self, src: Path) -> bool:
        return src.is_dir()

    @override
    async def do_directory(self, src: Path, dst: Node) -> Node:
        folder_name = src.name
        node = await self.drive.create_directory(folder_name, dst, exist_ok=True)
        return node

    @override
    async def get_children(self, src: Path) -> list[Path]:
        return [_ for _ in src.iterdir()]

    @override
    async def do_file(self, src: Path, dst: Node) -> None:
        node = await _else_none(self.drive.get_child_by_name(src.name, dst))
        if not node:
            media_info = await get_media_info(src)
            node = await upload_file_from_local(
                self.drive, src, dst, media_info=media_info
            )
            if not node:
                raise Exception("upload failed")

        local_size = src.stat().st_size
        if local_size != node.size:
            raise Exception(f"{src} size mismatch")
        local_hash = await self.get_local_file_hash(src)
        if local_hash != node.hash:
            raise Exception(f"{src} checksum mismatch")

    @override
    def format_source(self, src: Path) -> str:
        return src.name


async def upload_list(
    srcs: Iterable[Path], dst: Node, *, drive: Drive, pool: Executor, jobs: int
) -> bool:
    handler = UploadHandler(drive=drive, pool=pool)
    return await walk_list(handler, srcs, dst, jobs=jobs)


async def _else_none(aw: Awaitable[Node]) -> Node | None:
    try:
        return await aw
    except Exception:
        return None
