from concurrent.futures import Executor
from collections.abc import Iterable
from pathlib import Path
from typing import override

from wcpan.drive.core.types import Node, Drive
from wcpan.drive.core.lib import download_file_to_local

from ._queue import AbstractHandler, walk_list


class DownloadHandler(AbstractHandler[Node, Path]):
    def __init__(self, *, drive: Drive, pool: Executor) -> None:
        super().__init__(drive=drive, pool=pool)

    @override
    async def count_all(self, src: Node) -> int:
        total = 1
        async for _r, d, f in self.drive.walk(src):
            total += len(d) + len(f)
        return total

    @override
    def source_is_trashed(self, src: Node) -> bool:
        return src.is_trashed

    @override
    def source_is_directory(self, src: Node) -> bool:
        return src.is_directory

    @override
    async def do_directory(self, src: Node, dst: Path) -> Path:
        full_path = dst / src.name
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path

    @override
    async def get_children(self, src: Node) -> list[Node]:
        return await self.drive.get_children(src)

    @override
    async def do_file(self, src: Node, dst: Path) -> None:
        local_src = dst / src.name
        if local_src.is_file():
            return

        if local_src.exists():
            raise RuntimeError(f"{local_src} is not a file")

        local_src = await download_file_to_local(self.drive, src, dst)
        local_hash = await self.get_local_file_hash(local_src)
        if local_hash != src.hash:
            raise RuntimeError(f"{dst} checksum mismatch")

    @override
    def format_source(self, src: Node) -> str:
        return src.name


async def download_list(
    srcs: Iterable[Node], dst: Path, *, drive: Drive, pool: Executor, jobs: int
) -> bool:
    handler = DownloadHandler(drive=drive, pool=pool)
    return await walk_list(handler, srcs, dst, jobs=jobs)
