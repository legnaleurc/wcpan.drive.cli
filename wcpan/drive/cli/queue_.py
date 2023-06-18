import asyncio
from concurrent.futures import Executor
import contextlib
from logging import getLogger
import os
import pathlib
from typing import Generic, TypeVar

from wcpan.drive.core.drive import (
    Drive,
    download_to_local,
    upload_from_local,
)
from wcpan.drive.core.types import Node
from wcpan.queue import AioQueue

from .util import get_hash, get_media_info


SrcT = TypeVar("SrcT")
DstT = TypeVar("DstT")


class AbstractQueue(Generic[SrcT, DstT]):
    def __init__(
        self,
        drive: Drive,
        pool: Executor,
        jobs: int,
    ) -> None:
        self.drive = drive
        self.failed = []
        self._queue = AioQueue.fifo()
        self._consumer_count = jobs
        self._consumers = None
        self._pool = pool
        self._counter = 0
        self._table = {}
        self._total = 0

    async def get_local_file_hash(self, local_path: pathlib.Path) -> str:
        hasher = await self.drive.get_hasher()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool,
            get_hash,
            local_path,
            hasher,
        )

    async def run(
        self,
        src_list: list[SrcT],
        dst: DstT,
    ) -> None:
        if not src_list:
            return
        self._counter = 0
        self._table = {}
        total = (self.count_tasks(_) for _ in src_list)
        total = await asyncio.gather(*total)
        self._total = sum(total)
        for src in src_list:
            await self._queue.push(self._run_one_task(src, dst))
        try:
            await self._queue.consume(self._consumer_count)
        finally:
            self._queue.purge()

    async def count_tasks(self, src: SrcT) -> int:
        raise NotImplementedError()

    def source_is_folder(self, src: SrcT) -> bool:
        raise NotImplementedError()

    async def do_folder(self, src: SrcT, dst: DstT) -> DstT | None:
        raise NotImplementedError()

    async def get_children(self, src: SrcT) -> list[SrcT]:
        raise NotImplementedError()

    async def do_file(self, src: SrcT, dst: DstT) -> DstT | None:
        raise NotImplementedError()

    def get_source_hash(self, src: SrcT) -> str:
        raise NotImplementedError()

    async def get_source_display(self, src: SrcT) -> str:
        raise NotImplementedError()

    async def _run_one_task(self, src: SrcT, dst: DstT) -> DstT | None:
        self._update_counter_table(src)
        async with self._log_guard(src):
            if self.source_is_folder(src):
                rv = await self._run_for_folder(src, dst)
            else:
                rv = await self._run_for_file(src, dst)
        return rv

    async def _run_for_folder(self, src: SrcT, dst: DstT) -> DstT | None:
        try:
            rv = await self.do_folder(src, dst)
        except Exception:
            getLogger(__name__).exception("failed on folder action")
            display = await self.get_source_display(src)
            self._add_failed(display)
            rv = None

        if not rv:
            return None

        children = await self.get_children(src)
        for child in children:
            await self._queue.push(self._run_one_task(child, rv))

        return rv

    async def _run_for_file(self, src: SrcT, dst: DstT) -> DstT | None:
        try:
            rv = await self.do_file(src, dst)
        except Exception:
            getLogger(__name__).exception("failed on file action")
            display = await self.get_source_display(src)
            self._add_failed(display)
            rv = None
        return rv

    def _add_failed(self, src: str) -> None:
        self.failed.append(src)

    @contextlib.asynccontextmanager
    async def _log_guard(self, src: SrcT):
        await self._log("begin", src)
        try:
            yield
        finally:
            await self._log("end", src)

    async def _log(self, begin_or_end: str, src: SrcT) -> None:
        progress = self._get_progress(src)
        display = await self.get_source_display(src)
        getLogger(__name__).info(f"{progress} {begin_or_end} {display}")

    def _get_progress(self, src: SrcT) -> str:
        key = self.get_source_hash(src)
        id_ = self._table[key]
        return f"[{id_}/{self._total}]"

    def _update_counter_table(self, src: SrcT) -> None:
        key = self.get_source_hash(src)
        self._counter += 1
        self._table[key] = self._counter


class UploadQueue(AbstractQueue):
    def __init__(
        self,
        drive: Drive,
        pool: Executor,
        jobs: int,
    ) -> None:
        super(UploadQueue, self).__init__(drive, pool, jobs)

    async def count_tasks(self, local_path: pathlib.Path) -> int:
        total = 1
        for dummy_root, folders, files in os.walk(local_path):
            total = total + len(folders) + len(files)
        return total

    def source_is_folder(self, local_path: pathlib.Path) -> bool:
        return local_path.is_dir()

    async def do_folder(
        self,
        local_path: pathlib.Path,
        parent_node: Node,
    ) -> Node:
        folder_name = local_path.name
        node = await self.drive.create_folder(
            parent_node,
            folder_name,
            exist_ok=True,
        )
        return node

    async def get_children(
        self,
        local_path: pathlib.Path,
    ) -> list[pathlib.Path]:
        rv = os.listdir(local_path)
        rv = [local_path / _ for _ in rv]
        return rv

    async def do_file(
        self,
        local_path: pathlib.Path,
        parent_node: Node,
    ) -> Node:
        media_info = await get_media_info(local_path)
        node = await upload_from_local(
            self.drive,
            parent_node,
            local_path,
            media_info=media_info,
            exist_ok=True,
        )
        local_size = local_path.stat().st_size
        if local_size != node.size:
            raise Exception(f"{local_path} size mismatch")
        local_hash = await self.get_local_file_hash(local_path)
        if local_hash != node.hash_:
            raise Exception(f"{local_path} checksum mismatch")
        return node

    def get_source_hash(self, local_path: pathlib.Path) -> str:
        return str(local_path)

    async def get_source_display(self, local_path: pathlib.Path) -> str:
        return str(local_path)


class DownloadQueue(AbstractQueue):
    def __init__(
        self,
        drive: Drive,
        pool: Executor,
        jobs: int,
    ) -> None:
        super(DownloadQueue, self).__init__(drive, pool, jobs)

    async def count_tasks(self, node: Node) -> int:
        total = 1
        children = await self.drive.get_children(node)
        count = (self.count_tasks(_) for _ in children)
        count = await asyncio.gather(*count)
        return total + sum(count)

    def source_is_folder(self, node: Node) -> bool:
        return node.is_folder

    async def do_folder(
        self,
        node: Node,
        local_path: pathlib.Path,
    ) -> pathlib.Path:
        full_path = local_path / node.name
        os.makedirs(full_path, exist_ok=True)
        return full_path

    async def get_children(self, node: Node) -> list[Node]:
        return await self.drive.get_children(node)

    async def do_file(
        self,
        node: Node,
        local_path: pathlib.Path,
    ) -> pathlib.Path:
        local_path = await download_to_local(self.drive, node, local_path)
        local_hash = await self.get_local_file_hash(local_path)
        if local_hash != node.hash_:
            raise Exception(f"{local_path} checksum mismatch")
        return local_path

    def get_source_hash(self, node: Node) -> str:
        return node.id_

    async def get_source_display(self, node: Node) -> str:
        return await self.drive.get_path(node)


class VerifyQueue(AbstractQueue):
    def __init__(
        self,
        drive: Drive,
        pool: Executor,
        jobs: int,
    ) -> None:
        super(VerifyQueue, self).__init__(drive, pool, jobs)

    async def count_tasks(self, local_path: pathlib.Path) -> int:
        total = 1
        for dummy_root, folders, files in os.walk(local_path):
            total = total + len(folders) + len(files)
        return total

    def source_is_folder(self, local_path: pathlib.Path) -> bool:
        return local_path.is_dir()

    async def do_folder(
        self,
        local_path: pathlib.Path,
        parent_node: Node,
    ) -> Node | None:
        dir_name = local_path.name

        node = await self._get_child_node(
            local_path,
            dir_name,
            parent_node,
        )
        if not node:
            return None
        if not node.is_folder:
            getLogger(__name__).error(f"[NOT_FOLDER] {local_path}")
            return None

        getLogger(__name__).info(f"[OK] {local_path}")
        return node

    async def get_children(
        self,
        local_path: pathlib.Path,
    ) -> list[pathlib.Path]:
        rv = os.listdir(local_path)
        rv = [local_path / _ for _ in rv]
        return rv

    async def do_file(
        self,
        local_path: pathlib.Path,
        parent_node: Node,
    ) -> Node | None:
        file_name = local_path.name

        node = await self._get_child_node(
            local_path,
            file_name,
            parent_node,
        )
        if not node:
            return None
        if not node.is_file:
            getLogger(__name__).error(f"[NOT_FILE] {local_path}")
            return None

        local_hash = await self.get_local_file_hash(local_path)
        if local_hash != node.hash_:
            getLogger(__name__).error(f"[WRONG_HASH] {local_path}")
            return None

        getLogger(__name__).info(f"[OK] {local_path}")
        return node

    def get_source_hash(self, local_path: pathlib.Path) -> str:
        return str(local_path)

    async def get_source_display(self, local_path: pathlib.Path) -> str:
        return str(local_path)

    async def _get_child_node(
        self,
        local_path: pathlib.Path,
        name: str,
        parent_node: Node,
    ) -> Node | None:
        child_node = await self.drive.get_node_by_name_from_parent(
            name,
            parent_node,
        )
        if not child_node:
            getLogger(__name__).error(f"[MISSING] {local_path}")
            return None
        if child_node.trashed:
            getLogger(__name__).error(f"[TRASHED] {local_path}")
            return None
        return child_node
