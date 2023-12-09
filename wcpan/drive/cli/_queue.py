from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from concurrent.futures import Executor
from contextlib import contextmanager
from logging import getLogger, Logger
from pathlib import Path

from wcpan.drive.core.types import Drive
from wcpan.queue import AioQueue

from ._lib import get_hash


class AbstractHandler[S, D](metaclass=ABCMeta):
    def __init__(self, *, drive: Drive, pool: Executor) -> None:
        self.drive = drive
        self._pool = pool

    async def get_local_file_hash(self, local_path: Path) -> str:
        from asyncio import get_running_loop

        create_hasher = await self.drive.get_hasher_factory()
        loop = get_running_loop()
        return await loop.run_in_executor(
            self._pool, get_hash, local_path, create_hasher
        )

    @abstractmethod
    async def count_all(self, src: S) -> int:
        raise NotImplementedError

    @abstractmethod
    def source_is_trashed(self, src: S) -> bool:
        raise NotImplementedError

    @abstractmethod
    def source_is_directory(self, src: S) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def do_directory(self, src: S, dst: D) -> D:
        raise NotImplementedError

    @abstractmethod
    async def get_children(self, src: S) -> list[S]:
        raise NotImplementedError

    @abstractmethod
    async def do_file(self, src: S, dst: D) -> None:
        raise NotImplementedError

    @abstractmethod
    def format_source(self, src: S) -> str:
        raise NotImplementedError


class ProgressTracker:
    def __init__(self, total: int) -> None:
        self._total = total
        self._now = 0
        self._error = 0

    @contextmanager
    def collect(self, name: str):
        _L().info(f"[{self._now}/{self._total}] [B] {name}")
        try:
            yield
        except Exception:
            self._error += 1
            raise
        finally:
            self._now += 1
            _L().info(f"[{self._now}/{self._total}] [E] {name}")

    @property
    def has_error(self) -> bool:
        return self._error > 0


async def walk_list[
    S, D
](handler: AbstractHandler[S, D], srcs: Iterable[S], dst: D, *, jobs: int) -> bool:
    from asyncio import as_completed

    total = 0
    for _ in as_completed(handler.count_all(_) for _ in srcs):
        total += await _
    tracker = ProgressTracker(total)

    with AioQueue[None].fifo() as queue:
        for src in srcs:
            await queue.push(
                _walk_unknown(src, dst, queue=queue, handler=handler, tracker=tracker)
            )
        await queue.consume(jobs)

    return tracker.has_error


async def _walk_unknown[
    S, D
](
    src: S,
    dst: D,
    *,
    queue: AioQueue[None],
    handler: AbstractHandler[S, D],
    tracker: ProgressTracker,
) -> None:
    if handler.source_is_trashed(src):
        return
    if handler.source_is_directory(src):
        await queue.push(
            _walk_directory(src, dst, queue=queue, handler=handler, tracker=tracker)
        )
    else:
        await queue.push(_walk_file(src, dst, handler=handler, tracker=tracker))


async def _walk_directory[
    S, D
](
    src: S,
    dst: D,
    *,
    queue: AioQueue[None],
    handler: AbstractHandler[S, D],
    tracker: ProgressTracker,
) -> None:
    with tracker.collect(handler.format_source(src)):
        try:
            new_directory = await handler.do_directory(src, dst)
            children = await handler.get_children(src)
        except Exception:
            _L().exception("directory failed")
            return

    for child in children:
        await queue.push(
            _walk_unknown(
                child, new_directory, queue=queue, handler=handler, tracker=tracker
            )
        )


async def _walk_file[
    S, D
](src: S, dst: D, *, handler: AbstractHandler[S, D], tracker: ProgressTracker) -> None:
    with tracker.collect(handler.format_source(src)):
        try:
            await handler.do_file(src, dst)
        except Exception:
            _L().exception("file failed")
            return


def _L() -> Logger:
    return getLogger(__name__)
