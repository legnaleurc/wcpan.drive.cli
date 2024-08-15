import json
import math
import sys
from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image
from wcpan.drive.core.types import MediaInfo, CreateHasher, Drive
import yaml


def _get_hash_off_main(local_path: Path, create_hasher: CreateHasher) -> str:
    from asyncio import run

    CHUNK_SIZE = 64 * 1024

    async def calc():
        hasher = await create_hasher()
        with open(local_path, "rb") as fin:
            while True:
                chunk = fin.read(CHUNK_SIZE)
                if not chunk:
                    break
                await hasher.update(chunk)
        return await hasher.hexdigest()

    return run(calc())


async def get_file_hash(path: Path, /, *, pool: Executor, drive: Drive) -> str:
    from asyncio import get_running_loop

    factory = await drive.get_hasher_factory()
    loop = get_running_loop()
    return await loop.run_in_executor(pool, _get_hash_off_main, path, factory)


def cout(*values: object) -> None:
    print(*values, file=sys.stdout, flush=True)


def cerr(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def print_as_yaml(data: Any) -> None:
    yaml.safe_dump(
        data,
        stream=sys.stdout,
        allow_unicode=True,
        encoding=sys.stdout.encoding,
        default_flow_style=False,
    )


def get_image_info(local_path: Path) -> MediaInfo:
    image = Image.open(str(local_path))  # type: ignore
    width, height = image.size
    return MediaInfo.image(width=width, height=height)


async def get_video_info(local_path: Path) -> MediaInfo:
    from asyncio import create_subprocess_exec
    from asyncio.subprocess import PIPE

    cmd = (
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-select_streams",
        "v:0",
        "-print_format",
        "json",
        "-i",
        str(local_path),
    )
    cp = await create_subprocess_exec(*cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, _err = await cp.communicate()
    data = json.loads(out)
    format_ = data["format"]
    ms_duration = math.floor(float(format_["duration"]) * 1000)
    video = data["streams"][0]
    width = video["width"]
    height = video["height"]
    return MediaInfo.video(width=width, height=height, ms_duration=ms_duration)


def get_mime_type(local_path: Path) -> str:
    import magic

    return magic.from_file(local_path, mime=True)  # type: ignore


async def get_media_info(local_path: Path) -> MediaInfo | None:
    mime_type = get_mime_type(local_path)
    if not mime_type:
        return None

    if mime_type.startswith("image/"):
        return get_image_info(local_path)

    if mime_type.startswith("video/"):
        return await get_video_info(local_path)

    return None


def create_executor() -> Executor:
    from multiprocessing import get_start_method

    if get_start_method() == "spawn":
        return ProcessPoolExecutor(initializer=_initialize_worker)
    else:
        return ProcessPoolExecutor()


def _initialize_worker() -> None:
    from signal import signal, SIG_IGN, SIGINT

    signal(SIGINT, SIG_IGN)
