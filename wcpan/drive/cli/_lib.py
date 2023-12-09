import asyncio
import json
import math
import sys
from typing import Any
from pathlib import Path

from PIL import Image
from wcpan.drive.core.types import MediaInfo, CreateHasher
import yaml


def get_hash(local_path: Path, create_hasher: CreateHasher) -> str:
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


def print_as_yaml(data: Any) -> None:
    yaml.safe_dump(
        data,
        stream=sys.stdout,
        allow_unicode=True,
        encoding=sys.stdout.encoding,
        default_flow_style=False,
    )


def get_image_info(local_path: Path) -> MediaInfo:
    image = Image.open(str(local_path))
    width, height = image.size
    return MediaInfo.image(width=width, height=height)


async def get_video_info(local_path: Path) -> MediaInfo:
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
    cp = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _err = await cp.communicate()
    data = json.loads(out)
    format_ = data["format"]
    ms_duration = math.floor(float(format_["duration"]) * 1000)
    video = data["streams"][0]
    width = video["width"]
    height = video["height"]
    return MediaInfo.video(width=width, height=height, ms_duration=ms_duration)
