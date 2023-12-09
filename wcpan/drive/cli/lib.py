import mimetypes
from pathlib import Path

from wcpan.drive.core.types import MediaInfo

from ._lib import get_image_info, get_video_info
from ._cfg import create_drive_from_config as create_drive_from_config


__all__ = ("get_media_info", "create_drive_from_config")


async def get_media_info(local_path: Path) -> MediaInfo | None:
    type_, _ext = mimetypes.guess_type(local_path)
    if not type_:
        return None

    if type_.startswith("image/"):
        return get_image_info(local_path)

    if type_.startswith("video/"):
        return await get_video_info(local_path)

    return None
