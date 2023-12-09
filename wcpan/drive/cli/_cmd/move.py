from asyncio import as_completed
from argparse import Namespace
from logging import getLogger
from pathlib import PurePath

from wcpan.drive.core.types import Drive
from wcpan.drive.core.lib import move_node

from .lib import SubCommand, require_authorized, get_path_by_id_or_path


def add_move_command(commands: SubCommand):
    mv_parser = commands.add_parser(
        "rename",
        aliases=["mv"],
        help="rename file/folder",
    )
    mv_parser.set_defaults(action=_action_rename)
    mv_parser.add_argument("source_id_or_path", type=str, nargs="+")
    mv_parser.add_argument("destination_path", type=str)


@require_authorized
async def _action_rename(drive: Drive, args: Namespace) -> int:
    src_list: list[str] = args.source_id_or_path
    dst_path = PurePath(args.destination_path)
    rv = 0
    for _ in as_completed(_rename_node(drive, _, dst_path) for _ in src_list):
        try:
            await _
        except Exception:
            rv = 1
    return rv


async def _rename_node(drive: Drive, id_or_path: str, dst: PurePath) -> None:
    try:
        path = await get_path_by_id_or_path(drive, id_or_path)
    except Exception:
        getLogger(__name__).error(f"{id_or_path} does not exist")
        raise

    try:
        await move_node(drive, path, dst)
    except Exception as e:
        getLogger(__name__).error(f"failed to move {id_or_path}, reason: {str(e)}")
        raise
