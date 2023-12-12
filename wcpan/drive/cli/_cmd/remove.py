from asyncio import as_completed
from argparse import Namespace

from wcpan.drive.core.types import Drive

from .lib import (
    SubCommand,
    add_bool_argument,
    require_authorized,
    get_node_by_id_or_path,
)
from .._lib import cerr


def add_remove_command(commands: SubCommand):
    parser = commands.add_parser(
        "remove",
        aliases=["rm"],
        help="trash files/folders",
    )
    add_bool_argument(parser, "restore")
    parser.set_defaults(action=_action_remove, restore=False)
    parser.add_argument("id_or_path", type=str, nargs="+")


@require_authorized
async def _action_remove(drive: Drive, kwargs: Namespace) -> int:
    id_or_path: str = kwargs.id_or_path
    restore: bool = kwargs.restore
    rv = 0
    for _ in as_completed(_trash_node(drive, _, not restore) for _ in id_or_path):
        try:
            await _
        except Exception:
            rv = 1
    return rv


async def _trash_node(drive: Drive, id_or_path: str, trashed: bool) -> None:
    try:
        node = await get_node_by_id_or_path(drive, id_or_path)
    except Exception:
        cerr(f"{id_or_path} does not exist")
        raise

    try:
        await drive.move(node, trashed=trashed)
    except Exception as e:
        cerr(f"operation failed on {id_or_path}, reason: {str(e)}")
        raise
