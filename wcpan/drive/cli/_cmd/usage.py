from asyncio import as_completed
from argparse import Namespace
from logging import getLogger

from wcpan.drive.core.types import Drive

from .lib import SubCommand, add_bool_argument, for_k_av, get_node_by_id_or_path


def add_usage_command(commands: SubCommand):
    usage_parser = commands.add_parser(
        "usage",
        aliases=["du"],
        help="calculate space usage for files, recursively for folders [offline]",
    )
    usage_parser.add_argument("id_or_path", type=str, nargs="+")
    add_bool_argument(usage_parser, "comma")
    usage_parser.set_defaults(action=_action_usage, comma=False)


async def _action_usage(drive: Drive, args: Namespace) -> int:
    src_list: list[str] = args.id_or_path
    use_comma: bool = args.comma

    rv = 0
    for _ in as_completed(for_k_av(_, _get_usage(drive, _)) for _ in src_list):
        try:
            src, usage = await _
            if use_comma:
                print(f"{usage:,} - {src}")
            else:
                print(f"{usage} - {src}")
        except Exception:
            rv = 1
    return rv


async def _get_usage(drive: Drive, id_or_path: str) -> int:
    try:
        node = await get_node_by_id_or_path(drive, id_or_path)
    except Exception:
        getLogger(__name__).error(f"{id_or_path} does not exist")
        raise

    if not node.is_directory:
        return node.size

    rv = 0
    async for _root, _folders, files in drive.walk(node):
        rv += sum(_.size for _ in files)
    return rv
