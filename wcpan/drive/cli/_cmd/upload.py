from argparse import Namespace
from pathlib import Path

from wcpan.drive.core.types import Drive

from .lib import SubCommand, require_authorized, get_node_by_id_or_path, create_executor
from .._upload import upload_list


def add_upload_command(commands: SubCommand):
    ul_parser = commands.add_parser(
        "upload",
        aliases=["ul"],
        help="upload files/folders",
    )
    ul_parser.set_defaults(action=_action_upload)
    ul_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="maximum simultaneously upload jobs (default: %(default)s)",
    )
    ul_parser.add_argument("source", type=str, nargs="+")
    ul_parser.add_argument("id_or_path", type=str)


@require_authorized
async def _action_upload(drive: Drive, args: Namespace) -> int:
    with create_executor() as pool:
        node = await get_node_by_id_or_path(drive, args.id_or_path)
        src_list = [Path(_) for _ in args.source]

        ok = await upload_list(src_list, node, drive=drive, pool=pool, jobs=args.jobs)

    return 0 if ok else 1
