from asyncio import as_completed
from argparse import Namespace
from pathlib import Path

from wcpan.drive.core.types import Drive

from .lib import SubCommand, require_authorized, get_node_by_id_or_path, create_executor
from .._download import download_list


def add_download_command(commands: SubCommand):
    dl_parser = commands.add_parser(
        "download",
        aliases=["dl"],
        help="download files/folders",
    )
    dl_parser.set_defaults(action=_action_download)
    dl_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="maximum simultaneously download jobs (default: %(default)s)",
    )
    dl_parser.add_argument("id_or_path", type=str, nargs="+")
    dl_parser.add_argument("destination", type=str)


@require_authorized
async def _action_download(drive: Drive, args: Namespace) -> int:
    with create_executor() as pool:
        g = (get_node_by_id_or_path(drive, _) for _ in args.id_or_path)
        ag = (await _ for _ in as_completed(g))
        node_list = [_ async for _ in ag if not _.is_trashed]
        dst = Path(args.destination)

        ok = await download_list(node_list, dst, drive=drive, pool=pool, jobs=args.jobs)

    return 0 if ok else 1
