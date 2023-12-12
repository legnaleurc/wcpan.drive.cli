from asyncio import as_completed
from argparse import Namespace
from pathlib import Path

from wcpan.drive.core.types import Drive

from .lib import SubCommand, require_authorized, get_node_by_id_or_path, create_executor
from .._download import download_list


def add_download_command(commands: SubCommand):
    parser = commands.add_parser(
        "download",
        aliases=["dl"],
        help="download files/folders",
    )
    parser.set_defaults(action=_action_download)
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="maximum simultaneously download jobs (default: %(default)s)",
    )
    parser.add_argument("id_or_path", type=str, nargs="+")
    parser.add_argument("destination", type=str)


@require_authorized
async def _action_download(drive: Drive, kwargs: Namespace) -> int:
    id_or_path: list[str] = kwargs.id_or_path
    destination: str = kwargs.destination

    with create_executor() as pool:
        g = (get_node_by_id_or_path(drive, _) for _ in id_or_path)
        ag = (await _ for _ in as_completed(g))
        node_list = [_ async for _ in ag if not _.is_trashed]
        dst = Path(destination)

        ok = await download_list(
            node_list, dst, drive=drive, pool=pool, jobs=kwargs.jobs
        )

    return 0 if ok else 1
