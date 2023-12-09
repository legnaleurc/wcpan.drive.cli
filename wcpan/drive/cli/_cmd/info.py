from argparse import Namespace

from wcpan.drive.core.types import Drive

from .lib import SubCommand, get_node_by_id_or_path
from .._lib import print_as_yaml


def add_info_command(commands: SubCommand):
    info_parser = commands.add_parser(
        "info",
        aliases=["i"],
        help="display file information [offline]",
    )
    info_parser.set_defaults(action=_action_info)
    info_parser.add_argument("id_or_path", type=str)


async def _action_info(drive: Drive, args: Namespace) -> int:
    from dataclasses import asdict

    node = await get_node_by_id_or_path(drive, args.id_or_path)
    print_as_yaml(asdict(node))
    return 0
