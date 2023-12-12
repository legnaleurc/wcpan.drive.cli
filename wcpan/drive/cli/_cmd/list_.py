from argparse import Namespace

from wcpan.drive.core.types import Drive

from .lib import SubCommand, get_node_by_id_or_path
from .._lib import cout


def add_list_command(commands: SubCommand):
    parser = commands.add_parser(
        "list",
        aliases=["ls"],
        help="list folder [offline]",
    )
    parser.set_defaults(action=_action_list)
    parser.add_argument("id_or_path", type=str)


async def _action_list(drive: Drive, kwargs: Namespace) -> int:
    id_or_path: str = kwargs.id_or_path

    node = await get_node_by_id_or_path(drive, id_or_path)
    nodes = await drive.get_children(node)
    for node in nodes:
        cout(f"{node.id}: {node.name}")
    return 0
