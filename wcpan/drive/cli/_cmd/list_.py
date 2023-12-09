from argparse import Namespace

from wcpan.drive.core.types import Drive

from .lib import SubCommand, get_node_by_id_or_path


def add_list_command(commands: SubCommand):
    list_parser = commands.add_parser(
        "list",
        aliases=["ls"],
        help="list folder [offline]",
    )
    list_parser.set_defaults(action=_action_list)
    list_parser.add_argument("id_or_path", type=str)


async def _action_list(drive: Drive, args: Namespace) -> int:
    node = await get_node_by_id_or_path(drive, args.id_or_path)
    nodes = await drive.get_children(node)
    for node in nodes:
        print(f"{node.id}: {node.name}")
    return 0
