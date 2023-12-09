from argparse import Namespace

from wcpan.drive.core.types import Drive

from .lib import SubCommand, get_node_by_id_or_path
from .._interaction import interact


def add_shell_command(commands: SubCommand):
    shell_parser = commands.add_parser(
        "shell",
        help="start an interactive shell",
    )
    shell_parser.set_defaults(action=_action_shell)
    shell_parser.add_argument("id_or_path", type=str, nargs="?")


async def _action_shell(drive: Drive, args: Namespace) -> int:
    if not args.id_or_path:
        node = await drive.get_root()
    else:
        node = await get_node_by_id_or_path(drive, args.id_or_path)

    if not node or not node.is_directory:
        print(f"{args.id_or_path} is not a folder")
        return 1

    interact(drive, node)
    return 0
