from argparse import Namespace

from wcpan.drive.core.types import Drive

from .lib import SubCommand, add_bool_argument, require_authorized, add_help_message
from .._lib import print_as_yaml


def add_trash_command(commands: SubCommand):
    parser = commands.add_parser(
        "trash",
        help="actions for trashes",
    )
    commands = parser.add_subparsers()
    list_parser = commands.add_parser("list", help="list trash [offline]")
    add_bool_argument(list_parser, "flatten")
    list_parser.set_defaults(
        action=_action_trash_list,
        flatten=False,
    )
    purge_parser = commands.add_parser("purge", help="purge trash")
    add_bool_argument(purge_parser, "ask", short_false="y")
    purge_parser.set_defaults(
        action=_action_trash_purge,
        ask=True,
    )

    add_help_message(parser)


async def _action_trash_list(drive: Drive, args: Namespace) -> int:
    node_list = await drive.get_trashed_nodes(args.flatten)
    rv = [
        {
            "id": _.id,
            "name": _.name,
            "modified": str(_.ctime),
        }
        for _ in node_list
    ]
    print_as_yaml(rv)
    return 0


@require_authorized
async def _action_trash_purge(drive: Drive, args: Namespace) -> int:
    ask: bool = args.ask

    node_list = await drive.get_trashed_nodes()
    count = len(node_list)
    print(f"Purging {count} items in trash ...")

    if ask:
        answer = input("Are you sure? [y/N]")
        answer = answer.lower()
        if answer != "y":
            print("Aborted.")
            return 0

    try:
        await drive.purge_trash()
    except Exception as e:
        print(str(e))
        return 1

    print("Done.")
    return 0
