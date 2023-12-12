from argparse import Namespace

from wcpan.drive.core.types import Drive

from .lib import SubCommand, add_bool_argument, require_authorized, add_help_message
from .._lib import print_as_yaml, cout


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

    usage_parser = commands.add_parser("usage", help="trash size usage [offline]")
    add_bool_argument(usage_parser, "comma")
    usage_parser.set_defaults(
        action=_action_trash_usage,
        comma=True,
    )

    purge_parser = commands.add_parser("purge", help="purge trash")
    add_bool_argument(purge_parser, "ask", short_false="y")
    purge_parser.set_defaults(
        action=_action_trash_purge,
        ask=True,
    )

    add_help_message(parser)


async def _action_trash_list(drive: Drive, kwargs: Namespace) -> int:
    flatten: bool = kwargs.flatten

    node_list = await drive.get_trashed_nodes(flatten)
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


async def _action_trash_usage(drive: Drive, kwargs: Namespace) -> int:
    comma: bool = kwargs.comma

    node_list = await drive.get_trashed_nodes()
    rv = sum(_.size for _ in node_list)
    if comma:
        cout(f"{rv:,}")
    else:
        cout(f"{rv}")
    return 0


@require_authorized
async def _action_trash_purge(drive: Drive, kwargs: Namespace) -> int:
    ask: bool = kwargs.ask

    node_list = await drive.get_trashed_nodes()
    count = len(node_list)
    cout(f"Purging {count} items in trash ...")

    if ask:
        answer = input("Are you sure? [y/N]")
        answer = answer.lower()
        if answer != "y":
            cout("Aborted.")
            return 0

    try:
        await drive.purge_trash()
    except Exception as e:
        cout(str(e))
        return 1

    cout("Done.")
    return 0
