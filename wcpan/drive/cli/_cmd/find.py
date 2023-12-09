from asyncio import as_completed
from argparse import Namespace

from wcpan.drive.core.types import Drive

from .lib import SubCommand, add_bool_argument, for_k_av


def add_find_command(commands: SubCommand):
    find_parser = commands.add_parser(
        "find",
        aliases=["f"],
        help="find files/folders by pattern [offline]",
    )
    add_bool_argument(find_parser, "id_only")
    add_bool_argument(find_parser, "include_trash")
    find_parser.add_argument("pattern", type=str)
    find_parser.set_defaults(action=_action_find, id_only=False, include_trash=False)


async def _action_find(drive: Drive, args: Namespace) -> int:
    pattern: str = args.pattern
    id_only: bool = args.id_only

    nodes = await drive.find_nodes_by_regex(pattern)
    if not args.include_trash:
        nodes = (_ for _ in nodes if not _.is_trashed)

    if id_only:
        for node in nodes:
            print(node.id)
        return 0

    pairs = [
        await _
        for _ in as_completed(for_k_av(_.id, drive.resolve_path(_)) for _ in nodes)
    ]
    pairs.sort(key=lambda _: _[1])
    for id_, path in pairs:
        print(f"{id_}: {path}")

    return 0
