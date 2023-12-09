from argparse import Namespace
from collections.abc import AsyncIterator

from wcpan.drive.core.types import Drive, ChangeAction

from .lib import SubCommand, add_bool_argument, require_authorized
from .._lib import print_as_yaml


def add_sync_command(commands: SubCommand):
    sync_parser = commands.add_parser(
        "sync",
        aliases=["s"],
        help="synchronize database",
    )
    add_bool_argument(sync_parser, "verbose", short_true="v")
    sync_parser.set_defaults(action=_action_sync)


@require_authorized
async def _action_sync(drive: Drive, args: Namespace) -> int:
    chunks = _chunks_of(drive.sync(), 100)
    async for changes in chunks:
        if not args.verbose:
            print(len(changes))
        else:
            for change in changes:
                print_as_yaml(change)
    return 0


async def _chunks_of(
    ag: AsyncIterator[ChangeAction], size: int
) -> AsyncIterator[list[ChangeAction]]:
    chunk: list[ChangeAction] = []
    async for item in ag:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
