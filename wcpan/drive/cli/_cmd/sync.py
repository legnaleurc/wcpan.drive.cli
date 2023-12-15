from argparse import Namespace
from dataclasses import asdict
from typing import TypeGuard

from wcpan.drive.core.types import Drive, ChangeAction, RemoveAction, UpdateAction

from .lib import SubCommand, add_bool_argument, require_authorized
from .._lib import cout, print_as_yaml


def add_sync_command(commands: SubCommand):
    parser = commands.add_parser(
        "sync",
        aliases=["s"],
        help="synchronize database",
    )
    add_bool_argument(parser, "verbose", short_true="v")
    parser.set_defaults(action=_action_sync)


@require_authorized
async def _action_sync(drive: Drive, kwargs: Namespace) -> int:
    verbose: bool = kwargs.verbose

    count = 0
    async for change in drive.sync():
        if verbose:
            if _is_remove(change):
                print_as_yaml([change[1]])
            if _is_update(change):
                print_as_yaml([asdict(change[1])])
        count += 1
    if not verbose:
        cout(f"{count}")
    return 0


def _is_remove(change: ChangeAction) -> TypeGuard[RemoveAction]:
    return change[0]


def _is_update(change: ChangeAction) -> TypeGuard[UpdateAction]:
    return not change[0]
