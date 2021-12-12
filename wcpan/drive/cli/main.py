from typing import List
import argparse
import asyncio
import functools
import io
import pathlib
import sys

from wcpan.logger import setup as setup_logger

from wcpan.drive.core.drive import DriveFactory
from wcpan.drive.core.util import (
    create_executor,
    get_default_config_path,
    get_default_data_path,
)

from . import __version__ as VERSION
from .queue_ import DownloadQueue, UploadQueue
from .util import (
    chunks_of,
    get_node_by_id_or_path,
    get_usage,
    get_utc_now,
    humanize,
    print_as_yaml,
    print_id_node_dict,
    trash_node,
    traverse_node,
    UploadVerifier,
    wait_for_value,
)
from .interaction import interact


async def main(args: List[str] = None) -> int:
    if args is None:
        args = sys.argv

    setup_logger((
        'wcpan.drive',
    ))

    args = parse_args(args[1:])
    if not args.action:
        await args.fallback_action()
        return 0

    factory = DriveFactory()
    factory.config_path = args.config_prefix
    factory.data_path = args.data_prefix
    factory.load_config()

    return await args.action(factory, args)


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser('wcpan.drive.cli')

    parser.add_argument('--version', '-v',
        action='version',
        version=f'wcpan.drive.cli {VERSION}',
    )

    parser.add_argument('--config-prefix',
        default=get_default_config_path(),
        help=(
            'specify configuration file path'
            ' (default: %(default)s)'
        ),
    )
    parser.add_argument('--data-prefix',
        default=get_default_data_path(),
        help=(
            'specify data file path'
            ' (default: %(default)s)'
        ),
    )

    commands = parser.add_subparsers()

    sync_parser = commands.add_parser('sync', aliases=['s'],
        help='synchronize database',
    )
    add_bool_argument(sync_parser, 'verbose', 'v')
    sync_parser.add_argument('-f', '--from', type=int, dest='from_',
        default=None,
        help=(
            'synchronize from certain check point, and do not update cache'
            ' (default: %(default)s)'
        ),
    )
    sync_parser.set_defaults(action=action_sync)

    find_parser = commands.add_parser('find', aliases=['f'],
        help='find files/folders by pattern [offline]',
    )
    add_bool_argument(find_parser, 'id_only')
    add_bool_argument(find_parser, 'include_trash')
    find_parser.add_argument('pattern', type=str)
    find_parser.set_defaults(action=action_find, id_only=False,
                             include_trash=False)

    info_parser = commands.add_parser('info', aliases=['i'],
        help='display file information [offline]',
    )
    info_parser.set_defaults(action=action_info)
    info_parser.add_argument('id_or_path', type=str)

    list_parser = commands.add_parser('list', aliases=['ls'],
        help='list folder [offline]',
    )
    list_parser.set_defaults(action=action_list)
    list_parser.add_argument('id_or_path', type=str)

    tree_parser = commands.add_parser('tree',
        help='recursive list folder [offline]',
    )
    tree_parser.set_defaults(action=action_tree)
    tree_parser.add_argument('id_or_path', type=str)

    usage_parser = commands.add_parser('usage', aliases=['du'],
        help=(
            'calculate space usage for files, '
            'recursively for folders [offline]'
        ),
    )
    usage_parser.add_argument('id_or_path', type=str, nargs='+')
    add_bool_argument(usage_parser, 'comma')
    usage_parser.set_defaults(action=action_usage, comma=False)

    dl_parser = commands.add_parser('download', aliases=['dl'],
        help='download files/folders',
    )
    dl_parser.set_defaults(action=action_download)
    dl_parser.add_argument('-j', '--jobs', type=int,
        default=1,
        help=(
            'maximum simultaneously download jobs'
            ' (default: %(default)s)'
        ),
    )
    dl_parser.add_argument('id_or_path', type=str, nargs='+')
    dl_parser.add_argument('destination', type=str)

    ul_parser = commands.add_parser('upload', aliases=['ul'],
        help='upload files/folders',
    )
    ul_parser.set_defaults(action=action_upload)
    ul_parser.add_argument('-j', '--jobs', type=int,
        default=1,
        help=(
            'maximum simultaneously upload jobs'
            ' (default: %(default)s)'
        ),
    )
    ul_parser.add_argument('source', type=str, nargs='+')
    ul_parser.add_argument('id_or_path', type=str)

    rm_parser = commands.add_parser('remove', aliases=['rm'],
        help='trash files/folders',
    )
    rm_parser.set_defaults(action=action_remove)
    rm_parser.add_argument('id_or_path', type=str, nargs='+')

    mv_parser = commands.add_parser('rename', aliases=['mv'],
        help='rename file/folder',
    )
    mv_parser.set_defaults(action=action_rename)
    mv_parser.add_argument('source_id_or_path', type=str, nargs='+')
    mv_parser.add_argument('destination_path', type=str)

    mkdir_parser = commands.add_parser('mkdir',
        help='create folder',
    )
    mkdir_parser.set_defaults(action=action_mkdir)
    mkdir_parser.add_argument('path', type=str)

    shell_parser = commands.add_parser('shell',
        help='start an interactive shell',
    )
    shell_parser.set_defaults(action=action_shell)
    shell_parser.add_argument('id_or_path', type=str, nargs='?')

    v_parser = commands.add_parser('verify', aliases=['v'],
        help='verify uploaded files/folders',
    )
    v_parser.set_defaults(action=action_verify)
    v_parser.add_argument('source', type=str, nargs='+')
    v_parser.add_argument('id_or_path', type=str)

    d_parser = commands.add_parser('doctor',
        help='check file system error'
    )
    d_parser.set_defaults(action=action_doctor)

    q_parser = commands.add_parser('quota', help='check upload quota')
    q_parser.set_defaults(action=action_quota)

    sout = io.StringIO()
    parser.print_help(sout)
    fallback = functools.partial(action_help, sout.getvalue())
    parser.set_defaults(action=None, fallback_action=fallback)

    args = parser.parse_args(args)

    return args


def add_bool_argument(
    parser: argparse.ArgumentParser,
    name: str,
    short_name: str = None,
) -> None:
    flag = name.replace('_', '-')
    pos_flags = ['--' + flag]
    if short_name:
        pos_flags.append('-' + short_name)
    neg_flag = '--no-' + flag
    parser.add_argument(*pos_flags, dest=name, action='store_true')
    parser.add_argument(neg_flag, dest=name, action='store_false')


async def action_help(message: str) -> None:
    print(message)


async def action_sync(factory: DriveFactory, args: argparse.Namespace) -> int:
    async with factory() as drive:
        chunks = chunks_of(drive.sync(check_point=args.from_), 100)
        async for changes in chunks:
            if not args.verbose:
                print(len(changes))
            else:
                for change in changes:
                    print_as_yaml(change)
        return 0


async def action_find(factory: DriveFactory, args: argparse.Namespace) -> int:
    async with factory() as drive:
        nodes = await drive.find_nodes_by_regex(args.pattern)
        if not args.include_trash:
            nodes = (_ for _ in nodes if not _.trashed)
        nodes = (wait_for_value(_.id_, drive.get_path(_)) for _ in nodes)
        nodes = await asyncio.gather(*nodes)
        nodes = dict(nodes)

    if args.id_only:
        for id_ in nodes:
            print(id_)
    else:
        print_id_node_dict(nodes)

    return 0


async def action_info(factory: DriveFactory, args: argparse.Namespace) -> int:
    async with factory() as drive:
        node = await get_node_by_id_or_path(drive, args.id_or_path)
    print_as_yaml(node.to_dict())
    return 0


async def action_list(factory: DriveFactory, args: argparse.Namespace) -> int:
    async with factory() as drive:
        node = await get_node_by_id_or_path(drive, args.id_or_path)
        nodes = await drive.get_children(node)
    nodes = {_.id_: _.name for _ in nodes}
    print_id_node_dict(nodes)
    return 0


async def action_tree(factory: DriveFactory, args: argparse.Namespace) -> int:
    async with factory() as drive:
        node = await get_node_by_id_or_path(drive, args.id_or_path)
        await traverse_node(drive, node, 0)
    return 0


async def action_usage(factory: DriveFactory, args: argparse.Namespace) -> int:
    async with factory() as drive:
        node_list = (get_node_by_id_or_path(drive, _) for _ in args.id_or_path)
        node_list = await asyncio.gather(*node_list)
        node_list = (get_usage(drive, _) for _ in node_list)
        node_list = await asyncio.gather(*node_list)
    for usage, src in zip(node_list, args.id_or_path):
        if args.comma:
            label = f'{usage:,}'
        else:
            label = f'{usage}'
        print(f'{label} - {src}')
    return 0


async def action_download(
    factory: DriveFactory,
    args: argparse.Namespace,
) -> int:
    with create_executor() as pool:
        async with factory(pool=pool) as drive:
            node_list = (get_node_by_id_or_path(drive, _)
                            for _ in args.id_or_path)
            node_list = await asyncio.gather(*node_list)
            node_list = [_ for _ in node_list if not _.trashed]

            async with DownloadQueue(drive, pool, args.jobs) as queue_:
                dst = pathlib.Path(args.destination)
                await queue_.run(node_list, dst)

    if not queue_.failed:
        return 0
    print('download failed:')
    print_as_yaml(queue_.failed)
    return 1


async def action_upload(factory: DriveFactory, args: argparse.Namespace) -> int:
    with create_executor() as pool:
        async with factory(pool=pool) as drive:
            node = await get_node_by_id_or_path(drive, args.id_or_path)

            async with UploadQueue(drive, pool, args.jobs) as queue_:
                src_list = [pathlib.Path(_) for _ in args.source]
                await queue_.run(src_list, node)

    if not queue_.failed:
        return 0
    print('upload failed:')
    print_as_yaml(queue_.failed)
    return 1


async def action_remove(factory: DriveFactory, args: argparse.Namespace) -> int:
    async with factory() as drive:
        rv = (trash_node(drive, _) for _ in args.id_or_path)
        rv = await asyncio.gather(*rv)
    rv = filter(None, rv)
    rv = list(rv)
    if not rv:
        return 0
    print('trash failed:')
    print_as_yaml(rv)
    return 1


async def action_rename(factory: DriveFactory, args: argparse.Namespace) -> int:
    async with factory() as drive:
        async def rename(id_or_path: str, dst: str) -> pathlib.PurePath:
            node = await get_node_by_id_or_path(drive, id_or_path)
            path = await drive.get_path(node)
            return await drive.rename_node_by_path(path, dst)

        node_list = (
            rename(_, args.destination_path)
            for _ in args.source_id_or_path
        )
        await asyncio.gather(*node_list)
    return 0


async def action_mkdir(factory: DriveFactory, args: argparse.Namespace) -> int:
    async with factory() as drive:
        path = pathlib.PurePath(args.path)
        parent_path = path.parent
        name = path.name
        parent_node = await drive.get_node_by_path(parent_path)
        await drive.create_folder(parent_node, name, exist_ok=True)
    return 0


async def action_shell(factory: DriveFactory, args: argparse.Namespace) -> int:
    async with factory() as drive:
        if not args.id_or_path:
            node = await drive.get_root_node()
        else:
            node = await get_node_by_id_or_path(drive, args.id_or_path)

    if not node or not node.is_folder:
        print(f'{args.id_or_path} is not a folder')
        return 1

    interact(factory, node)
    return 0


async def action_verify(factory: DriveFactory, args: argparse.Namespace) -> int:
    with create_executor() as pool:
        async with factory(pool=pool) as drive:
            node = await get_node_by_id_or_path(drive, args.id_or_path)

            v = UploadVerifier(drive, pool)
            tasks = (pathlib.Path(local_path) for local_path in args.source)
            tasks = [v.run(local_path, node) for local_path in tasks]
            await asyncio.wait(tasks)

    return 0


async def action_doctor(factory: DriveFactory, args: argparse.Namespace) -> int:
    async with factory() as drive:
        for node in await drive.find_multiple_parents_nodes():
            print(f'{node.name} has multiple parents, please select one parent:')
            parent_list = (drive.get_node_by_id(_) for _ in node.parent_list)
            parent_list = await asyncio.gather(*parent_list)
            for index, parent_node in enumerate(parent_list):
                print(f'{index}: {parent_node.name}')
            try:
                choice = input()
                choice = int(choice)
                parent = parent_list[choice]
                await drive.set_node_parent_by_id(node, parent.id_)
            except Exception as e:
                print('unknown error, skipped', e)
                continue


async def action_quota(factory: DriveFactory, args: argparse.Namespace) -> int:
    SECONDS_PER_HOUR = 60 * 60
    SECONDS_PER_DAY = SECONDS_PER_HOUR * 24
    async with factory() as drive:
        now = get_utc_now()
        yesterday = int(now.timestamp()) - SECONDS_PER_DAY
        total = 0
        for h in range(24):
            begin = yesterday + h * SECONDS_PER_HOUR
            end = begin + SECONDS_PER_HOUR
            size = await drive.get_uploaded_size(begin, end)
            total += size
            print(humanize(size))
        print(f'Total: {humanize(total)}')
    return 0
