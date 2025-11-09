import asyncio
import enum
import pathlib
import shlex
import tempfile
from asyncio import as_completed
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory

from wcpan.drive.core.exceptions import UnauthorizedError
from wcpan.drive.core.lib import download_file_to_local, move_node
from wcpan.drive.core.types import Drive, Node

from ._cmd.lib import get_node_by_id_or_path as get_node_by_id_or_path_cmd
from ._cmd.lib import get_path_by_id_or_path as get_path_by_id_or_path_cmd
from ._download import download_list
from ._lib import cerr, cout, print_as_yaml
from ._upload import upload_list
from .lib import create_executor


class TokenType(enum.Enum):
    Global = enum.auto()
    Path = enum.auto()


class ShellContext(object):
    def __init__(self, drive: Drive, home_node: Node) -> None:
        self._drive = drive
        self._home = home_node
        self._cwd = home_node
        self._prev_cwd: Node | None = None
        self._actions = {
            "help": self._help,
            "ls": self._list,
            "cd": self._chdir,
            "mkdir": self._mkdir,
            "sync": self._sync,
            "pwd": self._pwd,
            "find": self._find,
            "info": self._info,
            "hash": self._hash,
            "id_to_path": self._id_to_path,
            "path_to_id": self._path_to_id,
            "rm": self._remove,
            "remove": self._remove,
            "mv": self._move,
            "rename": self._move,
            "upload": self._upload,
            "ul": self._upload,
            "download": self._download,
            "dl": self._download,
            "cat": self._cat,
            "du": self._usage,
            "usage": self._usage,
            "trash": self._trash,
            "exit": self._exit,
            "quit": self._exit,
        }
        self._cache = ChildrenCache(drive)
        self._should_exit = False

    def get_prompt(self) -> str:
        if not self._cwd.name:
            name = "/"
        else:
            name = self._cwd.name
        return f"{name} > "

    async def execute_async(self, line: str) -> None:
        cmd = shlex.split(line)

        if not cmd:
            return

        command = cmd[0]
        if command not in self._actions:
            cout(f"unknown command {command}")
            return

        action = self._actions[command]
        try:
            await action(*cmd[1:])
        except TypeError as e:
            cout(e)
        except UnauthorizedError:
            cout("not authorized")
        except Exception as e:
            cout(f"error: {e}")

    def _get_global(self, prefix: str) -> list[str]:
        cmd = self._actions.keys()
        cmd = [c for c in cmd if c.startswith(prefix)]
        return cmd

    async def _get_path(self, prefix: str, path: str) -> list[str]:
        children = await self._cache.get(self._cwd, path)
        children = [c for c in children if c.startswith(prefix)]
        return children

    async def _help(self) -> None:
        cmd = self._actions.keys()
        for c in cmd:
            cout(c)

    async def _list(self, *args: str) -> None:
        # Parse flags
        long_format = False
        show_all = False
        human_readable = False
        recursive = False
        paths: list[str] = []

        for arg in args:
            if arg.startswith("-"):
                if "l" in arg:
                    long_format = True
                if "a" in arg:
                    show_all = True
                if "h" in arg:
                    human_readable = True
                if "R" in arg:
                    recursive = True
            else:
                paths.append(arg)

        # Get target node(s)
        if not paths:
            nodes = [self._cwd]
        else:
            nodes = []
            for path_str in paths:
                path = await normalize_path(self._drive, self._cwd, path_str)
                node = await self._drive.get_node_by_path(path)
                if not node:
                    cout(f"{path_str} not found")
                    continue
                nodes.append(node)

        # List each node
        for node in nodes:
            if recursive:
                await self._list_recursive(
                    node, long_format, show_all, human_readable, ""
                )
            else:
                await self._list_directory(node, long_format, show_all, human_readable)

    async def _list_directory(
        self, node: Node, long_format: bool, show_all: bool, human_readable: bool
    ) -> None:
        children = await self._drive.get_children(node)
        if not show_all:
            children = [c for c in children if not c.is_trashed]

        for child in children:
            if long_format:
                size = child.size if not child.is_directory else 0
                if human_readable:
                    size_str = self._format_size(size)
                else:
                    size_str = str(size)
                cout(f"{child.id}  {size_str:>10}  {child.mtime}  {child.name}")
            else:
                cout(child.name)

    async def _list_recursive(
        self,
        node: Node,
        long_format: bool,
        show_all: bool,
        human_readable: bool,
        prefix: str,
    ) -> None:
        path = await self._drive.resolve_path(node)
        cout(f"{prefix}{path}:")
        await self._list_directory(node, long_format, show_all, human_readable)

        children = await self._drive.get_children(node)
        if not show_all:
            children = [c for c in children if not c.is_trashed]

        for child in children:
            if child.is_directory:
                await self._list_recursive(
                    child, long_format, show_all, human_readable, prefix + "  "
                )

    def _format_size(self, size: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}PB"

    async def _chdir(self, src: str | None = None) -> None:
        if not src:
            self._prev_cwd = self._cwd
            self._cwd = self._home
            return

        # Handle special directories
        if src == "-":
            if self._prev_cwd is None:
                cout("no previous directory")
                return
            self._prev_cwd, self._cwd = self._cwd, self._prev_cwd
            return
        elif src == "~":
            self._prev_cwd = self._cwd
            self._cwd = self._home
            return

        path = await normalize_path(self._drive, self._cwd, src)
        node = await self._drive.get_node_by_path(path)
        if not node:
            cout(f"unknown path {src}")
            return
        if not node.is_directory:
            cout(f"{src} is not a folder")
            return

        self._prev_cwd = self._cwd
        self._cwd = node

    async def _mkdir(self, *args: str) -> None:
        if not args:
            cout("invalid name")
            return

        # Parse flags
        create_parents = False
        paths: list[str] = []

        for arg in args:
            if arg == "-p":
                create_parents = True
            elif arg.startswith("-"):
                cout(f"unknown flag: {arg}")
                return
            else:
                paths.append(arg)

        if not paths:
            cout("invalid name")
            return

        for path_str in paths:
            path = pathlib.PurePath(path_str)
            if path.is_absolute():
                # Absolute path
                parent_path = path.parent
                name = path.name
                if create_parents:
                    # Create parent directories
                    current_path = pathlib.PurePath("/")
                    for part in parent_path.parts[1:]:  # Skip root
                        current_path = current_path / part
                        parent_node = await self._drive.get_node_by_path(current_path)
                        if not parent_node:
                            # Create this parent
                            parent_parent_path = current_path.parent
                            parent_parent_node = await self._drive.get_node_by_path(
                                parent_parent_path
                            )
                            parent_node = await self._drive.create_directory(
                                part, parent_parent_node, exist_ok=True
                            )
                    # Create final directory
                    await self._drive.create_directory(
                        name, parent_node, exist_ok=create_parents
                    )
                else:
                    parent_node = await self._drive.get_node_by_path(parent_path)
                    if not parent_node:
                        cout(f"parent directory {parent_path} does not exist")
                        return
                    await self._drive.create_directory(
                        name, parent_node, exist_ok=create_parents
                    )
            else:
                # Relative path
                if create_parents:
                    # Create nested path
                    current_node = self._cwd
                    for part in path.parts[:-1]:
                        child = await self._drive.get_child_by_name(part, current_node)
                        if not child:
                            current_node = await self._drive.create_directory(
                                part, current_node, exist_ok=True
                            )
                        else:
                            if not child.is_directory:
                                cout(f"{part} is not a directory")
                                return
                            current_node = child
                    # Create final directory
                    await self._drive.create_directory(
                        path.parts[-1], current_node, exist_ok=create_parents
                    )
                else:
                    # Simple case: create in current directory
                    await self._drive.create_directory(
                        path_str, self._cwd, exist_ok=create_parents
                    )

    async def _sync(self) -> None:
        self._cache.reset()
        async for change in self._drive.sync():
            cout(change)

    async def _pwd(self) -> None:
        path = await self._drive.resolve_path(self._cwd)
        cout(path)

    async def _find(self, *args: str) -> None:
        if not args:
            cout("pattern required")
            return

        # Parse flags
        id_only = False
        include_trash = False
        pattern: str | None = None

        for arg in args:
            if arg == "--id-only":
                id_only = True
            elif arg == "--include-trash":
                include_trash = True
            elif not arg.startswith("-"):
                pattern = arg
            else:
                cout(f"unknown flag: {arg}")
                return

        if not pattern:
            cout("pattern required")
            return

        node_list = await self._drive.find_nodes_by_regex(pattern)
        if not include_trash:
            node_list = [n for n in node_list if not n.is_trashed]

        if id_only:
            for node in node_list:
                cout(node.id)
        else:
            path_list = [self._drive.resolve_path(node) for node in node_list]
            path_list = await asyncio.gather(*path_list)
            id_list = [node.id for node in node_list]
            rv = zip(id_list, path_list)
            for id_, path in rv:
                cout(f"{id_}: {path}")

    async def _info(self, src: str) -> None:
        from dataclasses import asdict

        node = await self._drive.get_node_by_id(src)
        if not node:
            cout("null")
        else:
            print_as_yaml(asdict(node))

    async def _hash(self, *args: str) -> None:
        base_path = await self._drive.resolve_path(self._cwd)
        node_list = [
            get_node_by_path_or_id(self._drive, base_path, path_or_id)
            for path_or_id in args
        ]
        node_list = await asyncio.gather(*node_list)
        hash_list = [node.hash_ for node in node_list]
        rv = zip(args, hash_list)
        for path_or_id, hash_ in rv:
            cout(f"{hash_} - {path_or_id}")

    async def _id_to_path(self, src: str) -> None:
        node = await self._drive.get_node_by_id(src)
        if not node:
            cout(f"{src} not found")
            return
        path = await self._drive.resolve_path(node)
        cout(path)

    async def _path_to_id(self, src: str) -> None:
        path = await normalize_path(self._drive, self._cwd, src)
        node = await self._drive.get_node_by_path(path)
        if not node:
            cout(f"{src} not found")
            return
        cout(node.id)

    async def _remove(self, *args: str) -> None:
        if not args:
            cout("path required")
            return

        # Parse flags
        restore = False
        purge = False
        paths: list[str] = []

        for arg in args:
            if arg == "--restore":
                restore = True
            elif arg == "--purge":
                purge = True
            elif not arg.startswith("-"):
                paths.append(arg)
            else:
                cout(f"unknown flag: {arg}")
                return

        if restore and purge:
            cerr("`--purge` flag conflicts with `--restore`")
            return

        if not paths:
            cout("path required")
            return

        try:
            for path_str in paths:
                node = await get_node_by_path_or_id(self._drive, self._cwd, path_str)
                if purge:
                    await self._drive.delete(node)
                else:
                    await self._drive.move(node, trashed=not restore)
        except UnauthorizedError:
            cout("not authorized")
        except Exception as e:
            cerr(f"operation failed: {e}")

    async def _move(self, *args: str) -> None:
        if len(args) < 2:
            cout("source and destination required")
            return

        sources = args[:-1]
        destination = args[-1]

        try:
            dst_path = pathlib.PurePath(destination)
            for src_str in sources:
                src_path = await get_path_by_id_or_path(self._drive, self._cwd, src_str)
                await move_node(self._drive, src_path, dst_path)
        except UnauthorizedError:
            cout("not authorized")
        except Exception as e:
            cerr(f"operation failed: {e}")

    async def _upload(self, *args: str) -> None:
        if len(args) < 2:
            cout("source and destination required")
            return

        # Parse flags
        jobs = 1
        sources: list[str] = []
        destination: str | None = None

        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "-j" or arg == "--jobs":
                if i + 1 >= len(args):
                    cout("jobs value required")
                    return
                try:
                    jobs = int(args[i + 1])
                except ValueError:
                    cout("invalid jobs value")
                    return
                i += 2
            elif not arg.startswith("-"):
                sources.append(arg)
                i += 1
            else:
                cout(f"unknown flag: {arg}")
                return

        if not sources:
            cout("source required")
            return

        # Last argument is destination
        destination = sources.pop()
        if not sources:
            cout("at least one source required")
            return

        # Get destination node
        try:
            dst_node = await get_node_by_path_or_id(self._drive, self._cwd, destination)
            if not dst_node.is_directory:
                cout(f"{destination} is not a directory")
                return

            # Upload files
            with create_executor() as pool:
                src_paths = [Path(s) for s in sources]
                ok = await upload_list(
                    src_paths, dst_node, drive=self._drive, pool=pool, jobs=jobs
                )
                if not ok:
                    cerr("upload failed")
        except UnauthorizedError:
            cout("not authorized")
        except Exception as e:
            cerr(f"upload failed: {e}")

    async def _download(self, *args: str) -> None:
        if len(args) < 2:
            cout("source and destination required")
            return

        # Parse flags
        jobs = 1
        include_trash = False
        sources: list[str] = []
        destination: str | None = None

        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "-j" or arg == "--jobs":
                if i + 1 >= len(args):
                    cout("jobs value required")
                    return
                try:
                    jobs = int(args[i + 1])
                except ValueError:
                    cout("invalid jobs value")
                    return
                i += 2
            elif arg == "--include-trash":
                include_trash = True
                i += 1
            elif not arg.startswith("-"):
                sources.append(arg)
                i += 1
            else:
                cout(f"unknown flag: {arg}")
                return

        if not sources:
            cout("source required")
            return

        # Last argument is destination
        destination = sources.pop()
        if not sources:
            cout("at least one source required")
            return

        try:
            # Get source nodes
            node_list = []
            for src_str in sources:
                node = await get_node_by_path_or_id(self._drive, self._cwd, src_str)
                if not node.is_trashed or include_trash:
                    node_list.append(node)

            # Download files
            dst_path = Path(destination)
            with create_executor() as pool:
                ok = await download_list(
                    node_list,
                    dst_path,
                    drive=self._drive,
                    pool=pool,
                    jobs=jobs,
                    include_trash=include_trash,
                )
                if not ok:
                    cerr("download failed")
        except UnauthorizedError:
            cout("not authorized")
        except Exception as e:
            cerr(f"download failed: {e}")

    async def _cat(self, src: str) -> None:
        if not src:
            cout("file path required")
            return

        try:
            node = await get_node_by_path_or_id(self._drive, self._cwd, src)
            if node.is_directory:
                cout(f"{src} is a directory")
                return

            # Download to temp file and display
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                local_file = await download_file_to_local(self._drive, node, tmp_path)
                with open(local_file, "rb") as f:
                    content = f.read()
                    try:
                        # Try to decode as text
                        text = content.decode("utf-8")
                        cout(text, end="")
                    except UnicodeDecodeError:
                        # If not text, just output raw bytes
                        import sys

                        sys.stdout.buffer.write(content)
                        sys.stdout.buffer.flush()
        except UnauthorizedError:
            cout("not authorized")
        except Exception as e:
            cerr(f"cat failed: {e}")

    async def _usage(self, *args: str) -> None:
        if not args:
            cout("path required")
            return

        # Parse flags
        use_comma = True
        paths: list[str] = []

        for arg in args:
            if arg == "--no-comma":
                use_comma = False
            elif not arg.startswith("-"):
                paths.append(arg)
            else:
                cout(f"unknown flag: {arg}")
                return

        if not paths:
            cout("path required")
            return

        try:
            for path_str in paths:
                node = await get_node_by_path_or_id(self._drive, self._cwd, path_str)
                if not node.is_directory:
                    usage = node.size
                else:
                    usage = 0
                    async for _root, _folders, files in self._drive.walk(node):
                        usage += sum(_.size for _ in files)

                if use_comma:
                    cout(f"{usage:,} - {path_str}")
                else:
                    cout(f"{usage} - {path_str}")
        except Exception as e:
            cerr(f"usage failed: {e}")

    async def _trash(self, *args: str) -> None:
        if not args:
            cout("trash subcommand required (list, usage, purge)")
            return

        subcommand = args[0]
        sub_args = args[1:]

        if subcommand == "list":
            await self._trash_list(*sub_args)
        elif subcommand == "usage" or subcommand == "df":
            await self._trash_usage(*sub_args)
        elif subcommand == "purge" or subcommand == "prune":
            await self._trash_purge(*sub_args)
        else:
            cout(f"unknown trash subcommand: {subcommand}")

    async def _trash_list(self, *args: str) -> None:
        flatten = False

        for arg in args:
            if arg == "--flatten":
                flatten = True
            elif not arg.startswith("-"):
                cout(f"unknown argument: {arg}")
                return
            else:
                cout(f"unknown flag: {arg}")
                return

        try:
            node_list = await self._drive.get_trashed_nodes(flatten)
            node_list.sort(key=lambda _: _.mtime)
            rv = [
                {
                    "id": _.id,
                    "name": _.name,
                    "ctime": str(_.ctime),
                    "mtime": str(_.mtime),
                }
                for _ in node_list
            ]
            print_as_yaml(rv)
        except Exception as e:
            cerr(f"trash list failed: {e}")

    async def _trash_usage(self, *args: str) -> None:
        use_comma = True

        for arg in args:
            if arg == "--no-comma":
                use_comma = False
            elif not arg.startswith("-"):
                cout(f"unknown argument: {arg}")
                return
            else:
                cout(f"unknown flag: {arg}")
                return

        try:
            calculator = UsageCalculator(self._drive)
            node_list = await self._drive.get_trashed_nodes()
            rv = await calculator(node_list)
            if use_comma:
                cout(f"{rv:,}")
            else:
                cout(f"{rv}")
        except Exception as e:
            cerr(f"trash usage failed: {e}")

    async def _trash_purge(self, *args: str) -> None:
        ask = True

        for arg in args:
            if arg == "-y" or arg == "--no-ask":
                ask = False
            elif not arg.startswith("-"):
                cout(f"unknown argument: {arg}")
                return
            else:
                cout(f"unknown flag: {arg}")
                return

        try:
            node_list = await self._drive.get_trashed_nodes()
            count = len(node_list)
            cout(f"Purging {count} items in trash ...")

            if ask:
                answer = input("Are you sure? [y/N]")
                answer = answer.lower()
                if answer != "y":
                    cout("Aborted.")
                    return

            await self._drive.purge_trash()
            cout("Done.")
        except UnauthorizedError:
            cout("not authorized")
        except Exception as e:
            cerr(f"trash purge failed: {e}")

    async def _exit(self) -> None:
        self._should_exit = True

    def should_exit(self) -> bool:
        return self._should_exit


class ChildrenCache(object):
    def __init__(self, drive: Drive) -> None:
        self._drive = drive
        self._cache: dict[str, list[str]] = {}

    async def get(self, cwd: Node, src: str) -> list[str]:
        key = f"{cwd.id}:{src}"
        if key in self._cache:
            return self._cache[key]

        path = await normalize_path(self._drive, cwd, src)
        node = await self._drive.get_node_by_path(path)
        if not node:
            parent_path = path.parent
            node = await self._drive.get_node_by_path(parent_path)
            assert node

        children = await self._drive.get_children(node)
        self._cache[key] = [child.name for child in children]
        return self._cache[key]

    def reset(self) -> None:
        self._cache = {}


class DriveCompleter(Completer):
    def __init__(self, context: ShellContext) -> None:
        self._context = context

    async def get_completions_async(self, document, complete_event):
        text = document.text_before_cursor
        end_index = document.cursor_position

        type_, token = parse_completion(text, end_index)

        # Extract the prefix (the part of the token that's been typed so far)
        # The token from parse_completion is the full token, but we need just the prefix
        # Find where the token ends in the text
        if token:
            # Find the last occurrence of the token before the cursor
            token_start = text.rfind(token, 0, end_index)
            if token_start != -1:
                # The prefix is from token_start to cursor position
                prefix = text[token_start:end_index]
            else:
                # If token not found, use word before cursor as prefix
                prefix = document.get_word_before_cursor(WORD=True) or ""
        else:
            prefix = document.get_word_before_cursor(WORD=True) or ""

        if type_ == TokenType.Global:
            values = self._context._get_global(prefix)
        elif type_ == TokenType.Path:
            values = await self._context._get_path(prefix, token)
        else:
            return

        for value in values:
            yield Completion(value, start_position=-len(prefix))


def resolve_path(
    from_: pathlib.PurePath,
    to: pathlib.PurePath,
) -> pathlib.PurePath:
    rv = from_
    for part in to.parts:
        if part == ".":
            continue
        elif part == "..":
            rv = rv.parent
        else:
            rv = rv / part
    return rv


async def normalize_path(
    drive: Drive,
    node: Node,
    string: str,
) -> pathlib.PurePath:
    path = pathlib.PurePath(string)
    if not path.is_absolute():
        current_path = await drive.resolve_path(node)
        path = resolve_path(current_path, path)
    return path


def parse_completion(whole_text: str, end_index: int) -> tuple[TokenType, str]:
    lexer = shlex.shlex(whole_text, posix=True)
    lexer.whitespace_split = True

    cmd: list[tuple[int, str]] = []
    offset = 0

    while True:
        try:
            token = lexer.get_token()
        except ValueError:
            idx = whole_text.find(token, offset)
            assert idx >= 0
            offset = idx + len(token)
            cmd.append((idx, lexer.token))
            break

        if token == lexer.eof:
            break

        idx = whole_text.find(token, offset)
        assert idx >= 0
        offset = idx + len(token)
        cmd.append((idx, token))

    idx = None
    token = None
    token_list = [(idx, offset, token) for idx, (offset, token) in enumerate(cmd)]
    for idx, offset, token in reversed(token_list):
        if offset <= end_index:
            break

    if idx == 0:
        return TokenType.Global, token
    else:
        return TokenType.Path, token


async def interact_async(drive: Drive, home_node: Node) -> None:
    context = ShellContext(drive, home_node)
    completer = DriveCompleter(context)
    session = PromptSession(completer=completer, history=InMemoryHistory())

    while True:
        prompt = context.get_prompt()
        try:
            line = await session.prompt_async(prompt)
        except EOFError:
            break

        await context.execute_async(line)

        if context.should_exit():
            break

    # reset anchor
    cout()


async def get_node_by_path_or_id(
    drive: Drive,
    cwd: pathlib.PurePath | Node,
    path_or_id: str,
) -> Node:
    # Handle Node as cwd
    if isinstance(cwd, Node):
        cwd_path = await drive.resolve_path(cwd)
    else:
        cwd_path = cwd

    node = await drive.get_node_by_id(path_or_id)
    if node:
        return node

    path = pathlib.PurePath(path_or_id)
    if path.is_absolute():
        node = await drive.get_node_by_path(path)
        return node

    path = resolve_path(cwd_path, path)
    node = await drive.get_node_by_path(path)
    return node


async def get_path_by_id_or_path(
    drive: Drive,
    cwd: pathlib.PurePath | Node,
    id_or_path: str,
) -> pathlib.PurePath:
    # Try using the command version first
    try:
        return await get_path_by_id_or_path_cmd(drive, id_or_path)
    except Exception:
        pass
    # Fallback to manual resolution
    if id_or_path.startswith("/"):
        return pathlib.PurePath(id_or_path)
    node = await drive.get_node_by_id(id_or_path)
    if node:
        path = await drive.resolve_path(node)
        return path
    # Handle Node as cwd
    if isinstance(cwd, Node):
        cwd_path = await drive.resolve_path(cwd)
    else:
        cwd_path = cwd
    path = pathlib.PurePath(id_or_path)
    if path.is_absolute():
        return path
    return resolve_path(cwd_path, path)


class UsageCalculator:
    def __init__(self, drive: Drive) -> None:
        self._drive = drive
        self._known: set[str] = set()

    async def __call__(self, node_list: list[Node]) -> int:
        rv = 0
        for node in node_list:
            if node.is_directory:
                children = await self._drive.get_children(node)
                rv += await self(children)
            elif node.id not in self._known:
                rv += node.size
                self._known.add(node.id)
        return rv
