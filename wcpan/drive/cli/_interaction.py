import asyncio
import enum
import pathlib
import shlex

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory

from wcpan.drive.core.types import Drive, Node

from ._lib import cout, print_as_yaml


class TokenType(enum.Enum):
    Global = enum.auto()
    Path = enum.auto()


class ShellContext(object):
    def __init__(self, drive: Drive, home_node: Node) -> None:
        self._drive = drive
        self._home = home_node
        self._cwd = home_node
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
        }
        self._cache = ChildrenCache(drive)

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

    async def _list(self, src: str | None = None) -> None:
        if not src:
            node = self._cwd
        else:
            path = await normalize_path(self._drive, self._cwd, src)
            node = await self._drive.get_node_by_path(path)
            if not node:
                cout(f"{src} not found")
                return

        children = await self._drive.get_children(node)
        for child in children:
            cout(child.name)

    async def _chdir(self, src: str | None = None) -> None:
        if not src:
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

        self._cwd = node

    async def _mkdir(self, src: str) -> None:
        if not src:
            cout(f"invalid name")
            return

        await self._drive.create_directory(self._cwd, src)

    async def _sync(self) -> None:
        self._cache.reset()
        async for change in self._drive.sync():
            cout(change)

    async def _pwd(self) -> None:
        path = await self._drive.resolve_path(self._cwd)
        cout(path)

    async def _find(self, src: str) -> None:
        node_list = await self._drive.find_nodes_by_regex(src)
        path_list = [self._drive.resolve_path(node) for node in node_list]
        path_list = await asyncio.gather(*path_list)
        id_list = [node.id for node in node_list]
        rv = zip(id_list, path_list)
        for id_, path in rv:
            cout(f"{id_} - {path}")

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

    # reset anchor
    cout()


async def get_node_by_path_or_id(
    drive: Drive,
    cwd: pathlib.PurePath,
    path_or_id: str,
) -> Node:
    node = await drive.get_node_by_id(path_or_id)
    if node:
        return node

    path = pathlib.PurePath(path_or_id)
    if path.is_absolute():
        node = await drive.get_node_by_path(path)
        return node

    path = resolve_path(cwd, path)
    node = await drive.get_node_by_path(path)
    return node
