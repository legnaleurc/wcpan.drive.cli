from argparse import Namespace

from wcpan.drive.core.types import Drive

from .lib import SubCommand


def add_auth_command(commands: SubCommand):
    auth_parser = commands.add_parser(
        "auth",
        aliases=["a"],
        help="authorize user",
    )
    auth_parser.set_defaults(action=_action_auth)


async def _action_auth(drive: Drive, args: Namespace) -> int:
    url = await drive.get_oauth_url()
    print("Access the following URL to authorize user:\n")
    print(url)
    print("")
    print("Paste the redireced URL or provided code here:")
    answer = input("")
    await drive.set_oauth_token(answer)
    return 0
