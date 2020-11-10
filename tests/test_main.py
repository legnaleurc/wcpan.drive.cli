from contextlib import AsyncExitStack
from io import StringIO
from pathlib import PurePath
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from wcpan.drive.cli.main import main


class TestMain(IsolatedAsyncioTestCase):

    async def asyncSetUp(self) -> None:
        async with AsyncExitStack() as stack:
            FakeDriveFactory = stack.enter_context(patch('wcpan.drive.cli.main.DriveFactory'))
            self._drive = setup_drive_factory(FakeDriveFactory)
            self._raii = stack.pop_all()

    async def asyncTearDown(self) -> None:
        await self._raii.aclose()

    @patch('sys.stderr', new_callable=StringIO)
    async def testMkdirNoArgs(self, stderr):
        with self.assertRaises(SystemExit):
            await main(['', 'mkdir'])

    @patch('sys.stderr', new_callable=StringIO)
    async def testMkdirWithPath(self, stderr):
        parent_node = MagicMock()
        self._drive.get_node_by_path.return_value = parent_node

        rv = await main(['', 'mkdir', '/var/log'])
        self.assertEqual(rv, 0)

        path = PurePath('/var/log')
        self._drive.get_node_by_path.assert_called_once_with(path.parent)
        self._drive.create_folder.assert_called_once_with(parent_node, path.name, exist_ok=True)


def setup_drive_factory(FakeDriveFactory: MagicMock) -> None:
    fake_drive = AsyncMock()
    FakeDriveFactory.return_value = MagicMock(return_value=fake_drive)
    fake_drive.configure_mock(**{
        '__aenter__.return_value': fake_drive,
    })
    return fake_drive
