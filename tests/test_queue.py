import contextlib
import hashlib
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch, MagicMock

from wcpan.drive.core.test import test_factory, TestDriver
from wcpan.drive.core.util import create_executor
from wcpan.drive.cli.queue_ import DownloadQueue, UploadQueue


class TestQueue(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    @patch("wcpan.drive.cli.queue_.upload_from_local")
    async def testUpload(self, fake_upload: MagicMock):
        async with upload_context(1) as (work_folder, drive, queue):
            # fake upload
            fake_upload.side_effect = bypass_upload

            root_node = await drive.get_root_node()
            await queue.run(
                [
                    work_folder / "folder",
                ],
                root_node,
            )

            self.assertFalse(queue.failed)
            self.assertTrue(fake_upload.called)
            self.assertEqual(len(fake_upload.mock_calls), 2)

    @patch("wcpan.drive.cli.queue_.upload_from_local")
    async def testUploadFailed(self, fake_upload: MagicMock):
        async with upload_context(1) as (work_folder, drive, queue):
            # fake upload
            fake_upload.side_effect = Exception("unknown")

            root_node = await drive.get_root_node()
            await queue.run(
                [
                    work_folder / "folder",
                ],
                root_node,
            )

            self.assertTrue(queue.failed)

    @patch("wcpan.drive.cli.queue_.download_to_local")
    async def testDownload(self, fake_download: MagicMock):
        async with download_context(1) as (work_folder, drive, queue):
            # fake download
            fake_download.side_effect = bypass_download

            node = await drive.get_node_by_path("/folder")
            await queue.run(
                [
                    node,
                ],
                work_folder,
            )

            self.assertFalse(queue.failed)

    @patch("wcpan.drive.cli.queue_.download_to_local")
    async def testDownload(self, fake_download: MagicMock):
        async with download_context(1) as (work_folder, drive, queue):
            # fake download
            fake_download.side_effect = Exception("unknown")

            node = await drive.get_node_by_path("/folder")
            await queue.run(
                [
                    node,
                ],
                work_folder,
            )

            self.assertTrue(queue.failed)


@contextlib.contextmanager
def common_context():
    with TemporaryDirectory() as work_folder:
        with create_executor() as pool:
            yield Path(work_folder), pool


@contextlib.asynccontextmanager
async def upload_context(jobs: int):
    with common_context() as (work_folder, pool):
        # prepare default files
        file_ = work_folder / "file"
        with file_.open("w") as fout:
            fout.write("file")
            fout.flush()

        folder = work_folder / "folder"
        folder.mkdir()

        file_ = folder / "file_1"
        with file_.open("w") as fout:
            fout.write("file_1")
            fout.flush()

        file_ = folder / "file_2"
        with file_.open("w") as fout:
            fout.write("file_2")
            fout.flush()

        with test_factory() as factory:
            async with factory(pool=pool) as drive:
                async for changes in drive.sync():
                    pass

                async with UploadQueue(drive, pool, jobs) as queue:
                    yield work_folder, drive, queue


@contextlib.asynccontextmanager
async def download_context(jobs: int):
    with common_context() as (work_folder, pool):
        with test_factory() as factory:
            async with factory(pool=pool) as drive:
                driver: TestDriver = drive.remote

                async for changes in drive.sync():
                    pass

                root_node = await drive.get_root_node()

                node = driver.pseudo.build_node()
                node.to_folder("file", root_node)
                node.to_file(0, "d41d8cd98f00b204e9800998ecf8427e", "text/plain")
                node.commit()

                node = driver.pseudo.build_node()
                node.to_folder("folder", root_node)
                folder = node.commit()

                node = driver.pseudo.build_node()
                node.to_folder("file_1", folder)
                node.to_file(1234, "9b4c8a5e36d3be7e2c4b1d75ded8c8a1", "text/plain")
                node.commit()

                node = driver.pseudo.build_node()
                node.to_folder("file_2", folder)
                node.to_file(4321, "b4aca3b69b75167fc91bb1fb48ea1c14", "text/plain")
                node.commit()

                async for changes in drive.sync():
                    pass

                async with DownloadQueue(drive, pool, jobs) as queue:
                    yield work_folder, drive, queue


async def bypass_upload(
    drive,
    parent_node,
    file_path,
    media_info,
    *,
    exist_ok=False,
):
    hasher = hashlib.md5()
    with open(file_path, "rb") as fin:
        chunk = fin.read()
        hasher.update(chunk)
    node = MagicMock()
    node.size = len(chunk)
    node.hash_ = hasher.hexdigest()
    return node


async def bypass_download(drive, node, local_path):
    local_file = local_path / node.name
    with open(local_file, "wb") as fout:
        for i in range(node.size):
            fout.write(b"\0")
        fout.flush()
    return local_file
