from unittest.mock import AsyncMock, MagicMock
import hashlib

from wcpan.drive.core.abc import Hasher


class FakeHasher(Hasher):

    def __init__(self):
        self._hasher = hashlib.md5()

    def __getstate__(self):
        return hashlib.md5

    def __setstate__(self, hasher_class):
        self._hasher = hasher_class()

    def update(self, data):
        self._hasher.update(data)

    def hexdigest(self):
        return self._hasher.hexdigest()

    def digest(self):
        return self._hasher.digest()

    def copy(self):
        return self._hasher.copy()


def setup_drive_factory(FakeDriveFactory: MagicMock) -> None:
    fake_drive = AsyncMock()
    FakeDriveFactory.return_value = MagicMock(return_value=fake_drive)
    fack_methods = {
        '__aenter__.return_value': fake_drive,
    }
    fake_drive.configure_mock(**fack_methods)
    return fake_drive
