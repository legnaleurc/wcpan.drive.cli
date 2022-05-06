from unittest.mock import AsyncMock, MagicMock


def setup_drive_factory(FakeDriveFactory: MagicMock) -> None:
    fake_drive = AsyncMock()
    FakeDriveFactory.return_value = MagicMock(return_value=fake_drive)
    fack_methods = {
        '__aenter__.return_value': fake_drive,
    }
    fake_drive.configure_mock(**fack_methods)
    return fake_drive
