# wcpan.drive.cli

Command line tool for `wcpan.drive`.

This package needs a driver to actually work with a cloud drive.

## Config

You will need a `core.yaml` file in **config path**.
By default it is `$HOME/.config/wcpan.drive`, but you can change it by
`--config-prefix` flag.

```sh
python3 -m wcpan.drive.cli --config-prefix=/path/to/config ...
```

Here is an example of `core.yaml`:

```yaml
version: 1
database: nodes.sqlite
driver: wcpan.drive.google.driver.GoogleDriver
middleware: []
```

## Command Line Usage

Get the latest help:

```sh
python3 -m wcpan.drive.cli -h
```

You need to authorize an user first.

```sh
python3 -m wcpan.drive.cli auth
```

Then you should build local cache.
Many commands reliy on this cache to avoid making too many API requests.

Note that this is the **ONLY** command that will update the cache.
Which means after `upload`, `mkdir`, `remove`, `rename`, you need to run this
command to make the cache up-to-date.

```sh
python3 -m wcpan.drive.cli sync
```

The `remove` command only put files to trash can, it does **NOT** permanently
remove **ANY** files.
Removing a folder will also remove all its descendants.

```
python3 -m wcpan.drive.cli remove file1 file2 ...
```
