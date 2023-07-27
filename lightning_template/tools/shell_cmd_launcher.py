from jsonargparse import CLI

from lightning_template.utils.cli import shell_cmd_launcher


def main():
    CLI(shell_cmd_launcher)


if __name__ == "__main__":
    main()
