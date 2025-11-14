import argparse
from data.cleaner import DatasetCleaner


def clean_files(sub_action, dry_run: bool):
    cleaner = DatasetCleaner(dry_run=dry_run)
    if sub_action == "split":
        cleaner.split_audio()
    elif sub_action == "clean":
        cleaner.clean()


if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument("action", metavar="A", help="Action to perform")
    arguments.add_argument("sub-action", metavar="S", help="Sub-action to perform")
    arguments.add_argument("--dry-run",
                           help="Shows what would be done without actually doing it",
                           action=argparse.BooleanOptionalAction)

    args = arguments.parse_args()

    if args.action == "clean":
        clean_files(args.sub_action, dry_run=args.dry_run)
    else:
        print("Invalid action")
