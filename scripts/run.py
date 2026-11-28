import argparse
from scripts.script_utils import run_task

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--server", default="mine")
    parser.add_argument("--script", required=True)
    args, extra_args = parser.parse_known_args()

    params = {}
    remaining = iter(extra_args)

    for option in remaining:
        key, equals, value = option.partition("=")
        if not key.startswith("--") or key == "--":
            parser.error(f"Wrong args, {option!r}")

        if not equals:
            value = next(remaining, None)
            if value is None or value.startswith("--"):
                parser.error(f"Missing value for {key}")

        params[key[2:]] = value

    raise SystemExit(run_task(args.script, args.server, params))