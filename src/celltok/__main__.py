"""Unified CLI: python -m celltok <train|evaluate|data|report|check>."""
import importlib
import sys


COMMANDS = {
    "train": "celltok.train",
    "report": "celltok.report",
    "check": "celltok.checks",
    "evaluate": {"tokenizer": "celltok.evaluation.tokenizer"},
    "data": {"convert": "celltok.data.convert", "freeze": "celltok.data.freeze", "validate": "celltok.data.validate"},
}


def main():
    arguments = sys.argv[1:]
    if not arguments or arguments[0] in ("-h", "--help"):
        print(__doc__)
        for name, target in COMMANDS.items():
            print(f"  {name}: {', '.join(target) if isinstance(target, dict) else target}")
        return
    name = arguments.pop(0)
    if name not in COMMANDS:
        raise SystemExit(f"Unknown command: {name}")
    target = COMMANDS[name]
    if isinstance(target, dict):
        if not arguments or arguments[0] in ("-h", "--help"):
            print(f"{name}: {', '.join(target)}")
            return
        action = arguments.pop(0)
        if action not in target:
            raise SystemExit(f"Unknown {name} action: {action}")
        target = target[action]
    module = importlib.import_module(target)
    sys.argv = [module.__file__, *arguments]
    module.main()


if __name__ == "__main__":
    main()
