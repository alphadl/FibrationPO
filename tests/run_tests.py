#!/usr/bin/env python3
# Run tests without pytest: python tests/run_tests.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run() -> None:
    import importlib.util
    test_dir = Path(__file__).parent
    for name in ["test_rgf", "test_apc_obj", "test_fbg", "test_fiber_po", "test_fiber_po_domain"]:
        p = test_dir / f"{name}.py"
        if not p.exists():
            continue
        spec = importlib.util.spec_from_file_location(name, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in dir(mod):
            if attr.startswith("test_") and callable(getattr(mod, attr)):
                getattr(mod, attr)()
                print(f"  OK {name}.{attr}")
    print("All tests passed.")


if __name__ == "__main__":
    run()
