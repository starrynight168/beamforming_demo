import sys
from importlib import import_module
from pathlib import Path

if not __package__:
    sys.path[0] = str(Path(__file__).resolve().parents[1])

from tools.shared.registry import TOOLS
from tools.shared.run_output import emit


def main():
    while True:
        print("\n超声工具总入口")
        for index, spec in enumerate(TOOLS, 1):
            print(f"  {index}. {spec['name']}")
        print("  0. 退出")
        try:
            selected = input("请选择工具: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        if selected in {"0", "q", "quit", "exit", "退出"}:
            print("已退出。")
            return
        if not selected.isdigit() or not 1 <= int(selected) <= len(TOOLS):
            emit(None, "warning", "选择无效，请重新输入。")
            continue
        try:
            import_module(TOOLS[int(selected) - 1]["entry"]).main()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                emit(None, "error", f"工具退出码：{exc.code}")
        except (OSError, ValueError) as exc:
            emit(None, "error", str(exc))
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        except Exception as exc:
            emit(None, "error", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
