"""Clear the persisted locked speaker identity."""

from src.lock_state import LOCK_FILE, clear_lock


def main() -> None:
    if clear_lock():
        print(f"Unlocked. Removed {LOCK_FILE}")
    else:
        print("No locked identity was set.")


if __name__ == "__main__":
    main()
