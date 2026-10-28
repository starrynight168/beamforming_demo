"""Compatibility shim for tools.core.ablate_sound_speed."""

from tools.core.ablate_sound_speed import *  # noqa: F401, F403
from tools.core.ablate_sound_speed import main

if __name__ == "__main__":
    main()
