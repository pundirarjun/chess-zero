"""
RL iteration orchestrator.

Runs RL18 -> RL25 automatically without modifying training/rl_training.py.

Starting data:
    RL17 checkpoint
    RL17 replay buffer

The RL17 files are expected somewhere under /kaggle/input.

For every iteration:
    previous checkpoint/replay buffer
        ->
    run rl_training.py
        ->
    new checkpoint/replay buffer
        ->
    delete ONLY the previous replay buffer

All RL model checkpoints are retained.

The script is also resumable:
if RL18-RL21 already exist, it will continue from the latest
completed iteration instead of starting again.
"""

from __future__ import annotations

import os
import re
import sys
import shutil
import subprocess
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

START_ITERATION = 25
END_ITERATION = 30

PROJECT_ROOT = Path(__file__).resolve().parent

RL_TRAINING_SCRIPT = PROJECT_ROOT / "training" / "rl_training.py"

WORKING_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

KAGGLE_INPUT_DIR = Path("/kaggle/input/datasets/arjunthakur9999/checkpoints/replay_buffer_rl25/replay_buffer_rl25.pt")

# Files required to start RL18
START_PREVIOUS_ITERATION = START_ITERATION - 1


# ============================================================
# PATH HELPERS
# ============================================================

def checkpoint_name(iteration: int) -> str:
    return f"rl_iteration_{iteration}.pt"


def replay_name(iteration: int) -> str:
    return f"replay_buffer_rl{iteration}.pt"


def checkpoint_path(iteration: int) -> Path:
    return WORKING_CHECKPOINT_DIR / checkpoint_name(iteration)


def replay_path(iteration: int) -> Path:
    return WORKING_CHECKPOINT_DIR / replay_name(iteration)


# ============================================================
# FIND RL17 FILES IN KAGGLE INPUT
# ============================================================

def find_in_kaggle_input(filename: str) -> Path | None:
    """
    Search /kaggle/input recursively for a required file.
    """

    print(f"\nSearching Kaggle Input for: {filename}")

    matches = list(KAGGLE_INPUT_DIR.rglob(filename))

    if not matches:
        return None

    if len(matches) > 1:
        print(f"Found {len(matches)} copies:")
        for path in matches:
            print("  ", path)

        print("Using:")
        print("  ", matches[0])

    return matches[0]


def prepare_starting_files() -> None:
    """
    Make sure RL17 checkpoint and replay buffer are available
    in the working checkpoints directory.
    """

    WORKING_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    previous_checkpoint = checkpoint_path(START_PREVIOUS_ITERATION)
    previous_replay = replay_path(START_PREVIOUS_ITERATION)

    # --------------------------------------------------------
    # RL17 checkpoint
    # --------------------------------------------------------

    if previous_checkpoint.exists():
        print("\nRL17 checkpoint already exists:")
        print(previous_checkpoint)
    else:
        source = find_in_kaggle_input(
            checkpoint_name(START_PREVIOUS_ITERATION)
        )

        if source is None:
            raise FileNotFoundError(
                f"\nCould not find {checkpoint_name(START_PREVIOUS_ITERATION)} "
                f"anywhere under {KAGGLE_INPUT_DIR}"
            )

        print("\nCopying RL17 checkpoint:")
        print("FROM:", source)
        print("TO:  ", previous_checkpoint)

        shutil.copy2(source, previous_checkpoint)

    # --------------------------------------------------------
    # RL17 replay buffer
    # --------------------------------------------------------

    if previous_replay.exists():
        print("\nRL17 replay buffer already exists:")
        print(previous_replay)
    else:
        source = find_in_kaggle_input(
            replay_name(START_PREVIOUS_ITERATION)
        )

        if source is None:
            raise FileNotFoundError(
                f"\nCould not find {replay_name(START_PREVIOUS_ITERATION)} "
                f"anywhere under {KAGGLE_INPUT_DIR}"
            )

        print("\nCopying RL17 replay buffer:")
        print("FROM:", source)
        print("TO:  ", previous_replay)

        shutil.copy2(source, previous_replay)

    print("\nRL17 starting files are ready.")


# ============================================================
# CREATE TEMPORARY RL TRAINING SCRIPT
# ============================================================

def create_iteration_script(iteration: int) -> Path:
    """
    Create a temporary copy of rl_training.py with the iteration
    number replaced.

    The original rl_training.py is NEVER modified.
    """

    if not RL_TRAINING_SCRIPT.exists():
        raise FileNotFoundError(
            f"RL training script not found:\n{RL_TRAINING_SCRIPT}"
        )

    source_text = RL_TRAINING_SCRIPT.read_text(encoding="utf-8")

    # --------------------------------------------------------
    # Replace RL_ITERATION
    # --------------------------------------------------------

    pattern = r"(?m)^RL_ITERATION\s*=\s*\d+\s*$"

    replacement = f"RL_ITERATION = {iteration}"

    modified_text, count = re.subn(
        pattern,
        replacement,
        source_text,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            "Could not find exactly one 'RL_ITERATION = ...' "
            "line in rl_training.py."
        )

    # --------------------------------------------------------
    # Temporary script location
    # --------------------------------------------------------

    temp_script = (
        PROJECT_ROOT
        / "training"
        / f"_rl_training_iteration_{iteration}.py"
    )

    temp_script.write_text(
        modified_text,
        encoding="utf-8",
    )

    print("\nTemporary training script created:")
    print(temp_script)

    return temp_script


# ============================================================
# VERIFY ITERATION OUTPUT
# ============================================================

def verify_iteration_outputs(iteration: int) -> None:
    """
    Verify that the iteration successfully produced both:
        rl_iteration_N.pt
        replay_buffer_rlN.pt
    """

    checkpoint = checkpoint_path(iteration)
    replay = replay_path(iteration)

    print("\nChecking iteration outputs...")

    if not checkpoint.exists():
        raise RuntimeError(
            f"RL{iteration} finished but checkpoint was not found:\n"
            f"{checkpoint}"
        )

    if checkpoint.stat().st_size == 0:
        raise RuntimeError(
            f"RL{iteration} checkpoint exists but is empty:\n"
            f"{checkpoint}"
        )

    if not replay.exists():
        raise RuntimeError(
            f"RL{iteration} finished but replay buffer was not found:\n"
            f"{replay}"
        )

    if replay.stat().st_size == 0:
        raise RuntimeError(
            f"RL{iteration} replay buffer exists but is empty:\n"
            f"{replay}"
        )

    print("Checkpoint:", checkpoint)
    print("Checkpoint size:", checkpoint.stat().st_size / (1024 ** 2), "MB")

    print("Replay buffer:", replay)
    print("Replay buffer size:", replay.stat().st_size / (1024 ** 2), "MB")

    print(f"RL{iteration} output verification: SUCCESS")


# ============================================================
# DELETE OLD REPLAY BUFFER
# ============================================================

def delete_old_replay_buffer(iteration: int) -> None:
    """
    Delete ONLY the old replay buffer.

    The old model checkpoint is deliberately retained.
    """

    old_replay = replay_path(iteration)

    if not old_replay.exists():
        print(
            f"\nOld replay buffer already absent: {old_replay}"
        )
        return

    size_mb = old_replay.stat().st_size / (1024 ** 2)

    print("\nDeleting old replay buffer:")
    print(old_replay)
    print(f"Size: {size_mb:.2f} MB")

    old_replay.unlink()

    print("Old replay buffer deleted.")

    # Explicitly show that checkpoint remains
    old_checkpoint = checkpoint_path(iteration)

    if old_checkpoint.exists():
        print("Old checkpoint KEPT:")
        print(old_checkpoint)
    else:
        print(
            "WARNING: old checkpoint was not found. "
            "This script did not delete it."
        )


# ============================================================
# RUN ONE RL ITERATION
# ============================================================

def run_iteration(iteration: int) -> None:
    """
    Run exactly one RL iteration.
    """

    previous_iteration = iteration - 1

    previous_checkpoint = checkpoint_path(previous_iteration)
    previous_replay = replay_path(previous_iteration)

    current_checkpoint = checkpoint_path(iteration)
    current_replay = replay_path(iteration)

    print("\n")
    print("=" * 70)
    print(f"STARTING RL ITERATION {iteration}")
    print("=" * 70)

    print("\nPrevious checkpoint:")
    print(previous_checkpoint)

    print("\nPrevious replay buffer:")
    print(previous_replay)

    print("\nExpected output checkpoint:")
    print(current_checkpoint)

    print("\nExpected output replay buffer:")
    print(current_replay)

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if not previous_checkpoint.exists():
        raise RuntimeError(
            f"Previous checkpoint missing for RL{iteration}:\n"
            f"{previous_checkpoint}"
        )

    if not previous_replay.exists():
        raise RuntimeError(
            f"Previous replay buffer missing for RL{iteration}:\n"
            f"{previous_replay}"
        )

    # --------------------------------------------------------
    # If this iteration already exists, don't retrain it.
    # --------------------------------------------------------

    if current_checkpoint.exists() and current_replay.exists():

        print("\nRL iteration already completed:")
        print(f"RL{iteration}")

        print("\nCheckpoint:")
        print(current_checkpoint)

        print("\nReplay buffer:")
        print(current_replay)

        return

    # --------------------------------------------------------
    # Create temporary training script
    # --------------------------------------------------------

    temp_script = create_iteration_script(iteration)

    try:

        print("\n")
        print("=" * 70)
        print(f"RUNNING rl_training.py AS RL{iteration}")
        print("=" * 70)

        # ----------------------------------------------------
        # Run training script
        # ----------------------------------------------------

        result = subprocess.run(
            [
                sys.executable,
                str(temp_script),
            ],
            cwd=str(PROJECT_ROOT),
            check=False,
        )

        # ----------------------------------------------------
        # Training failed
        # ----------------------------------------------------

        if result.returncode != 0:

            print("\n")
            print("=" * 70)
            print(f"RL{iteration} FAILED")
            print("=" * 70)

            print(
                "\nThe previous replay buffer was NOT deleted."
            )

            raise RuntimeError(
                f"rl_training.py failed for RL{iteration} "
                f"with exit code {result.returncode}."
            )

        # ----------------------------------------------------
        # Verify outputs before deleting anything
        # ----------------------------------------------------

        verify_iteration_outputs(iteration)

    finally:

        # ----------------------------------------------------
        # Delete temporary script
        # ----------------------------------------------------

        if temp_script.exists():
            temp_script.unlink()

            print("\nTemporary training script removed:")
            print(temp_script)

    # --------------------------------------------------------
    # Only AFTER successful verification:
    #
    # delete previous replay buffer.
    #
    # NEVER delete previous checkpoint.
    # --------------------------------------------------------

    delete_old_replay_buffer(previous_iteration)

    print("\n")
    print("=" * 70)
    print(f"RL{iteration} COMPLETE")
    print("=" * 70)


# ============================================================
# STORAGE SUMMARY
# ============================================================

def print_storage_summary() -> None:

    print("\n")
    print("=" * 70)
    print("CURRENT CHECKPOINT DIRECTORY")
    print("=" * 70)

    if not WORKING_CHECKPOINT_DIR.exists():
        print("Checkpoint directory does not exist.")
        return

    total_size = 0

    for path in sorted(WORKING_CHECKPOINT_DIR.iterdir()):

        if not path.is_file():
            continue

        size = path.stat().st_size
        total_size += size

        print(
            f"{path.name:<40} "
            f"{size / (1024 ** 2):>10.2f} MB"
        )

    print("-" * 70)
    print(
        f"TOTAL: {total_size / (1024 ** 3):.2f} GB"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CHESS AI RL ITERATION ORCHESTRATOR")
    print("=" * 70)

    print(f"\nRL range: RL{START_ITERATION} -> RL{END_ITERATION}")

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nRL training script:")
    print(RL_TRAINING_SCRIPT)

    print("\nWorking checkpoint directory:")
    print(WORKING_CHECKPOINT_DIR)

    # --------------------------------------------------------
    # Prepare RL17
    # --------------------------------------------------------

    prepare_starting_files()

    # --------------------------------------------------------
    # Run RL18 -> RL25
    # --------------------------------------------------------

    for iteration in range(START_ITERATION, END_ITERATION + 1):

        run_iteration(iteration)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_storage_summary()

    print("\n")
    print("=" * 70)
    print("RL TRAINING CHAIN COMPLETE")
    print("=" * 70)

    print(f"\nSuccessfully reached RL{END_ITERATION}.")

    print("\nFinal model:")
    print(checkpoint_path(END_ITERATION))

    print("\nFinal replay buffer:")
    print(replay_path(END_ITERATION))

    print("\nAll RL model checkpoints have been retained.")
    print("Only older replay buffers were removed.")


# ============================================================

if __name__ == "__main__":
    main()