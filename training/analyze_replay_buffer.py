import sys
import os

# ==================================================
# Make project root importable
# ==================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import numpy as np


# ==================================================
# CONFIGURATION
# ==================================================

REPLAY_BUFFER_PATH = (
    "checkpoints/replay_buffer.pt"
)


# ==================================================
# LOAD REPLAY BUFFER
# ==================================================

print(
    "Loading replay buffer..."
)

data = torch.load(
    REPLAY_BUFFER_PATH,
    map_location="cpu",
    weights_only=False
)


# ==================================================
# DETERMINE DATA FORMAT
# ==================================================

if isinstance(data, dict):

    print(
        "\nReplay buffer file contains a dictionary."
    )

    print(
        "Keys:",
        list(data.keys())
    )

    if "buffer" in data:

        samples = data["buffer"]

    elif "samples" in data:

        samples = data["samples"]

    else:

        raise ValueError(
            "Could not find replay samples in checkpoint."
        )

else:

    samples = data


# ==================================================
# BASIC INFORMATION
# ==================================================

print(
    "\n=============================="
)

print(
    "REPLAY BUFFER ANALYSIS"
)

print(
    "=============================="
)

print(
    "Number of samples:",
    len(samples)
)


if len(samples) == 0:

    print(
        "Replay buffer is empty."
    )

    sys.exit(0)


# ==================================================
# EXTRACT VALUES
# ==================================================

values = []


for sample in samples:

    # Expected:
    #
    # sample = (
    #     state,
    #     policy,
    #     value
    # )

    if len(sample) != 3:

        raise ValueError(
            "Unexpected sample format."
        )

    state = sample[0]

    policy = sample[1]

    value = sample[2]

    values.append(
        float(value)
    )


values = np.asarray(
    values,
    dtype=np.float32
)


# ==================================================
# VALUE DISTRIBUTION
# ==================================================

wins = np.sum(
    values > 0.5
)

draws = np.sum(
    np.abs(values) < 0.5
)

losses = np.sum(
    values < -0.5
)


total = len(values)


print(
    "\n=============================="
)

print(
    "VALUE DISTRIBUTION"
)

print(
    "=============================="
)

print(
    "Win (+1):",
    int(wins)
)

print(
    "Draw (0):",
    int(draws)
)

print(
    "Loss (-1):",
    int(losses)
)


print(
    "\nPercentages:"
)

print(
    f"+1: {wins / total:.4f}"
)

print(
    f" 0: {draws / total:.4f}"
)

print(
    f"-1: {losses / total:.4f}"
)


# ==================================================
# CHECK EXACT VALUES
# ==================================================

unique_values, counts = np.unique(
    values,
    return_counts=True
)


print(
    "\n=============================="
)

print(
    "EXACT VALUE COUNTS"
)

print(
    "=============================="
)

for value, count in zip(
    unique_values,
    counts
):

    print(
        f"{value:+.1f}: {int(count)}"
    )


# ==================================================
# FIRST SAMPLE
# ==================================================

first_sample = samples[0]

first_state = first_sample[0]

first_policy = first_sample[1]

first_value = first_sample[2]


print(
    "\n=============================="
)

print(
    "FIRST SAMPLE"
)

print(
    "=============================="
)

print(
    "State shape:",
    np.asarray(
        first_state
    ).shape
)

print(
    "Policy shape:",
    np.asarray(
        first_policy
    ).shape
)

print(
    "Policy sum:",
    np.asarray(
        first_policy
    ).sum()
)

print(
    "Value:",
    first_value
)


# ==================================================
# POLICY STATISTICS
# ==================================================

policy_nonzero_counts = []

policy_entropies = []


for sample in samples:

    policy = np.asarray(
        sample[1],
        dtype=np.float32
    )

    nonzero = np.count_nonzero(
        policy > 0
    )

    policy_nonzero_counts.append(
        nonzero
    )

    # Numerical safety
    safe_policy = policy[
        policy > 0
    ]

    if len(safe_policy) > 0:

        entropy = -np.sum(
            safe_policy
            * np.log(
                safe_policy
            )
        )

        policy_entropies.append(
            entropy
        )


print(
    "\n=============================="
)

print(
    "POLICY STATISTICS"
)

print(
    "=============================="
)

print(
    "Average non-zero actions:",
    np.mean(
        policy_nonzero_counts
    )
)

print(
    "Minimum non-zero actions:",
    np.min(
        policy_nonzero_counts
    )
)

print(
    "Maximum non-zero actions:",
    np.max(
        policy_nonzero_counts
    )
)

if policy_entropies:

    print(
        "Average policy entropy:",
        np.mean(
            policy_entropies
        )
    )

    print(
        "Minimum policy entropy:",
        np.min(
            policy_entropies
        )
    )

    print(
        "Maximum policy entropy:",
        np.max(
            policy_entropies
        )
    )


# ==================================================
# SUMMARY
# ==================================================

print(
    "\n=============================="
)

print(
    "SUMMARY"
)

print(
    "=============================="
)

print(
    f"Total samples: {total}"
)

print(
    f"Win samples:   {int(wins)} "
    f"({wins / total * 100:.2f}%)"
)

print(
    f"Draw samples:  {int(draws)} "
    f"({draws / total * 100:.2f}%)"
)

print(
    f"Loss samples:  {int(losses)} "
    f"({losses / total * 100:.2f}%)"
)