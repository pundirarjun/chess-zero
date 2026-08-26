import numpy as np


def analyze_training_data(
    training_data
):

    if not training_data:

        print(
            "No training data."
        )

        return


    values = np.array(
        [
            sample[2]
            for sample in training_data
        ],
        dtype=np.float32
    )


    print(
        "\n=============================="
    )

    print(
        "TRAINING DATA ANALYSIS"
    )

    print(
        "=============================="
    )


    print(
        "Number of samples:",
        len(training_data)
    )


    print(
        "\nValue distribution:"
    )


    print(
        "White-win perspective (+1):",
        np.sum(values == 1.0)
    )


    print(
        "Draw (0):",
        np.sum(values == 0.0)
    )


    print(
        "Loss (-1):",
        np.sum(values == -1.0)
    )


    print(
        "\nValue percentages:"
    )


    total = len(values)


    print(
        "+1:",
        np.mean(values == 1.0)
    )


    print(
        "0:",
        np.mean(values == 0.0)
    )


    print(
        "-1:",
        np.mean(values == -1.0)
    )


    # ------------------------------------------
    # Policy statistics
    # ------------------------------------------

    policies = np.array(
        [
            sample[1]
            for sample in training_data
        ],
        dtype=np.float32
    )


    policy_entropies = []


    for policy in policies:

        nonzero = policy[
            policy > 0
        ]

        entropy = -np.sum(
            nonzero * np.log(
                nonzero
            )
        )

        policy_entropies.append(
            entropy
        )


    print(
        "\nPolicy statistics:"
    )


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