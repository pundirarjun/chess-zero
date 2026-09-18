import os

from kaggle_secrets import UserSecretsClient


# ============================================================
# LICHESS TOKEN
# ============================================================

user_secrets = UserSecretsClient()

try:
    LICHESS_TOKEN = user_secrets.get_secret("LICHESS_TOKEN")
except Exception as e:
    raise RuntimeError(
        "Could not access the Kaggle secret 'LICHESS_TOKEN'.\n"
        "Make sure the secret exists and is enabled for this notebook."
    ) from e


# ============================================================
# LICHESS API
# ============================================================

LICHESS_API = "https://lichess.org"


# ============================================================
# BOT SETTINGS
# ============================================================

MODEL_PATH = os.path.join(
    "checkpoints",
    "rl_iteration_23.pt"
)

NUM_SIMULATIONS = 100


# ============================================================
# SAFETY CHECK
# ============================================================

if not LICHESS_TOKEN:
    raise RuntimeError(
        "LICHESS_TOKEN is empty."
    )


print("Lichess configuration loaded.")