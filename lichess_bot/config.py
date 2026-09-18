import os


# ============================================================
# LICHESS CONFIGURATION
# ============================================================

# Your Lichess API token should be stored as an environment
# variable. DO NOT put the token directly into this file.
#
# Linux / Kaggle:
#
# export LICHESS_TOKEN="your_token_here"
#
# Kaggle:
# Add LICHESS_TOKEN as a secret.
#
LICHESS_TOKEN = os.getenv("LICHESS_TOKEN")


# ============================================================
# BOT SETTINGS
# ============================================================

# RL23 checkpoint
MODEL_PATH = "checkpoints/rl_iteration_23.pt"


# MCTS simulations per move
NUM_SIMULATIONS = 100


# ============================================================
# LICHESS API
# ============================================================

LICHESS_API = "https://lichess.org"


# ============================================================
# SAFETY CHECK
# ============================================================

if not LICHESS_TOKEN:
    raise RuntimeError(
        "LICHESS_TOKEN environment variable is not set.\n"
        "Please configure your Lichess API token before "
        "starting the bot."
    )