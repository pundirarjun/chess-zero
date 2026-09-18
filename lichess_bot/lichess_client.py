import requests
import json

from config import LICHESS_TOKEN, LICHESS_API


class LichessClient:

    def __init__(self, token=LICHESS_TOKEN):
        self.token = token

        self.headers = {
            "Authorization": f"Bearer {self.token}",
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)

    # ========================================================
    # ACCOUNT
    # ========================================================

    def get_account(self):
        """
        Get information about the authenticated Lichess account.
        """

        response = self.session.get(
            f"{LICHESS_API}/api/account"
        )

        response.raise_for_status()

        return response.json()

    # ========================================================
    # CHALLENGE
    # ========================================================

    def challenge_user(
        self,
        username,
        rated=False,
        clock_limit=600,
        clock_increment=0,
        color="random"
    ):
        """
        Challenge another Lichess user/bot.

        Parameters
        ----------
        username : str
            Lichess username of the opponent.

        rated : bool
            False = casual game.

        clock_limit : int
            Initial time in seconds.

        clock_increment : int
            Increment in seconds.

        color : str
            "white", "black", or "random".
        """

        url = f"{LICHESS_API}/api/challenge/{username}"

        data = {
            "rated": str(rated).lower(),
            "clock.limit": clock_limit,
            "clock.increment": clock_increment,
            "color": color,
            "variant": "standard",
        }

        response = self.session.post(
            url,
            data=data
        )

        response.raise_for_status()

        return response.json()

    # ========================================================
    # ACCEPT CHALLENGE
    # ========================================================

    def accept_challenge(self, challenge_id):
        """
        Accept an incoming challenge.
        """

        url = (
            f"{LICHESS_API}/api/challenge/"
            f"{challenge_id}/accept"
        )

        response = self.session.post(url)

        response.raise_for_status()

        return response.json() if response.content else {}

    # ========================================================
    # DECLINE CHALLENGE
    # ========================================================

    def decline_challenge(self, challenge_id):
        """
        Decline an incoming challenge.
        """

        url = (
            f"{LICHESS_API}/api/challenge/"
            f"{challenge_id}/decline"
        )

        response = self.session.post(url)

        response.raise_for_status()

        return response.json() if response.content else {}

    # ========================================================
    # EVENT STREAM
    # ========================================================

    def stream_events(self):
        """
        Stream events from Lichess.

        This includes:
        - challenges
        - game starts
        - game finishes
        """

        url = f"{LICHESS_API}/api/stream/event"

        with self.session.get(
            url,
            stream=True,
            timeout=None
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines():

                if not line:
                    continue

                try:
                    event = json.loads(
                        line.decode("utf-8")
                    )

                    yield event

                except json.JSONDecodeError:
                    continue

    # ========================================================
    # GAME STREAM
    # ========================================================

    def stream_game(self, game_id):
        """
        Stream the state of a particular game.
        """

        url = (
            f"{LICHESS_API}/api/bot/game/stream/"
            f"{game_id}"
        )

        with self.session.get(
            url,
            stream=True,
            timeout=None
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines():

                if not line:
                    continue

                try:
                    event = json.loads(
                        line.decode("utf-8")
                    )

                    yield event

                except json.JSONDecodeError:
                    continue

    # ========================================================
    # MAKE MOVE
    # ========================================================

    def make_move(self, game_id, move):
        """
        Send a move to Lichess.

        Parameters
        ----------
        game_id : str
            Lichess game ID.

        move : str
            UCI move, e.g. "e2e4".
        """

        url = (
            f"{LICHESS_API}/api/bot/game/"
            f"{game_id}/move/{move}"
        )

        response = self.session.post(url)

        response.raise_for_status()

        return response.json() if response.content else {}

    # ========================================================
    # RESIGN
    # ========================================================

    def resign_game(self, game_id):
        """
        Resign the current game.
        """

        url = (
            f"{LICHESS_API}/api/bot/game/"
            f"{game_id}/resign"
        )

        response = self.session.post(url)

        response.raise_for_status()

        return response.json() if response.content else {}