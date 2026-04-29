class UserDict(dict):
    """User list class."""
    def __init__(self):
        super().__init__()
        self.replay_map = None

    def make_replay_map(self, replay_users):
        self.replay_map = {}

        available = list(self.values())  # users you can match

        for replay_user in replay_users:
            match = _find_and_consume(replay_user, available)

            if match is None:
                raise ValueError(f"No match found for {replay_user.address}")

            self.replay_map[replay_user.address] = match.address

    def __getitem__(self, item):
        mapped = self.replay_map.get(item, item)
        return super().__getitem__(mapped)

def _find_and_consume(replay_user, available):
    for i, user in enumerate(available):
        if _matches(replay_user, user):
            return available.pop(i)

    return None

def _matches(replay_user, user):
    return replay_user.attitude == replay_user.attitude