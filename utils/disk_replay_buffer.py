import lmdb, os, pickle, random
from collections import namedtuple
import shutil
from pathlib import Path

# Two record types: one for RL, one for DAgger / imitation
Transition       = namedtuple("Transition",       ("state", "action", "reward", "next_state", "done"))
ExpertTransition = namedtuple("ExpertTransition", ("state", "action"))


def safe_copy_lmdb(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_db = src_dir / "data.mdb"
    if not src_db.exists():
        return
    tmp = dst_dir / ".data.mdb.tmp"
    shutil.copy2(src_db, tmp)           # write full temp file
    os.replace(tmp, dst_dir / "data.mdb")  # atomic swap-in
    # NOTE: do not copy lock.mdb


class DiskReplayBuffer:
    """LMDB‑backed ring buffer supporting **RL** *and* **DAgger/imitation** with dynamic resizing.

    Parameters
    ----------
    path : str
        Directory where LMDB files are stored.
    capacity : int
        Maximum number of items kept (circular behaviour).
    map_size : int
        Initial DB size in bytes (can be expanded dynamically).
    mode : str  {"rl", "dagger"}
        • ``"rl"``     ➜ store full (s, a, r, s', done)
        • ``"dagger"`` ➜ store only (s, expert_action)
    """

    def __init__(self, path="replay_buffer", *, capacity=500_000, map_size=int(10e9 * 5), mode="rl"):
        assert mode in {"rl", "dagger"}, "mode must be 'rl' or 'dagger'"
        self.mode = mode
        self.record = Transition if mode == "rl" else ExpertTransition
        self.capacity = capacity
        self.keys: list[bytes] = []
        self.length = 0
        self.next_idx = 0
        


        if path:
            os.makedirs(path, exist_ok=True)
            self.env = lmdb.open(
                path, map_size=map_size, subdir=True,
                lock=True, readahead=False, meminit=False
            )
            self.db = self.env.open_db(None)
            # ── Rebuild metadata if DB already has entries ──
            with self.env.begin(write=False) as txn:
                cursor = txn.cursor()
                for k, _ in cursor:
                    self.keys.append(k)
                self.length = len(self.keys)
                if self.length:
                    last_idx = int(self.keys[-1])
                    self.next_idx = (last_idx + 1) % self.capacity
                    print(f"Rebuilt buffer with {self.length} entries, next_idx={self.next_idx}")
                else:
                    print("Initialized empty buffer, ready to store transitions.")

    def _key(self, idx: int) -> bytes:
        return f"{idx:012d}".encode()

    def push(self, *args):
        """Insert one record. Args must match buffer mode."""
        if self.mode == "rl":
            assert len(args) == 5, "RL mode expects 5 arguments"
        else:
            assert len(args) == 2, "DAgger mode expects (state, expert_action)"
        entry = self.record(*args)
        self.push_many([entry])

    def push_many(self, entries):
        """
        Insert a batch of records at once. More efficient than many individual pushes.
        Args:
            entries: list of `Transition` or `ExpertTransition` objects.
        """
        assert isinstance(entries, list), "Expected a list of entries"
        for entry in entries:
            assert isinstance(entry, self.record), f"Invalid record type: {type(entry)}"

        while True:
            try:
                with self.env.begin(write=True) as txn:
                    for entry in entries:
                        key = self._key(self.next_idx)
                        txn.put(key, pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))
                        if self.length == self.capacity:
                            self.keys[self.next_idx] = key
                        else:
                            self.keys.append(key)
                            self.length += 1
                        self.next_idx = (self.next_idx + 1) % self.capacity
                break
            except lmdb.MapFullError:
                old_size = self.env.info()["map_size"]
                new_size = old_size * 2
                print(f"⚠️ LMDB full: expanding from {old_size >> 30} GB to {new_size >> 30} GB")
                self.env.set_mapsize(new_size)

    def sample(self, batch_size):
        assert self.length, "Buffer empty – cannot sample."
        idxs = random.sample(range(self.length), batch_size)
        keys = [self.keys[i] for i in idxs]
        batch = []
        with self.env.begin(write=False) as txn:
            for k in keys:
                batch.append(pickle.loads(txn.get(k)))
        return self.record(*zip(*batch))

    def __len__(self):
        return self.length

    def size(self):
        stat = self.env.stat()
        info = self.env.info()
        return {
            "transitions": self.length,
            "disk_bytes": stat["psize"] * (info["last_pgno"] + 1)
        }

    def clear(self):
        """Remove all entries from the buffer."""
        with self.env.begin(write=True, db=self.db) as txn:
            txn.drop(self.db, delete=False)

        self.keys.clear()
        self.length = 0
        self.next_idx = 0
