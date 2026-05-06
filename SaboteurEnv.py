from __future__ import annotations

import subprocess
import time
import shutil
import shlex
import weakref
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests


JsonDict = Dict[str, Any]
ActionLike = Union[int, JsonDict]
CommandLike = Union[str, Sequence[str]]


class SaboteurServerError(RuntimeError):
    """Raised when the Java Saboteur server returns an error response."""


def default_server_command(project_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """Return a cross-platform command for starting the Java server."""
    root = Path(project_dir).resolve() if project_dir is not None else Path(__file__).resolve().parent
    if os.name == "nt":
        wrapper = root / "mvnw.cmd"
        if wrapper.exists():
            return [str(wrapper), "exec:java"]
    else:
        wrapper = root / "mvnw"
        if wrapper.exists():
            return [str(wrapper), "exec:java"]
    return ["mvn", "exec:java"]


@dataclass(frozen=True)
class SaboteurAction:
    """Convenience constructors for server-compatible action dictionaries."""

    @staticmethod
    def discard(hand_index: int) -> JsonDict:
        return {"type": "DISCARD", "handIndex": hand_index}

    @staticmethod
    def play_path(hand_index: int, x: int, y: int, rotated: bool = False) -> JsonDict:
        return {
            "type": "PLAY_PATH",
            "handIndex": hand_index,
            "x": x,
            "y": y,
            "rotated": rotated,
        }

    @staticmethod
    def play_player(hand_index: int, target_player: int) -> JsonDict:
        return {
            "type": "PLAY_PLAYER",
            "handIndex": hand_index,
            "targetPlayer": target_player,
        }

    @staticmethod
    def play_map(hand_index: int, goal: str) -> JsonDict:
        return {"type": "PLAY_MAP", "handIndex": hand_index, "goal": goal}

    @staticmethod
    def play_rockfall(hand_index: int, x: int, y: int) -> JsonDict:
        return {"type": "PLAY_ROCKFALL", "handIndex": hand_index, "x": x, "y": y}


class SaboteurEnv:
    """
    Thin Python client for the Java Saboteur HTTP server.

    This intentionally does not depend on Gym/Gymnasium yet. Its API mirrors the
    common shape closely enough for training code:

        obs = env.reset()
        actions = env.legal_actions()
        obs, reward, done, info = env.step(actions[0])

    `action` can be either a legal action dict or an integer index into the most
    recent legal action list.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        server_command: Optional[CommandLike] = None,
        server_cwd: Optional[Union[str, Path]] = None,
        request_timeout: float = 10.0,
        startup_timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if isinstance(server_command, str):
            self.server_command = shlex.split(server_command, posix=False)
        else:
            self.server_command = list(server_command) if server_command is not None else None
        self.server_cwd = Path(server_cwd).resolve() if server_cwd is not None else Path(__file__).resolve().parent
        self.request_timeout = request_timeout
        self.startup_timeout = startup_timeout

        self.session = requests.Session()
        self.process: Optional[subprocess.Popen] = None
        self.last_response: Optional[JsonDict] = None
        self.last_observation: Optional[JsonDict] = None
        self.last_full_state: Optional[JsonDict] = None
        self.last_legal_actions: List[JsonDict] = []
        self.history: List[JsonDict] = []
        self._finalizer: Optional[weakref.finalize] = None

    def __enter__(self) -> "SaboteurEnv":
        self.make()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def make(self, start_server: Optional[bool] = None) -> "SaboteurEnv":
        """
        Ensure the Java server is reachable.

        If `start_server` is true, `server_command` is launched first. If it is
        omitted, the server is launched only when `server_command` was provided.
        """
        should_start = self.server_command is not None if start_server is None else start_server
        if should_start:
            if self.server_command is None:
                self.server_command = default_server_command(self.server_cwd)
            if self.process is None or self.process.poll() is not None:
                command = self._resolve_server_command()
                popen_kwargs = {
                    "cwd": self.server_cwd,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                    "text": True,
                }
                if os.name == "nt":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    popen_kwargs["start_new_session"] = True
                self.process = subprocess.Popen(command, **popen_kwargs)
                self._finalizer = weakref.finalize(self, self._terminate_process, self.process)
        self.wait_until_ready()
        return self

    def _resolve_server_command(self) -> List[str]:
        """
        Resolve the server executable for Windows-friendly subprocess startup.

        On Windows, PowerShell may find `mvn`, while Python's
        `subprocess.Popen` may need the actual `mvn.cmd` path. On macOS/Linux,
        `mvn` is normally executable directly. This keeps caller code simple:

            SaboteurEnv(server_command=["mvn", "exec:java"])
        """
        if self.server_command is None:
            raise ValueError("server_command is required")

        command = list(self.server_command)
        executable = command[0].strip("\"'")
        command[0] = executable
        executable_path = Path(executable)

        if executable_path.parent != Path("."):
            if executable_path.exists():
                command[0] = str(executable_path)
                return command
            raise FileNotFoundError(f"Server executable not found: {executable}")

        resolved = self._local_executable(executable)
        if resolved is None:
            resolved = shutil.which(executable)
        if resolved is None and not executable.lower().endswith((".exe", ".cmd", ".bat")):
            for suffix in (".cmd", ".bat", ".exe"):
                resolved = self._local_executable(executable + suffix)
                if resolved is None:
                    resolved = shutil.which(executable + suffix)
                if resolved is not None:
                    break

        if resolved is None and executable.lower() in {"mvn", "mvn.cmd"}:
            home_maven = sorted(Path.home().glob("maven/apache-maven-*/bin/mvn.cmd"))
            if home_maven:
                resolved = str(home_maven[-1])

        if resolved is None:
            raise FileNotFoundError(
                "Could not find server executable "
                f"{executable!r}. Pass a full path, for example "
                r"server_command=[r'C:\Users\Alpaca\maven\apache-maven-3.9.15\bin\mvn.cmd', 'exec:java']"
            )

        command[0] = resolved
        return command

    def _local_executable(self, executable: str) -> Optional[str]:
        local = self.server_cwd / executable
        if local.exists():
            return str(local)
        return None

    def close(self) -> None:
        """Terminate a server process started by this wrapper."""
        self.session.close()
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer()
        elif self.process is not None:
            self._terminate_process(self.process)
        self._finalizer = None
        self.process = None

    @staticmethod
    def _terminate_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    def wait_until_ready(self) -> None:
        """Poll `/health` until the Java server responds or startup times out."""
        deadline = time.monotonic() + self.startup_timeout
        last_error: Optional[BaseException] = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                details = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part)
                raise SaboteurServerError(
                    f"Saboteur server process exited with code {self.process.returncode}"
                    + (f":\n{details}" if details else "")
                )
            try:
                self.health()
                return
            except (requests.RequestException, SaboteurServerError) as exc:
                last_error = exc
                time.sleep(0.25)
        raise TimeoutError(f"Saboteur server was not ready at {self.base_url}") from last_error

    def health(self) -> JsonDict:
        """Return `/health` JSON."""
        return self._request("GET", "/health")

    def reset(self) -> JsonDict:
        """
        Start a new episode and return the initial observation.

        The Java server always creates a 4-player game with Python controlling
        player 3. Reset clears Python-side cumulative history.
        """
        response = self._request("POST", "/reset", json={})
        self._store_response(response, reset_history=True)
        return response["observation"]

    def step(self, action: ActionLike) -> Tuple[JsonDict, int, bool, JsonDict]:
        """
        Apply one controlled-player action.

        Args:
            action: A server action dict or an integer index into
                `self.last_legal_actions`.

        Returns:
            `(observation, reward, done, info)`.
        """
        action_dict = self.resolve_action(action)
        response = self._request("POST", "/step", json=action_dict)
        self._store_response(response, reset_history=False)

        observation = response["observation"]
        reward = int(response["reward"])
        done = bool(response["done"])
        info = {
            "winner": response.get("winner"),
            "fullState": response.get("fullState"),
            "legalActions": response.get("legalActions", []),
            "events": observation.get("events", []),
            "history": list(self.history),
        }
        return observation, reward, done, info

    def observe(self) -> JsonDict:
        """Fetch and return the current controlled-player observation."""
        response = self.state(view="both")
        self.last_observation = response.get("observation")
        self.last_full_state = response.get("fullState")
        if self.last_observation is None:
            raise SaboteurServerError("State response did not include observation")
        return self.last_observation

    def state(self, view: str = "both") -> JsonDict:
        """Fetch `/state`; `view` must be `observation`, `full`, or `both`."""
        return self._request("GET", "/state", params={"view": view})

    def legal_actions(self, refresh: bool = True) -> List[JsonDict]:
        """Return legal actions for player 3."""
        if refresh or not self.last_legal_actions:
            self.last_legal_actions = self._request("GET", "/legal-actions")
        return list(self.last_legal_actions)

    def sample_action(self) -> JsonDict:
        """Return the first legal action. Useful for smoke tests."""
        actions = self.legal_actions(refresh=True)
        if not actions:
            raise SaboteurServerError("No legal actions are available")
        return actions[0]

    def resolve_action(self, action: ActionLike) -> JsonDict:
        """Convert an action index or dict into a server action dict."""
        if isinstance(action, int):
            if not self.last_legal_actions:
                self.legal_actions(refresh=True)
            try:
                return self.last_legal_actions[action]
            except IndexError as exc:
                raise IndexError(f"Action index {action} is out of range") from exc
        return dict(action)

    def render_text(self, observation: Optional[JsonDict] = None) -> str:
        """
        Return a compact text board view.

        Cells with cards are shown as `#`, empty cells as `.`, start as `S`, and
        goal positions as `G`. This is only a debugging helper.
        """
        obs = observation or self.last_observation
        if obs is None:
            raise SaboteurServerError("No observation is available")
        board = obs["board"]
        width = int(board["width"])
        height = int(board["height"])
        grid = [["." for _ in range(width)] for _ in range(height)]
        for cell in board["cells"]:
            if cell.get("hasCard"):
                grid[int(cell["y"])][int(cell["x"])] = "#"
        start = board["start"]
        grid[int(start["y"])][int(start["x"])] = "S"
        for goal in board["goals"].values():
            grid[int(goal["y"])][int(goal["x"])] = "G"
        return "\n".join(" ".join(row) for row in grid)

    def _store_response(self, response: JsonDict, reset_history: bool) -> None:
        self.last_response = response
        self.last_observation = response.get("observation")
        self.last_full_state = response.get("fullState")
        self.last_legal_actions = list(response.get("legalActions", []))

        if reset_history:
            self.history = []
        if self.last_observation is not None:
            self.history.extend(self.last_observation.get("events", []))

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.request_timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise SaboteurServerError(f"Could not reach Saboteur server at {url}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise SaboteurServerError(
                f"Server returned non-JSON response ({response.status_code})"
            ) from exc

        if response.status_code >= 400:
            message = data.get("error", response.text) if isinstance(data, dict) else response.text
            raise SaboteurServerError(f"{response.status_code}: {message}")
        return data


if __name__ == "__main__":
    env = SaboteurEnv(server_command=default_server_command())
    env.make(start_server=True)
    obs = env.reset()
    print(env.render_text(obs))
    print("Legal actions:", len(env.legal_actions(refresh=False)))
    obs, reward, done, info = env.step(0)
    print("Reward:", reward, "Done:", done, "Winner:", info["winner"])
    print("Events:", len(info["events"]))
