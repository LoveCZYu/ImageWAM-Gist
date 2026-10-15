import base64
import json
import os
import socket
import time
from typing import Any, Dict, Optional

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def _decode_numpy(payload: str) -> Any:
    def object_hook(value):
        if "__numpy_array__" not in value:
            return value
        raw = base64.b64decode(value["data"])
        return np.frombuffer(raw, dtype=value["dtype"]).reshape(value["shape"])

    return json.loads(payload, object_hook=object_hook)


class ImageWAMRemoteClient:
    def __init__(self, host: str, port: int, timeout: float = 600.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.socket: Optional[socket.socket] = None
        self._connect()

    def _connect(self) -> None:
        last_error = None
        for _ in range(240):
            candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            candidate.settimeout(self.timeout)
            try:
                candidate.connect((self.host, self.port))
                self.socket = candidate
                print(f"[imagewam-client] connected to {self.host}:{self.port}", flush=True)
                return
            except OSError as exc:
                last_error = exc
                candidate.close()
                time.sleep(1.0)
        raise ConnectionError(f"Could not connect to ImageWAM server: {last_error}")

    def call(self, command: str, payload: Any = None) -> Any:
        if self.socket is None:
            raise ConnectionError("ImageWAM server socket is closed.")
        request = json.dumps(
            {"cmd": command, "obs": payload},
            cls=_NumpyEncoder,
        ).encode("utf-8")
        self.socket.sendall(len(request).to_bytes(4, "big"))
        self.socket.sendall(request)
        header = self._recv_exact(4)
        response = _decode_numpy(self._recv_exact(int.from_bytes(header, "big")).decode("utf-8"))
        if "error" in response:
            raise RuntimeError(f"{response['error']}\n{response.get('traceback', '')}")
        return response["res"]

    def _recv_exact(self, length: int) -> bytes:
        if self.socket is None:
            raise ConnectionError("ImageWAM server socket is closed.")
        chunks = []
        remaining = int(length)
        while remaining:
            chunk = self.socket.recv(remaining)
            if not chunk:
                raise ConnectionError("ImageWAM server closed the connection.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def get_model(usr_args: Dict[str, Any]) -> ImageWAMRemoteClient:
    return ImageWAMRemoteClient(
        host=str(usr_args.get("server_host", "127.0.0.1")),
        port=int(usr_args["port"]),
        timeout=float(usr_args.get("server_timeout", 600.0)),
    )


def eval(task_env, model: ImageWAMRemoteClient, observation: Dict[str, Any]) -> None:
    instruction = os.environ.get("IMAGEWAM_INSTRUCTION_OVERRIDE") or task_env.get_instruction()
    action = model.call(
        "remote_step",
        {
            "observation": observation,
            "instruction": instruction,
        },
    )
    if action is not None:
        task_env.take_action(np.asarray(action, dtype=np.float32), action_type="qpos")


def reset_model(model: ImageWAMRemoteClient) -> None:
    model.call("reset_model")
