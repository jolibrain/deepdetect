from __future__ import annotations

import importlib
import importlib.util
import os
import socket
import sys
import threading
import traceback
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from .protocol import ProtocolError, read_frame, write_frame


@dataclass
class Cancellation:
    requested: bool = False


class Reporter:
    def __init__(self, runtime: "WorkerRuntime") -> None:
        self._runtime = runtime

    def status(self, **payload: Any) -> None:
        self._runtime.event("status", payload)

    def metric(self, name: str, value: Any, *, iteration: Any = None) -> None:
        payload = {"name": name, "value": value}
        if iteration is not None:
            payload["iteration"] = iteration
        self._runtime.event("metric", payload)

    def artifact(self, **payload: Any) -> None:
        self._runtime.event("artifact", payload)

    def log(self, level: str, message: str, **payload: Any) -> None:
        self._runtime.event("log", {"level": level, "message": message, **payload})


class WorkerRuntime:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self._send_lock = threading.Lock()
        self.worker: Any = None
        self.context: dict[str, Any] = {}
        self.cancellation = Cancellation()
        self.train_thread: threading.Thread | None = None
        self.reporter = Reporter(self)

    def serve(self) -> int:
        while True:
            try:
                message = read_frame(self.sock)
            except EOFError:
                return 0
            except Exception as error:
                self.failure("protocol_error", error)
                return 2

            method = message.get("method")
            message_id = message.get("id")
            params = message.get("params") or {}
            try:
                if method == "hello":
                    self.reply(
                        message_id,
                        {
                            "protocol_version": 1,
                            "capabilities": {
                                "train": True,
                                "predict": True,
                                "task": "dummy",
                                "metric_frequency": "optimizer_step",
                                "template": "dummy",
                                "output_format": "deepdetect_supervised_v1",
                                "cpu_tensor_input": False,
                                "gpu_tensor_input_reserved": True,
                                "distributed_launcher": False,
                            },
                        },
                    )
                elif method == "configure":
                    self.context = dict(params)
                    self.worker = self._load_worker(params)
                    result = self.worker.configure(self.context)
                    self.reply(message_id, result or {})
                elif method == "train_start":
                    if self.worker is None:
                        raise RuntimeError("worker is not configured")
                    if self.train_thread is not None and self.train_thread.is_alive():
                        raise RuntimeError("training is already running")
                    self.cancellation = Cancellation()
                    self.reply(message_id, {"status": "started"})
                    self.train_thread = threading.Thread(
                        target=self._run_train,
                        args=(params,),
                        daemon=True,
                    )
                    self.train_thread.start()
                elif method == "train_cancel":
                    self.cancellation.requested = True
                    self.reply(message_id, {"status": "cancelling"})
                elif method == "predict":
                    if self.worker is None:
                        raise RuntimeError("worker is not configured")
                    self.reply(message_id, self.worker.predict(params))
                elif method == "shutdown":
                    self.reply(message_id, {"status": "shutdown"})
                    return 0
                else:
                    raise ProtocolError(f"unknown method: {method!r}")
            except Exception as error:
                self.reply_error(
                    message_id,
                    self.failure_payload(self.failure_category(error), error),
                )

    def _run_train(self, params: dict[str, Any]) -> None:
        try:
            result = self.worker.train(
                params,
                reporter=self.reporter,
                cancellation=self.cancellation,
            )
            self.event("train_result", result or {"status": "finished"})
        except Exception as error:
            payload = self.failure_payload("training_error", error)
            self.event("failure", payload)

    def _load_worker(self, params: dict[str, Any]) -> Any:
        mllib = params.get("mllib", {}) if isinstance(params, dict) else {}
        module_name = mllib.get("module") or "deepdetect.pytorch_worker.dummy_worker"
        class_name = mllib.get("class") or "DeepDetectWorker"
        module = self._import_module(str(module_name), mllib.get("entrypoint"))
        try:
            worker_class = getattr(module, str(class_name))
        except AttributeError as error:
            raise RuntimeError(f"worker class {class_name!r} not found") from error
        return worker_class()

    @staticmethod
    def _import_module(module_name: str, entrypoint: Any = None) -> ModuleType:
        if entrypoint:
            spec = importlib.util.spec_from_file_location(
                "_deepdetect_worker", entrypoint
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load worker entrypoint: {entrypoint}")
            module = importlib.util.module_from_spec(spec)
            sys.modules["_deepdetect_worker"] = module
            spec.loader.exec_module(module)
            return module
        return importlib.import_module(module_name)

    def reply(self, message_id: Any, result: dict[str, Any]) -> None:
        self.send({"id": message_id, "result": result})

    def reply_error(self, message_id: Any, error: dict[str, Any]) -> None:
        self.send({"id": message_id, "error": error})

    def event(self, name: str, payload: dict[str, Any]) -> None:
        self.send({"event": name, "payload": payload})

    def failure(self, category: str, error: BaseException) -> None:
        self.event("failure", self.failure_payload(category, error))

    @staticmethod
    def failure_category(error: BaseException) -> str:
        if isinstance(error, (ImportError, ModuleNotFoundError)):
            return "dependency_error"
        if isinstance(error, ProtocolError):
            return "protocol_error"
        if isinstance(error, RuntimeError) and "worker class" in str(error):
            return "worker_contract_error"
        return "internal_error"

    @staticmethod
    def failure_payload(category: str, error: BaseException) -> dict[str, Any]:
        return {
            "category": category,
            "message": str(error),
            "retryable": False,
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )[-4096:],
        }

    def send(self, message: dict[str, Any]) -> None:
        with self._send_lock:
            write_frame(self.sock, message)


def main() -> int:
    socket_path = os.environ.get("DEEPDETECT_WORKER_SOCKET")
    if not socket_path:
        print("DEEPDETECT_WORKER_SOCKET is not set", file=sys.stderr)
        return 2
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    try:
        return WorkerRuntime(sock).serve()
    finally:
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
