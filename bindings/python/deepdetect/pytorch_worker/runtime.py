from __future__ import annotations

import importlib
import importlib.util
import os
import socket
import sys
import threading
import traceback
from types import ModuleType
from typing import Any

from .protocol import ProtocolError, read_frame, write_frame
from .sdk import (
    Cancellation,
    DeepDetectWorkerBase,
    WorkerContext,
    WorkerContractError,
    WorkerReporter,
    WorkerSDKError,
    validate_optional_result_dict,
    validate_prediction_result,
)


class WorkerRuntime:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self._send_lock = threading.Lock()
        self.worker: Any = None
        self.context: dict[str, Any] = {}
        self.cancellation = Cancellation()
        self.train_thread: threading.Thread | None = None
        self.reporter = WorkerReporter(self.event)

    def serve(self) -> int:
        debug("runtime: serving")
        while True:
            try:
                message = read_frame(self.sock)
            except EOFError:
                debug("runtime: socket closed")
                return 0
            except Exception as error:
                self.failure("protocol_error", error)
                return 2

            method = message.get("method")
            message_id = message.get("id")
            params = message.get("params") or {}
            debug(f"runtime: received method={method!r} id={message_id!r}")
            try:
                if method == "hello":
                    debug("runtime: replying to hello")
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
                    debug("runtime: loading worker")
                    self.context = dict(params)
                    self.worker = self._load_worker(params)
                    debug("runtime: configuring worker")
                    result = self._configure_worker(params)
                    self.reply(message_id, result)
                    debug("runtime: worker configured")
                elif method == "train_start":
                    if self.worker is None:
                        raise RuntimeError("worker is not configured")
                    if self.train_thread is not None and self.train_thread.is_alive():
                        raise RuntimeError("training is already running")
                    self.cancellation = Cancellation()
                    self.reply(message_id, {"status": "started"})
                    debug("runtime: training thread starting")
                    self.train_thread = threading.Thread(
                        target=self._run_train,
                        args=(params,),
                        daemon=True,
                    )
                    self.train_thread.start()
                elif method == "train_cancel":
                    debug("runtime: cancellation requested")
                    self.cancellation.requested = True
                    self.reply(message_id, {"status": "cancelling"})
                elif method == "predict":
                    if self.worker is None:
                        raise RuntimeError("worker is not configured")
                    self.reply(message_id, self._predict(params))
                elif method == "shutdown":
                    debug("runtime: shutdown requested")
                    self.reply(message_id, {"status": "shutdown"})
                    return 0
                else:
                    raise ProtocolError(f"unknown method: {method!r}")
            except Exception as error:
                debug(f"runtime: method={method!r} failed: {error}")
                self.reply_error(
                    message_id,
                    self.failure_payload(self.failure_category(error), error),
                )

    def _run_train(self, params: dict[str, Any]) -> None:
        try:
            debug("runtime: worker.train entered")
            result = self.worker.train(
                params,
                reporter=self.reporter,
                cancellation=self.cancellation,
            )
            result = validate_optional_result_dict(
                result, WorkerContractError, "train result"
            )
            self.event("train_result", result or {"status": "finished"})
            debug("runtime: worker.train finished")
        except Exception as error:
            debug(f"runtime: worker.train failed: {error}")
            category = self.failure_category(error)
            if category == "internal_error":
                category = "training_error"
            payload = self.failure_payload(category, error)
            self.event("failure", payload)

    def _configure_worker(self, params: dict[str, Any]) -> dict[str, Any]:
        if isinstance(self.worker, DeepDetectWorkerBase):
            context = WorkerContext.from_configure_params(params)
            result = self.worker.configure(context)
        else:
            result = self.worker.configure(dict(params))
        return validate_optional_result_dict(
            result, WorkerContractError, "configure result"
        )

    def _predict(self, params: dict[str, Any]) -> dict[str, Any]:
        return validate_prediction_result(self.worker.predict(params))

    def _load_worker(self, params: dict[str, Any]) -> Any:
        mllib = params.get("mllib", {}) if isinstance(params, dict) else {}
        module_name = mllib.get("module") or "deepdetect.pytorch_worker.dummy_worker"
        class_name = mllib.get("class") or "DeepDetectWorker"
        debug(f"runtime: importing worker module={module_name!r} class={class_name!r}")
        module = self._import_module(str(module_name), mllib.get("entrypoint"))
        try:
            worker_class = getattr(module, str(class_name))
        except AttributeError as error:
            raise WorkerContractError(
                f"worker class {class_name!r} not found"
            ) from error
        if not callable(worker_class):
            raise WorkerContractError(f"worker class {class_name!r} is not callable")
        worker = worker_class()
        debug("runtime: worker instance created")
        for method in ("configure", "train", "predict"):
            if not callable(getattr(worker, method, None)):
                raise WorkerContractError(f"worker must implement {method}()")
        return worker

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
        if isinstance(error, WorkerSDKError):
            return error.category
        if isinstance(error, (ImportError, ModuleNotFoundError)):
            return "dependency_error"
        if isinstance(error, ProtocolError):
            return "protocol_error"
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


def debug(message: str) -> None:
    if os.environ.get("DEEPDETECT_DEBUG") or os.environ.get("DEEPDETECT_WORKER_DEBUG"):
        print(
            f"[deepdetect-debug][worker-runtime] {message}",
            file=sys.stderr,
            flush=True,
        )


def main() -> int:
    socket_path = os.environ.get("DEEPDETECT_WORKER_SOCKET")
    if not socket_path:
        print("DEEPDETECT_WORKER_SOCKET is not set", file=sys.stderr)
        return 2
    debug(f"runtime: connecting socket={socket_path}")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    try:
        return WorkerRuntime(sock).serve()
    finally:
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
