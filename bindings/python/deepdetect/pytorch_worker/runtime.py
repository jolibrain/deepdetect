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
    WorkerConnector,
    WorkerReporter,
    WorkerLaunchError,
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
        self.connector = WorkerConnector(self.request)
        self._next_request_id = -1
        self._pending_condition = threading.Condition()
        self._pending_responses: dict[int, dict[str, Any]] = {}

    def serve(self) -> int:
        debug("runtime: serving")
        while True:
            try:
                message = read_frame(self.sock)
            except EOFError:
                debug("runtime: socket closed")
                self._fail_pending_requests("worker supervisor disconnected")
                return 0
            except Exception as error:
                self._fail_pending_requests(str(error))
                self.failure("protocol_error", error)
                return 2

            method = message.get("method")
            message_id = message.get("id")
            params = message.get("params") or {}
            debug(f"runtime: received method={method!r} id={message_id!r}")
            if self._route_response(message):
                continue
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
                    self.failure_payload(
                        self.failure_category(error), error, method=method
                    ),
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
            payload = self.failure_payload(category, error, method="train")
            self.event("failure", payload)

    def _configure_worker(self, params: dict[str, Any]) -> dict[str, Any]:
        if isinstance(self.worker, DeepDetectWorkerBase):
            context = WorkerContext.from_configure_params(
                params,
                connector=self.connector,
            )
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
            if not isinstance(entrypoint, (str, bytes, os.PathLike)):
                raise WorkerLaunchError(
                    f"worker entrypoint must be a path: {entrypoint!r}"
                )
            if not os.path.exists(entrypoint):
                raise WorkerLaunchError(
                    f"worker entrypoint does not exist: {entrypoint}"
                )
            spec = importlib.util.spec_from_file_location(
                "_deepdetect_worker", entrypoint
            )
            if spec is None or spec.loader is None:
                raise WorkerLaunchError(f"cannot load worker entrypoint: {entrypoint}")
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

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._pending_condition:
            message_id = self._next_request_id
            self._next_request_id -= 1
            self._pending_responses[message_id] = {}
        self.send({"id": message_id, "method": method, "params": params})
        debug(f"runtime: sent request method={method!r} id={message_id!r}")
        with self._pending_condition:
            while not self._pending_responses[message_id]:
                self._pending_condition.wait()
            message = self._pending_responses.pop(message_id)
        error = message.get("error")
        if isinstance(error, dict):
            raise WorkerContractError(
                str(error.get("message") or f"worker request {method} failed")
            )
        result = message.get("result", {})
        if not isinstance(result, dict):
            raise WorkerContractError(f"worker request {method} result must be an object")
        return result

    def _route_response(self, message: dict[str, Any]) -> bool:
        message_id = message.get("id")
        if not isinstance(message_id, int) or message_id >= 0:
            return False
        if "method" in message:
            return False
        if "result" not in message and "error" not in message:
            return False
        with self._pending_condition:
            if message_id not in self._pending_responses:
                return False
            self._pending_responses[message_id] = dict(message)
            self._pending_condition.notify_all()
        debug(f"runtime: routed response id={message_id!r}")
        return True

    def _fail_pending_requests(self, message: str) -> None:
        with self._pending_condition:
            for message_id in list(self._pending_responses):
                self._pending_responses[message_id] = {
                    "error": {
                        "category": "protocol_error",
                        "message": message,
                    }
                }
            self._pending_condition.notify_all()

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
    def failure_payload(
        category: str, error: BaseException, *, method: str | None = None
    ) -> dict[str, Any]:
        payload = {
            "category": category,
            "message": str(error),
            "retryable": False,
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )[-4096:],
        }
        if method:
            payload["method"] = method
        return payload

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
