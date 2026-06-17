/**
 * DeepDetect
 * Copyright (c) 2026 Jolibrain
 *
 * This file is part of deepdetect.
 */

#include "backends/pytorch_worker/pytorchworkersupervisor.h"

#include "utils/utils.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <rapidjson/document.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace dd
{
  namespace
  {
    void close_fd(int &fd)
    {
      if (fd >= 0)
        {
          ::close(fd);
          fd = -1;
        }
    }

    int message_id(const std::string &message)
    {
      rapidjson::Document doc;
      doc.Parse<rapidjson::kParseNanAndInfFlag>(message.c_str());
      if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("id")
          || !doc["id"].IsInt())
        return -1;
      return doc["id"].GetInt();
    }

    bool debug_enabled()
    {
      const char *debug = std::getenv("DEEPDETECT_DEBUG");
      const char *worker_debug = std::getenv("DEEPDETECT_WORKER_DEBUG");
      return (debug && *debug) || (worker_debug && *worker_debug);
    }

    void debug_log(const std::string &message)
    {
      if (debug_enabled())
        std::cerr << "[deepdetect-debug][pytorch-supervisor] " << message
                  << std::endl;
    }
  }

  PytorchWorkerSupervisor::PytorchWorkerSupervisor(
      const std::string &repository, std::shared_ptr<spdlog::logger> logger)
      : _repository(repository), _logger(std::move(logger))
  {
  }

  PytorchWorkerSupervisor::~PytorchWorkerSupervisor()
  {
    try
      {
        shutdown();
      }
    catch (...)
      {
        terminate();
      }
  }

  std::string
  PytorchWorkerSupervisor::python_executable(const APIData &mllib_params) const
  {
    if (mllib_params.has("python"))
      {
        std::string python = mllib_params.get("python").get<std::string>();
        if (!python.empty())
          return python;
      }
    const char *env_python = std::getenv("DEEPDETECT_PYTHON");
    if (env_python && *env_python)
      return env_python;
    return "python3";
  }

  void PytorchWorkerSupervisor::start(const APIData &mllib_params)
  {
    if (running())
      return;
    debug_log("start: creating worker socket");

    char dir_template[] = "/tmp/dd-pytorch-worker-XXXXXX";
    char *created_dir = ::mkdtemp(dir_template);
    if (!created_dir)
      throw MLLibInternalException("failed creating pytorch worker temp dir: "
                                   + std::string(std::strerror(errno)));
    _socket_dir = created_dir;
    _socket_path = _socket_dir + "/worker.sock";
    debug_log("start: socket path " + _socket_path);

    _listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (_listen_fd < 0)
      throw MLLibInternalException("failed creating pytorch worker socket: "
                                   + std::string(std::strerror(errno)));

    sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, _socket_path.c_str(),
                 sizeof(addr.sun_path) - 1);
    if (::bind(_listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr))
        != 0)
      throw MLLibInternalException("failed binding pytorch worker socket: "
                                   + std::string(std::strerror(errno)));
    if (::listen(_listen_fd, 1) != 0)
      throw MLLibInternalException(
          "failed listening on pytorch worker socket: "
          + std::string(std::strerror(errno)));

    int stderr_pipe[2] = { -1, -1 };
    if (::pipe(stderr_pipe) != 0)
      throw MLLibInternalException(
          "failed creating pytorch worker stderr pipe: "
          + std::string(std::strerror(errno)));
    int flags = ::fcntl(stderr_pipe[0], F_GETFL, 0);
    if (flags >= 0)
      ::fcntl(stderr_pipe[0], F_SETFL, flags | O_NONBLOCK);

    std::string python = python_executable(mllib_params);
    debug_log("start: python executable " + python);
    _pid = ::fork();
    if (_pid < 0)
      {
        close_fd(stderr_pipe[0]);
        close_fd(stderr_pipe[1]);
        throw MLLibInternalException("failed forking pytorch worker: "
                                     + std::string(std::strerror(errno)));
      }
    if (_pid == 0)
      {
        close_fd(stderr_pipe[0]);
        ::dup2(stderr_pipe[1], STDERR_FILENO);
        close_fd(stderr_pipe[1]);
        ::setsid();
        ::setenv("DEEPDETECT_WORKER_SOCKET", _socket_path.c_str(), 1);
        ::setenv("DEEPDETECT_REPOSITORY", _repository.c_str(), 1);
        debug_log("child: exec python worker runtime");
        ::execlp(python.c_str(), python.c_str(), "-m",
                 "deepdetect.pytorch_worker.runtime",
                 static_cast<char *>(nullptr));
        std::cerr << "failed executing pytorch worker runtime with " << python
                  << ": " << std::strerror(errno) << std::endl;
        ::_exit(127);
      }
    close_fd(stderr_pipe[1]);
    _stderr_fd = stderr_pipe[0];

    debug_log("start: waiting for worker connection");
    auto start_time = std::chrono::steady_clock::now();
    bool ready = false;
    while (!ready)
      {
        pollfd pfds[2];
        pfds[0].fd = _listen_fd;
        pfds[0].events = POLLIN;
        pfds[0].revents = 0;
        pfds[1].fd = _stderr_fd;
        pfds[1].events = POLLIN;
        pfds[1].revents = 0;
        int nfds = _stderr_fd >= 0 ? 2 : 1;
        int poll_status = ::poll(pfds, nfds, 100);
        if (poll_status < 0 && errno != EINTR)
          {
            terminate();
            throw MLLibInternalException(
                "pytorch worker worker_launch_error: polling launch failed: "
                + std::string(std::strerror(errno)) + diagnostic_suffix());
          }
        if (_stderr_fd >= 0 && nfds > 1
            && (pfds[1].revents & (POLLIN | POLLHUP)))
          read_worker_stderr();
        if (poll_status > 0 && (pfds[0].revents & POLLIN))
          ready = true;
        if (reap_child() && !ready)
          {
            read_worker_stderr();
            terminate();
            throw MLLibInternalException(
                "pytorch worker worker_launch_error: worker exited before "
                "connecting"
                + diagnostic_suffix());
          }
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
        if (elapsed.count() >= 10000)
          {
            read_worker_stderr();
            terminate();
            debug_log("start: worker launch timeout");
            throw MLLibInternalException(
                "pytorch worker worker_launch_error: launch timeout; check "
                "Python environment"
                + diagnostic_suffix());
          }
      }
    _fd = ::accept(_listen_fd, nullptr, nullptr);
    close_fd(_listen_fd);
    if (_fd < 0)
      {
        terminate();
        throw MLLibInternalException("failed accepting pytorch worker socket: "
                                     + std::string(std::strerror(errno)));
      }
    debug_log("start: worker connected");
  }

  bool PytorchWorkerSupervisor::running() const
  {
    return _pid > 0 && _fd >= 0;
  }

  void PytorchWorkerSupervisor::shutdown()
  {
    debug_log("shutdown");
    if (running())
      {
        APIData params;
        try
          {
            request("shutdown", params, 1000);
          }
        catch (...)
          {
          }
      }
    wait_for_exit(1000);
    terminate();
  }

  void PytorchWorkerSupervisor::terminate()
  {
    debug_log("terminate");
    close_fd(_fd);
    close_fd(_listen_fd);
    if (_pid > 0)
      {
        ::kill(-_pid, SIGTERM);
        wait_for_exit(500);
        if (_pid > 0)
          {
            ::kill(-_pid, SIGKILL);
            wait_for_exit(500);
          }
      }
    read_worker_stderr();
    close_fd(_stderr_fd);
    cleanup_socket();
  }

  void PytorchWorkerSupervisor::wait_for_exit(int timeout_ms)
  {
    if (_pid <= 0)
      return;
    auto start = std::chrono::steady_clock::now();
    while (true)
      {
        read_worker_stderr();
        if (reap_child())
          return;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        if (elapsed.count() >= timeout_ms)
          return;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
  }

  bool PytorchWorkerSupervisor::reap_child()
  {
    if (_pid <= 0)
      return true;
    int status = 0;
    pid_t result = ::waitpid(_pid, &status, WNOHANG);
    if (result == _pid)
      {
        if (WIFEXITED(status))
          _exit_code = WEXITSTATUS(status);
        if (WIFSIGNALED(status))
          _signal = WTERMSIG(status);
        _pid = -1;
        return true;
      }
    if (result < 0 && errno == ECHILD)
      {
        _pid = -1;
        return true;
      }
    return false;
  }

  void PytorchWorkerSupervisor::read_worker_stderr()
  {
    if (_stderr_fd < 0)
      return;
    char buffer[4096];
    while (true)
      {
        ssize_t n = ::read(_stderr_fd, buffer, sizeof(buffer));
        if (n > 0)
          {
            append_stderr(buffer, static_cast<size_t>(n));
            continue;
          }
        if (n < 0 && errno == EINTR)
          continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
          return;
        return;
      }
  }

  void PytorchWorkerSupervisor::append_stderr(const char *data, size_t size)
  {
    constexpr size_t max_tail = 8192;
    _stderr_tail.append(data, size);
    if (_stderr_tail.size() > max_tail)
      _stderr_tail.erase(0, _stderr_tail.size() - max_tail);
  }

  std::string PytorchWorkerSupervisor::diagnostic_suffix() const
  {
    std::string suffix;
    if (_exit_code >= 0)
      suffix += " exit_code=" + std::to_string(_exit_code);
    if (_signal >= 0)
      suffix += " signal=" + std::to_string(_signal);
    if (!_stderr_tail.empty())
      suffix += " stderr_tail=" + _stderr_tail;
    return suffix;
  }

  void PytorchWorkerSupervisor::cleanup_socket()
  {
    if (!_socket_path.empty())
      ::unlink(_socket_path.c_str());
    if (!_socket_dir.empty())
      ::rmdir(_socket_dir.c_str());
    _socket_path.clear();
    _socket_dir.clear();
  }

  std::string
  PytorchWorkerSupervisor::make_request_json(int id, const std::string &method,
                                             const APIData &params) const
  {
    std::string params_json = params.toJSONString();
    return "{\"id\":" + std::to_string(id) + ",\"method\":\"" + method
           + "\",\"params\":" + params_json + "}";
  }

  void PytorchWorkerSupervisor::send_request(const std::string &method,
                                             const APIData &params,
                                             int &request_id)
  {
    if (!running())
      throw MLLibInternalException("pytorch worker is not running");
    request_id = _next_id++;
    debug_log("send request id=" + std::to_string(request_id)
              + " method=" + method);
    send_frame(make_request_json(request_id, method, params));
  }

  APIData PytorchWorkerSupervisor::request(const std::string &method,
                                           const APIData &params,
                                           int timeout_ms)
  {
    int id = -1;
    send_request(method, params, id);
    std::string message;
    while (read_message(message, timeout_ms))
      {
        if (message_id(message) == id)
          {
            debug_log("received response id=" + std::to_string(id)
                      + " method=" + method);
            std::string error_message = response_error_message(message);
            if (!error_message.empty())
              throw MLLibInternalException(error_message);
            rapidjson::Document doc;
            doc.Parse<rapidjson::kParseNanAndInfFlag>(message.c_str());
            APIData out;
            out.fromRapidJson(doc);
            return out;
          }
        _pending.push_back(message);
        debug_log("queued async worker event while waiting for method="
                  + method);
      }
    debug_log("timeout waiting for method=" + method);
    throw MLLibInternalException(
        "pytorch worker timeout_error: timeout waiting for response"
        + diagnostic_suffix());
  }

  std::string PytorchWorkerSupervisor::response_error_message(
      const std::string &message) const
  {
    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseNanAndInfFlag>(message.c_str());
    if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("error"))
      return "";
    const auto &error = doc["error"];
    std::string category = "internal_error";
    std::string msg = "worker returned an error";
    std::string method;
    if (error.IsObject())
      {
        if (error.HasMember("category") && error["category"].IsString())
          category = error["category"].GetString();
        if (error.HasMember("message") && error["message"].IsString())
          msg = error["message"].GetString();
        if (error.HasMember("method") && error["method"].IsString())
          method = error["method"].GetString();
      }
    std::string out = "pytorch worker " + category + ": ";
    if (!method.empty())
      out += method + ": ";
    out += msg;
    out += diagnostic_suffix();
    return out;
  }

  bool PytorchWorkerSupervisor::read_message(std::string &message,
                                             int timeout_ms)
  {
    if (!_pending.empty())
      {
        message = _pending.front();
        _pending.pop_front();
        debug_log("read queued worker message");
        return true;
      }
    read_worker_stderr();
    return read_frame(message, timeout_ms);
  }

  void PytorchWorkerSupervisor::send_frame(const std::string &message)
  {
    uint32_t size = htonl(static_cast<uint32_t>(message.size()));
    const char *header = reinterpret_cast<const char *>(&size);
    size_t written = 0;
    while (written < sizeof(size))
      {
        ssize_t n = ::write(_fd, header + written, sizeof(size) - written);
        if (n < 0 && errno == EINTR)
          continue;
        if (n <= 0)
          throw MLLibInternalException("failed writing pytorch worker frame");
        written += static_cast<size_t>(n);
      }
    written = 0;
    while (written < message.size())
      {
        ssize_t n
            = ::write(_fd, message.data() + written, message.size() - written);
        if (n < 0 && errno == EINTR)
          continue;
        if (n <= 0)
          throw MLLibInternalException("failed writing pytorch worker frame");
        written += static_cast<size_t>(n);
      }
  }

  bool PytorchWorkerSupervisor::read_frame(std::string &message,
                                           int timeout_ms)
  {
    if (!running())
      return false;
    pollfd pfd;
    pfd.fd = _fd;
    pfd.events = POLLIN;
    int poll_status = ::poll(&pfd, 1, timeout_ms);
    read_worker_stderr();
    if (poll_status == 0)
      return false;
    if (poll_status < 0)
      {
        if (errno == EINTR)
          return false;
        throw MLLibInternalException("polling pytorch worker failed: "
                                     + std::string(std::strerror(errno)));
      }
    uint32_t network_size = 0;
    if (!read_exact(&network_size, sizeof(network_size)))
      {
        read_worker_stderr();
        throw MLLibInternalException(disconnected_message());
      }
    uint32_t size = ntohl(network_size);
    if (size == 0 || size > 64 * 1024 * 1024)
      throw MLLibInternalException("invalid pytorch worker frame size");
    std::string payload(size, '\0');
    if (!read_exact(payload.data(), payload.size()))
      {
        read_worker_stderr();
        throw MLLibInternalException(disconnected_message());
      }
    message = std::move(payload);
    return true;
  }

  std::string PytorchWorkerSupervisor::disconnected_message() const
  {
    return "pytorch worker internal_error: disconnected" + diagnostic_suffix();
  }

  bool PytorchWorkerSupervisor::read_exact(void *buffer, size_t size)
  {
    char *dst = static_cast<char *>(buffer);
    size_t read_total = 0;
    while (read_total < size)
      {
        ssize_t n = ::read(_fd, dst + read_total, size - read_total);
        if (n < 0 && errno == EINTR)
          continue;
        if (n <= 0)
          return false;
        read_total += static_cast<size_t>(n);
      }
    return true;
  }
}
