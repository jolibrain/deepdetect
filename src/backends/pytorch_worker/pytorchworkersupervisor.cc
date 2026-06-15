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

    void throw_response_error(const std::string &message)
    {
      rapidjson::Document doc;
      doc.Parse<rapidjson::kParseNanAndInfFlag>(message.c_str());
      if (!doc.HasParseError() && doc.IsObject() && doc.HasMember("error"))
        {
          const auto &error = doc["error"];
          std::string category = "internal_error";
          std::string msg = "worker returned an error";
          if (error.IsObject())
            {
              if (error.HasMember("category") && error["category"].IsString())
                category = error["category"].GetString();
              if (error.HasMember("message") && error["message"].IsString())
                msg = error["message"].GetString();
            }
          throw MLLibInternalException("pytorch worker " + category + ": "
                                      + msg);
        }
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

    char dir_template[] = "/tmp/dd-pytorch-worker-XXXXXX";
    char *created_dir = ::mkdtemp(dir_template);
    if (!created_dir)
      throw MLLibInternalException("failed creating pytorch worker temp dir: "
                                  + std::string(std::strerror(errno)));
    _socket_dir = created_dir;
    _socket_path = _socket_dir + "/worker.sock";

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
      throw MLLibInternalException("failed listening on pytorch worker socket: "
                                  + std::string(std::strerror(errno)));

    std::string python = python_executable(mllib_params);
    _pid = ::fork();
    if (_pid < 0)
      throw MLLibInternalException("failed forking pytorch worker: "
                                  + std::string(std::strerror(errno)));
    if (_pid == 0)
      {
        ::setsid();
        ::setenv("DEEPDETECT_WORKER_SOCKET", _socket_path.c_str(), 1);
        ::setenv("DEEPDETECT_REPOSITORY", _repository.c_str(), 1);
        ::execlp(python.c_str(), python.c_str(), "-m",
                 "deepdetect.pytorch_worker.runtime",
                 static_cast<char *>(nullptr));
        ::_exit(127);
      }

    pollfd pfd;
    pfd.fd = _listen_fd;
    pfd.events = POLLIN;
    int poll_status = ::poll(&pfd, 1, 10000);
    if (poll_status <= 0)
      {
        terminate();
        throw MLLibInternalException(
            "pytorch worker launch timeout; check Python environment");
      }
    _fd = ::accept(_listen_fd, nullptr, nullptr);
    close_fd(_listen_fd);
    if (_fd < 0)
      {
        terminate();
        throw MLLibInternalException("failed accepting pytorch worker socket: "
                                    + std::string(std::strerror(errno)));
      }
  }

  bool PytorchWorkerSupervisor::running() const
  {
    return _pid > 0 && _fd >= 0;
  }

  void PytorchWorkerSupervisor::shutdown()
  {
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
    cleanup_socket();
  }

  void PytorchWorkerSupervisor::wait_for_exit(int timeout_ms)
  {
    if (_pid <= 0)
      return;
    auto start = std::chrono::steady_clock::now();
    while (true)
      {
        int status = 0;
        pid_t result = ::waitpid(_pid, &status, WNOHANG);
        if (result == _pid)
          {
            _pid = -1;
            return;
          }
        if (result < 0 && errno == ECHILD)
          {
            _pid = -1;
            return;
          }
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        if (elapsed.count() >= timeout_ms)
          return;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
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

  std::string PytorchWorkerSupervisor::make_request_json(
      int id, const std::string &method, const APIData &params) const
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
            throw_response_error(message);
            rapidjson::Document doc;
            doc.Parse<rapidjson::kParseNanAndInfFlag>(message.c_str());
            APIData out;
            out.fromRapidJson(doc);
            return out;
          }
        _pending.push_back(message);
      }
    throw MLLibInternalException("timeout waiting for pytorch worker response");
  }

  bool PytorchWorkerSupervisor::read_message(std::string &message,
                                             int timeout_ms)
  {
    if (!_pending.empty())
      {
        message = _pending.front();
        _pending.pop_front();
        return true;
      }
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
        ssize_t n = ::write(_fd, message.data() + written,
                            message.size() - written);
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
      throw MLLibInternalException("pytorch worker disconnected");
    uint32_t size = ntohl(network_size);
    if (size == 0 || size > 64 * 1024 * 1024)
      throw MLLibInternalException("invalid pytorch worker frame size");
    std::string payload(size, '\0');
    if (!read_exact(payload.data(), payload.size()))
      throw MLLibInternalException("pytorch worker disconnected");
    message = std::move(payload);
    return true;
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
