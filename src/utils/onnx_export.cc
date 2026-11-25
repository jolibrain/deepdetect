#include "onnx_export.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <poll.h>
#include <fcntl.h>
#include <pthread.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
namespace dd
{
  namespace
  {
    ssize_t write_no_sigpipe(int fd, const char *data, size_t size)
    {
      sigset_t block, previous, pending;
      sigemptyset(&block);
      sigaddset(&block, SIGPIPE);
      pthread_sigmask(SIG_BLOCK, &block, &previous);
      sigpending(&pending);
      ssize_t result = ::write(fd, data, size);
      int error = errno;
      if (result < 0 && error == EPIPE && !sigismember(&pending, SIGPIPE))
        {
          timespec zero{ 0, 0 };
          while (sigtimedwait(&block, nullptr, &zero) < 0 && errno == EINTR)
            {
            }
        }
      pthread_sigmask(SIG_SETMASK, &previous, nullptr);
      errno = error;
      return result;
    }

    std::string json(const rapidjson::Document &doc)
    {
      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      doc.Accept(writer);
      return buffer.GetString();
    }
  }

  OnnxExport::OnnxExport(const std::string &repository,
                         const std::string &python,
                         const std::string &start_json, double timeout,
                         std::function<bool()> cancelled,
                         std::function<void(const std::string &)> warning)
      : _repository(repository), _timeout(timeout),
        _cancelled(std::move(cancelled)), _warning(std::move(warning))
  {
    // A persisted failure must also suppress retries when Python itself could
    // not be launched in the previous invocation.
    rapidjson::Document request;
    request.Parse(start_json.c_str());
    if (request.IsObject() && request.HasMember("resume")
        && request["resume"].IsBool() && request["resume"].GetBool())
      {
        std::ifstream stream(std::filesystem::path(repository)
                             / "onnx/state.json");
        std::string contents((std::istreambuf_iterator<char>(stream)), {});
        rapidjson::Document state;
        state.Parse(contents.c_str());
        if (state.IsObject() && state.HasMember("status")
            && state["status"].IsString()
            && std::string(state["status"].GetString()) == "disabled")
          {
            _disabled = true;
            _failure_reason
                = state.HasMember("error") && state["error"].IsString()
                      ? state["error"].GetString()
                      : "previous export failed";
            return;
          }
      }
    int sockets[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets))
      {
        fail("cannot create ONNX supervisor socket");
        return;
      }
    // Daemons may start with closed standard descriptors. Keep the transport
    // above stderr so dup2/close actions cannot accidentally close stdin/out.
    for (int i = 0; i < 2; ++i)
      if (sockets[i] <= STDERR_FILENO)
        {
          int replacement = ::fcntl(sockets[i], F_DUPFD_CLOEXEC, 3);
          if (replacement < 0)
            {
              ::close(sockets[0]);
              ::close(sockets[1]);
              fail("cannot duplicate ONNX supervisor descriptor");
              return;
            }
          ::close(sockets[i]);
          sockets[i] = replacement;
        }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, sockets[1], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, sockets[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, sockets[0]);
    posix_spawn_file_actions_addclose(&actions, sockets[1]);
    char *argv[]
        = { const_cast<char *>(python.c_str()), const_cast<char *>("-m"),
            const_cast<char *>("deepdetect.onnx_export"),
            const_cast<char *>("--serve"), nullptr };
    pid_t pid;
    int error
        = posix_spawnp(&pid, python.c_str(), &actions, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    ::close(sockets[1]);
    if (error)
      {
        ::close(sockets[0]);
        fail(std::string("cannot launch ONNX supervisor: ")
             + std::strerror(error));
        return;
      }
    _fd = sockets[0];
    ::fcntl(_fd, F_SETFL, ::fcntl(_fd, F_GETFL) | O_NONBLOCK);
    _pid = pid;
    try
      {
        exchange(start_json, _timeout + 30);
      }
    catch (const std::exception &e)
      {
        stop();
        if (!_cancelled())
          fail(e.what());
      }
  }

  std::string OnnxExport::exchange(const std::string &message, double seconds)
  {
    const auto deadline
        = std::chrono::steady_clock::now()
          + std::chrono::milliseconds(static_cast<long long>(seconds * 1000));
    std::string request = message + "\n", response;
    size_t sent = 0;
    while (true)
      {
        if (_cancelled())
          throw std::runtime_error("training cancelled");
        if (std::chrono::steady_clock::now() >= deadline)
          throw std::runtime_error("ONNX supervisor timed out");
        pollfd pfd{
          _fd, static_cast<short>(sent < request.size() ? POLLOUT : POLLIN), 0
        };
        int ready = ::poll(&pfd, 1, 100);
        if (ready < 0 && errno == EINTR)
          continue;
        if (ready < 0)
          throw std::runtime_error("ONNX supervisor poll failed");
        if (!ready)
          continue;
        if (sent < request.size() && (pfd.revents & POLLOUT))
          {
            ssize_t n = write_no_sigpipe(_fd, request.data() + sent,
                                         request.size() - sent);
            if (n < 0 && (errno == EINTR || errno == EAGAIN))
              continue;
            if (n <= 0)
              throw std::runtime_error(
                  std::string("ONNX supervisor disconnected: ")
                  + std::strerror(errno));
            sent += n;
          }
        else if (pfd.revents & POLLIN)
          {
            char buffer[4096];
            ssize_t n = ::read(_fd, buffer, sizeof(buffer));
            if (n < 0 && (errno == EINTR || errno == EAGAIN))
              continue;
            if (n <= 0)
              throw std::runtime_error("ONNX supervisor exited");
            response.append(buffer, n);
            if (response.size() > 1024 * 1024)
              throw std::runtime_error("invalid ONNX response");
            if (response.find('\n') != std::string::npos)
              return response;
          }
        else if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))
          throw std::runtime_error("ONNX supervisor exited");
      }
  }

  void OnnxExport::notify(long iteration,
                          const std::vector<std::string> &paths) noexcept
  {
    _best_iteration = iteration;
    if (_disabled)
      {
        fail(_failure_reason);
        return;
      }
    if (_closed || _fd < 0)
      return;
    try
      {
        rapidjson::Document doc;
        doc.SetObject();
        auto &a = doc.GetAllocator();
        doc.AddMember("command", "notify", a);
        doc.AddMember("iteration", static_cast<int64_t>(iteration), a);
        rapidjson::Value files(rapidjson::kArrayType);
        for (const auto &path : paths)
          files.PushBack(rapidjson::Value(path.c_str(), a), a);
        doc.AddMember("checkpoints", files, a);
        exchange(json(doc), _timeout + 30);
      }
    catch (const std::exception &e)
      {
        stop();
        if (!_cancelled())
          fail(e.what());
      }
  }

  void OnnxExport::finish() noexcept
  {
    if (_closed)
      return;
    if (_fd >= 0)
      try
        {
          exchange("{\"command\":\"close\",\"cancel\":false}",
                   2 * _timeout + 30);
        }
      catch (const std::exception &e)
        {
          stop();
          if (!_cancelled())
            fail(e.what());
        }
    stop();
  }

  void OnnxExport::fail(const std::string &reason) noexcept
  {
    const bool warn = !_disabled;
    _disabled = true;
    _failure_reason = reason;
    try
      {
        std::filesystem::path root
            = std::filesystem::path(_repository) / "onnx";
        std::filesystem::create_directories(root);
        rapidjson::Document doc;
        std::ifstream previous(root / "state.json");
        std::string contents((std::istreambuf_iterator<char>(previous)), {});
        doc.Parse(contents.c_str());
        if (!doc.IsObject())
          doc.SetObject();
        auto &a = doc.GetAllocator();
        doc.RemoveMember("status");
        doc.RemoveMember("error");
        doc.AddMember("status", "disabled", a);
        doc.AddMember("error", rapidjson::Value(reason.c_str(), a), a);
        if (_best_iteration >= 0)
          {
            doc.RemoveMember("best_iteration");
            doc.AddMember("best_iteration",
                          static_cast<int64_t>(_best_iteration), a);
          }
        if (!doc.HasMember("exported_iteration"))
          doc.AddMember("exported_iteration", rapidjson::Value(), a);
        bool stale
            = !doc["exported_iteration"].IsInt64()
              || doc["exported_iteration"].GetInt64() != _best_iteration;
        doc.RemoveMember("stale");
        doc.AddMember("stale", stale, a);
        auto temporary
            = root / (".native-state-" + std::to_string(::getpid()));
        std::ofstream stream(temporary);
        stream << json(doc);
        stream.close();
        if (!stream)
          throw std::runtime_error("state write failed");
        std::filesystem::rename(temporary, root / "state.json");
      }
    catch (...)
      {
      }
    try
      {
        if (warn)
          _warning("ONNX export disabled for this run: " + reason);
      }
    catch (...)
      {
      }
  }

  void OnnxExport::stop() noexcept
  {
    if (_closed)
      return;
    _closed = true;
    if (_fd >= 0)
      {
        ::shutdown(_fd, SHUT_RDWR);
        ::close(_fd);
        _fd = -1;
      }
    if (_pid > 0)
      {
        // EOF lets Python cancel and reap its conversion before exiting.
        for (int i = 0; i < 100; ++i)
          {
            int status;
            pid_t result = ::waitpid(_pid, &status, WNOHANG);
            if (result == _pid || (result < 0 && errno == ECHILD))
              {
                _pid = -1;
                return;
              }
            ::usleep(10000);
          }
        ::kill(_pid, SIGKILL);
        while (::waitpid(_pid, nullptr, 0) < 0 && errno == EINTR)
          {
          }
        _pid = -1;
      }
  }
  OnnxExport::~OnnxExport()
  {
    stop();
  }
}
