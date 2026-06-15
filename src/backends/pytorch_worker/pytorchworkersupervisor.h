/**
 * DeepDetect
 * Copyright (c) 2026 Jolibrain
 *
 * This file is part of deepdetect.
 */

#ifndef PYTORCHWORKERSUPERVISOR_H
#define PYTORCHWORKERSUPERVISOR_H

#include "apidata.h"
#include "dd_spdlog.h"
#include "mllibstrategy.h"

#include <deque>
#include <memory>
#include <string>
#include <sys/types.h>

namespace dd
{
  class PytorchWorkerSupervisor
  {
  public:
    PytorchWorkerSupervisor(const std::string &repository,
                            std::shared_ptr<spdlog::logger> logger);
    ~PytorchWorkerSupervisor();

    PytorchWorkerSupervisor(const PytorchWorkerSupervisor &) = delete;
    PytorchWorkerSupervisor &
    operator=(const PytorchWorkerSupervisor &) = delete;

    void start(const APIData &mllib_params);
    void shutdown();
    void terminate();

    APIData request(const std::string &method, const APIData &params,
                    int timeout_ms = 10000);
    void send_request(const std::string &method, const APIData &params,
                      int &request_id);
    bool read_message(std::string &message, int timeout_ms);

    bool running() const;

  private:
    std::string python_executable(const APIData &mllib_params) const;
    std::string make_request_json(int id, const std::string &method,
                                  const APIData &params) const;
    void send_frame(const std::string &message);
    bool read_frame(std::string &message, int timeout_ms);
    bool read_exact(void *buffer, size_t size);
    void cleanup_socket();
    void wait_for_exit(int timeout_ms);

    std::string _repository;
    std::shared_ptr<spdlog::logger> _logger;
    pid_t _pid = -1;
    int _fd = -1;
    int _listen_fd = -1;
    int _next_id = 1;
    std::string _socket_dir;
    std::string _socket_path;
    std::deque<std::string> _pending;
  };
}

#endif
