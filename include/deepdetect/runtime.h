#ifndef DEEPDETECT_RUNTIME_H
#define DEEPDETECT_RUNTIME_H

#include <memory>
#include <string>

namespace deepdetect
{
  class Runtime
  {
  public:
    Runtime();
    ~Runtime();

    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;
    Runtime(Runtime &&) noexcept;
    Runtime &operator=(Runtime &&) noexcept;

    std::string build_info() const noexcept;
    std::string info(const std::string &request = "{}") const noexcept;
    std::string create_service(const std::string &name,
                               const std::string &request) noexcept;
    std::string service_info(const std::string &name) const noexcept;
    std::string delete_service(const std::string &name,
                               const std::string &request = "{}") noexcept;
    std::string predict(const std::string &request) noexcept;
    std::string train(const std::string &request) noexcept;
    std::string training_status(const std::string &request) noexcept;
    std::string cancel_training(const std::string &request) noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> _impl;
  };
}

#endif
