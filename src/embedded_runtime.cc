#include "deepdetect/runtime.h"

#include "dd_config.h"
#include "dd_spdlog.h"
#include "jsonapi.h"

#include <exception>
#include <mutex>
#include <shared_mutex>
#include <utility>

namespace
{
  std::string json_escape(const std::string &value)
  {
    std::string escaped;
    escaped.reserve(value.size());
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char c : value)
      {
        switch (c)
          {
          case '"':
            escaped += "\\\"";
            break;
          case '\\':
            escaped += "\\\\";
            break;
          case '\n':
            escaped += "\\n";
            break;
          case '\r':
            escaped += "\\r";
            break;
          case '\t':
            escaped += "\\t";
            break;
          default:
            if (c < 0x20)
              {
                escaped += "\\u00";
                escaped += hex[c >> 4];
                escaped += hex[c & 0x0f];
              }
            else
              {
                escaped += static_cast<char>(c);
              }
          }
      }
    return escaped;
  }

  std::string exception_response(const std::string &message)
  {
    return "{\"status\":{\"code\":500,\"msg\":\"InternalError\","
           "\"dd_code\":500,\"dd_msg\":\""
           + json_escape(message) + "\"}}";
  }

  template <typename F> std::string contain_exceptions(F &&call) noexcept
  {
    try
      {
        return call();
      }
    catch (const std::exception &error)
      {
        return exception_response(error.what());
      }
    catch (...)
      {
        return exception_response("unexpected native exception");
      }
  }

  bool log_level_from_string(const std::string &value,
                             spdlog::level::level_enum &level)
  {
    if (value == "trace")
      level = spdlog::level::trace;
    else if (value == "debug")
      level = spdlog::level::debug;
    else if (value == "info")
      level = spdlog::level::info;
    else if (value == "warn" || value == "warning")
      level = spdlog::level::warn;
    else if (value == "error" || value == "err")
      level = spdlog::level::err;
    else if (value == "critical")
      level = spdlog::level::critical;
    else if (value == "off")
      level = spdlog::level::off;
    else
      return false;
    return true;
  }
}

namespace deepdetect
{
  class Runtime::Impl
  {
  public:
    mutable std::shared_mutex mutex;
    dd::JsonAPI api;
  };

  Runtime::Runtime() : _impl(new Impl)
  {
  }

  Runtime::~Runtime() = default;
  Runtime::Runtime(Runtime &&) noexcept = default;
  Runtime &Runtime::operator=(Runtime &&) noexcept = default;

  std::string Runtime::build_info() const noexcept
  {
    return contain_exceptions([&]() {
      std::shared_lock<std::shared_mutex> lock(_impl->mutex);
#ifdef CPU_ONLY
      constexpr bool cuda_enabled = false;
#else
      constexpr bool cuda_enabled = true;
#endif
#ifdef USE_CUDNN
      constexpr bool cudnn_enabled = true;
#else
      constexpr bool cudnn_enabled = false;
#endif
      return std::string("{\"version\":\"") + json_escape(GIT_VERSION)
             + "\",\"commit\":\"" + json_escape(GIT_COMMIT_HASH)
             + "\",\"branch\":\"" + json_escape(GIT_BRANCH)
             + "\",\"build_type\":\"" + json_escape(BUILD_TYPE)
             + "\",\"compile_flags\":\"" + json_escape(COMPILE_FLAGS)
             + "\",\"dependency_versions\":\"" + json_escape(DEPS_VERSION)
             + "\",\"torch_enabled\":\"" + json_escape(DD_TORCH_ENABLED)
             + "\",\"torch_prebuilt\":\"" + json_escape(DD_TORCH_PREBUILT)
             + "\",\"torch_version\":\"" + json_escape(DD_TORCH_VERSION)
             + "\",\"torch_mode\":\"" + json_escape(DD_TORCH_MODE)
             + "\",\"cuda\":" + (cuda_enabled ? "true" : "false")
             + ",\"cudnn\":" + (cudnn_enabled ? "true" : "false") + "}";
    });
  }

  std::string Runtime::info(const std::string &request) const noexcept
  {
    return contain_exceptions([&]() {
      std::shared_lock<std::shared_mutex> lock(_impl->mutex);
      return _impl->api.jrender(_impl->api.info(request));
    });
  }

  std::string Runtime::create_service(const std::string &name,
                                      const std::string &request) noexcept
  {
    return contain_exceptions([&]() {
      std::unique_lock<std::shared_mutex> lock(_impl->mutex);
      return _impl->api.jrender(_impl->api.service_create(name, request));
    });
  }

  std::string Runtime::service_info(const std::string &name) const noexcept
  {
    return contain_exceptions([&]() {
      std::shared_lock<std::shared_mutex> lock(_impl->mutex);
      return _impl->api.jrender(_impl->api.service_status(name));
    });
  }

  std::string Runtime::set_log_level(const std::string &level) noexcept
  {
    return contain_exceptions([&]() {
      std::unique_lock<std::shared_mutex> lock(_impl->mutex);
      spdlog::level::level_enum parsed_level;
      if (!log_level_from_string(level, parsed_level))
        return std::string("{\"status\":{\"code\":400,\"msg\":\"BadRequest\",")
               + "\"dd_code\":400,\"dd_msg\":\"invalid log level: "
               + json_escape(level) + "\"}}";

      spdlog::set_level(parsed_level);
      return std::string("{\"status\":{\"code\":200,\"msg\":\"OK\"}}");
    });
  }

  std::string Runtime::set_service_log_level(const std::string &name,
                                             const std::string &level) noexcept
  {
    return contain_exceptions([&]() {
      std::unique_lock<std::shared_mutex> lock(_impl->mutex);
      auto logger = spdlog::get(name);
      if (logger == nullptr)
        return std::string("{\"status\":{\"code\":404,\"msg\":\"NotFound\",")
               + "\"dd_code\":404,\"dd_msg\":\"service logger not found: "
               + json_escape(name) + "\"}}";

      spdlog::level::level_enum parsed_level;
      if (!log_level_from_string(level, parsed_level))
        return std::string("{\"status\":{\"code\":400,\"msg\":\"BadRequest\",")
               + "\"dd_code\":400,\"dd_msg\":\"invalid log level: "
               + json_escape(level) + "\"}}";

      logger->set_level(parsed_level);
      return std::string("{\"status\":{\"code\":200,\"msg\":\"OK\"}}");
    });
  }

  std::string Runtime::delete_service(const std::string &name,
                                      const std::string &request) noexcept
  {
    return contain_exceptions([&]() {
      std::unique_lock<std::shared_mutex> lock(_impl->mutex);
      return _impl->api.jrender(_impl->api.service_delete(name, request));
    });
  }

  std::string Runtime::predict(const std::string &request) noexcept
  {
    return contain_exceptions([&]() {
      std::shared_lock<std::shared_mutex> lock(_impl->mutex);
      return _impl->api.jrender(_impl->api.service_predict(request));
    });
  }

  std::string Runtime::train(const std::string &request) noexcept
  {
    return contain_exceptions([&]() {
      std::shared_lock<std::shared_mutex> lock(_impl->mutex);
      return _impl->api.jrender(_impl->api.service_train(request));
    });
  }

  std::string Runtime::training_status(const std::string &request) noexcept
  {
    return contain_exceptions([&]() {
      std::shared_lock<std::shared_mutex> lock(_impl->mutex);
      return _impl->api.jrender(_impl->api.service_train_status(request));
    });
  }

  std::string Runtime::cancel_training(const std::string &request) noexcept
  {
    return contain_exceptions([&]() {
      std::shared_lock<std::shared_mutex> lock(_impl->mutex);
      return _impl->api.jrender(_impl->api.service_train_delete(request));
    });
  }
}
