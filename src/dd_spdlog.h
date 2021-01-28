
#ifdef USE_DD_SYSLOG
#define SPDLOG_ENABLE_SYSLOG
#endif

#include <spdlog/spdlog.h>

#if SPDLOG_VER_MAJOR > 0
#ifdef USE_DD_SYSLOG
#include <spdlog/sinks/syslog_sink.h>
#define DD_SPDLOG_LOGGER spdlog::syslog_logger_mt
#else
#include <spdlog/sinks/stdout_sinks.h>
#define DD_SPDLOG_LOGGER spdlog::stdout_logger_mt
#endif
#else
#ifdef USE_DD_SYSLOG
#define DD_SPDLOG_LOGGER spdlog::syslog_logger
#else
#define DD_SPDLOG_LOGGER spdlog::stdout_logger_mt
#endif
#endif
