#ifndef DD_ONNX_EXPORT_H
#define DD_ONNX_EXPORT_H

#include <functional>
#include <string>
#include <vector>

namespace dd
{
  // Per-training-call bridge to the shared Python supervisor. No model code or
  // ONNX dependency is loaded into the training process.
  class OnnxExport
  {
  public:
    OnnxExport(const std::string &repository, const std::string &python,
               const std::string &start_json, double timeout,
               std::function<bool()> cancelled,
               std::function<void(const std::string &)> warning);
    ~OnnxExport();
    OnnxExport(const OnnxExport &) = delete;
    OnnxExport &operator=(const OnnxExport &) = delete;
    void notify(long iteration,
                const std::vector<std::string> &paths) noexcept;
    void finish() noexcept;

  private:
    std::string exchange(const std::string &message, double seconds);
    void fail(const std::string &reason) noexcept;
    void stop() noexcept;
    int _fd = -1, _pid = -1;
    bool _disabled = false, _closed = false;
    std::string _repository, _failure_reason;
    long _best_iteration = -1;
    double _timeout;
    std::function<bool()> _cancelled;
    std::function<void(const std::string &)> _warning;
  };
}
#endif
