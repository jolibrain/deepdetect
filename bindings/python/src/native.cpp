#include "deepdetect/runtime.h"

#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_native, module)
{
  module.doc() = "Native DeepDetect runtime";

  py::class_<deepdetect::Runtime, std::shared_ptr<deepdetect::Runtime>>(
      module, "Runtime")
      .def("build_info", &deepdetect::Runtime::build_info,
           py::call_guard<py::gil_scoped_release>())
      .def("info", &deepdetect::Runtime::info, py::arg("request") = "{}",
           py::call_guard<py::gil_scoped_release>())
      .def("create_service", &deepdetect::Runtime::create_service,
           py::call_guard<py::gil_scoped_release>())
      .def("service_info", &deepdetect::Runtime::service_info,
           py::call_guard<py::gil_scoped_release>())
      .def("set_log_level", &deepdetect::Runtime::set_log_level,
           py::arg("level"), py::call_guard<py::gil_scoped_release>())
      .def("set_service_log_level", &deepdetect::Runtime::set_service_log_level,
           py::arg("name"), py::arg("level"),
           py::call_guard<py::gil_scoped_release>())
      .def("delete_service", &deepdetect::Runtime::delete_service,
           py::arg("name"), py::arg("request") = "{}",
           py::call_guard<py::gil_scoped_release>())
      .def("predict", &deepdetect::Runtime::predict,
           py::call_guard<py::gil_scoped_release>())
      .def("train", &deepdetect::Runtime::train,
           py::call_guard<py::gil_scoped_release>())
      .def("training_status", &deepdetect::Runtime::training_status,
           py::call_guard<py::gil_scoped_release>())
      .def("cancel_training", &deepdetect::Runtime::cancel_training,
           py::call_guard<py::gil_scoped_release>());

  module.def("runtime", []() {
    // The process-wide runtime deliberately outlives Python finalization.
    // CUDA may unregister before extension statics are destroyed, making
    // libtorch service destruction unsafe during interpreter teardown.
    static auto *instance
        = new std::shared_ptr<deepdetect::Runtime>(
            std::make_shared<deepdetect::Runtime>());
    return *instance;
  });
}
