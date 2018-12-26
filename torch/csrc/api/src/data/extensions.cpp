#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <torch/data/detail/queue.h>

template class torch::data::detail::Queue<int>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<torch::data::detail::Queue<int>, std::shared_ptr<torch::data::detail::Queue<int>>>(m, "DataQueue")
      .def(py::init<>())
      .def("push", &torch::data::detail::Queue<int>::push)
      .def("pop", &torch::data::detail::Queue<int>::pop);
}
