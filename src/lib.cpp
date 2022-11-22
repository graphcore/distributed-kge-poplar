// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
//
// Python bindings

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "poplar_kge.hpp"

namespace py = pybind11;

namespace {
template <class T>
void bindArray(py::module& m, const std::string& typeName) {
    using Src = py::array_t<T, py::array::c_style>;
    using Dest = poplar_kge::ArrayView<T>;
    std::ostringstream name;
    name << "ArrayView[" << typeName << "]";
    py::class_<Dest>(m, name.str().c_str()).def(py::init([](Src& data) {
        return Dest({data.shape(), data.shape() + data.ndim()}, data.mutable_data());
    }));
    py::implicitly_convertible<Src, Dest>();
}
void bindInt16ToFloat16Array(py::module& m) {
    // Hack - use the C++ type int16_t as a dummy placeholder for float16, then convert
    // to poplar_kge::float16
    using Src = py::array_t<int16_t, py::array::c_style>;
    using Dest = poplar_kge::ArrayView<poplar_kge::float16>;
    py::class_<Dest>(m, "ArrayView[float16]").def(py::init([](Src& data) {
        return Dest({data.shape(), data.shape() + data.ndim()},
                    reinterpret_cast<poplar_kge::float16*>(data.mutable_data()));
    }));
    py::implicitly_convertible<Src, Dest>();
}
}  // namespace

PYBIND11_MODULE(libpoplar_kge, m) {
    m.doc() =
        "Implements the core model of Poplar-KGE, a very low-level interface (please use Python "
        "wrapper)";
    bindArray<float>(m, "float32");
    bindArray<uint32_t>(m, "uint32");
    bindInt16ToFloat16Array(m);

    py::class_<poplar_kge::Engine>(m, "Engine",
                                   "Build graph, compile and load the executable onto device")
        .def(py::init<const poplar_kge::Batch&, const std::string&>(), py::arg("settings"),
             py::arg("gp_folder"))
        .def("run", &poplar_kge::Engine::run, "Execute a command (see Python for usage)",
             py::arg("command"), py::arg("data"));
}
