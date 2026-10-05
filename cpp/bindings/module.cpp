// Python bindings for the C++ core (module `qpricing._qpcore`).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "qp/black76.hpp"
#include "qp/pricer.hpp"
#include "qp/schrodinger.hpp"
#include "qp/spectral.hpp"

namespace py = pybind11;

namespace {

std::vector<qp::OptionType> to_types(const std::vector<bool>& is_call) {
    std::vector<qp::OptionType> out(is_call.size());
    for (std::size_t i = 0; i < is_call.size(); ++i) {
        out[i] = is_call[i] ? qp::OptionType::Call : qp::OptionType::Put;
    }
    return out;
}

py::array_t<double> as_array(const std::vector<double>& v) {
    return py::array_t<double>(static_cast<py::ssize_t>(v.size()), v.data());
}

}  // namespace

PYBIND11_MODULE(_qpcore, m) {
    m.doc() = "Imaginary-time Schrödinger option pricer (C++ core)";

    py::class_<qp::EffectiveVolParams>(m, "EffectiveVolParams")
        .def(py::init([](double a, double b, double rho, double mm, double s) {
                 return qp::EffectiveVolParams{a, b, rho, mm, s};
             }),
             py::arg("a"), py::arg("b"), py::arg("rho"), py::arg("m"), py::arg("s"))
        .def_static("constant", &qp::EffectiveVolParams::constant, py::arg("sigma"))
        .def_readwrite("a", &qp::EffectiveVolParams::a)
        .def_readwrite("b", &qp::EffectiveVolParams::b)
        .def_readwrite("rho", &qp::EffectiveVolParams::rho)
        .def_readwrite("m", &qp::EffectiveVolParams::m)
        .def_readwrite("s", &qp::EffectiveVolParams::s)
        .def("min_variance", &qp::EffectiveVolParams::min_variance)
        .def("__repr__", [](const qp::EffectiveVolParams& p) {
            return "EffectiveVolParams(a=" + std::to_string(p.a) + ", b=" + std::to_string(p.b) +
                   ", rho=" + std::to_string(p.rho) + ", m=" + std::to_string(p.m) +
                   ", s=" + std::to_string(p.s) + ")";
        });

    py::class_<qp::GridSpec>(m, "GridSpec")
        .def(py::init([](int n_space, int n_time, double width_sd, int rannacher_steps) {
                 return qp::GridSpec{n_space, n_time, width_sd, rannacher_steps};
             }),
             py::arg("n_space") = 1200, py::arg("n_time") = 200, py::arg("width_sd") = 8.0,
             py::arg("rannacher_steps") = 2)
        .def_readwrite("n_space", &qp::GridSpec::n_space)
        .def_readwrite("n_time", &qp::GridSpec::n_time)
        .def_readwrite("width_sd", &qp::GridSpec::width_sd)
        .def_readwrite("rannacher_steps", &qp::GridSpec::rannacher_steps);

    m.def(
        "black76_price",
        [](bool is_call, double F, double K, double T, double sigma, double df) {
            return qp::black76_price(is_call ? qp::OptionType::Call : qp::OptionType::Put, F, K, T,
                                     sigma, df);
        },
        py::arg("is_call"), py::arg("F"), py::arg("K"), py::arg("T"), py::arg("sigma"), py::arg("df"));

    m.def(
        "black76_implied_vol",
        [](bool is_call, double F, double K, double T, double price, double df) {
            return qp::black76_implied_vol(is_call ? qp::OptionType::Call : qp::OptionType::Put, F,
                                           K, T, price, df);
        },
        py::arg("is_call"), py::arg("F"), py::arg("K"), py::arg("T"), py::arg("price"), py::arg("df"));

    m.def(
        "schrodinger_price",
        [](const qp::EffectiveVolParams& p, double F, double T, double df,
           const std::vector<double>& strikes, const std::vector<bool>& is_call,
           const qp::GridSpec& grid) {
            std::vector<double> out;
            {
                py::gil_scoped_release release;
                out = qp::schrodinger_price(p, F, T, df, strikes, to_types(is_call), grid);
            }
            return as_array(out);
        },
        py::arg("params"), py::arg("F"), py::arg("T"), py::arg("df"), py::arg("strikes"),
        py::arg("is_call"), py::arg("grid") = qp::GridSpec{},
        "Price one expiry slice. Returns an array aligned with `strikes`.");

    m.def(
        "solve",
        [](const qp::EffectiveVolParams& p, double T, const qp::GridSpec& grid, int n_states) {
            const qp::SchrodingerProblem problem(p, T, grid);
            const auto d = qp::solve_density(problem, T);
            py::dict out;
            out["y"] = as_array(problem.y());
            out["x"] = as_array(problem.x());
            out["sigma"] = as_array(problem.sigma());
            out["drift"] = as_array(problem.drift());
            out["potential"] = as_array(problem.potential());
            out["gauge"] = as_array(problem.gauge());
            out["q"] = as_array(d.q);
            out["p_x"] = as_array(d.density_x(problem.sigma()));
            out["mass"] = d.mass();
            out["forward_ratio"] = d.forward_ratio();
            if (n_states > 0) {
                const auto spec = qp::lowest_states(problem, n_states);
                out["energies"] = as_array(spec.energies);
                py::list states;
                for (const auto& s : spec.states) states.append(as_array(s));
                out["states"] = states;
            }
            return out;
        },
        py::arg("params"), py::arg("T"), py::arg("grid") = qp::GridSpec{}, py::arg("n_states") = 0,
        "Full diagnostic solve: grid, potential well, density and (optionally) bound states.");
}
