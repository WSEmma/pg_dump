#include <ProcessGroupDump.hpp>
#include <iostream>

static std::vector<std::string> split(char separator, const std::string& string) {
    std::vector<std::string> pieces;
    std::stringstream ss(string);
    std::string item;
    while (std::getline(ss, item, separator)) {
        pieces.push_back(std::move(item));
    }
    return pieces;
}

static const std::string GLOO_SOCKET_IFNAME_ENV = "GLOO_SOCKET_IFNAME";

namespace c10d {

ProcessGroupDump::ProcessGroupDump(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options
): ProcessGroupGloo(store, rank, size, options) {

}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupDump::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts
) {
    std::cout << "ProcessGroupDump found (" << this->getRank() << "/" << this->getSize() << "): " << tensors.size() << std::endl;
    {
        py::gil_scoped_acquire _gil;
        py::object module = py::module::import("ar_queue");
        py::object _ar_queue_put = module.attr("_ar_queue").attr("put");
        for (size_t i = 0; i < tensors.size(); i++) {
            _ar_queue_put(tensors[i].clone());
        }
    }
    
    return this->ProcessGroupGloo::allreduce(tensors, opts);
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupDump::createProcessGroupDump(
        const c10::intrusive_ptr<Store>& store,
        int rank,
        int size,
        const std::chrono::milliseconds& timeout
    ) {
        // 初始化gloo
        // 参考代码：https://github.com/pytorch/pytorch/blob/63e16216d8830b6340816c873b035e1a31ad4636/torch/csrc/distributed/c10d/init.cpp#L1627
        auto options = ::c10d::ProcessGroupGloo::Options::create();

        // Use interfaces listed in "GLOO_SOCKET_IFNAME", if set.
        char* ifnameEnv = getenv(GLOO_SOCKET_IFNAME_ENV.c_str());
        if (ifnameEnv && strlen(ifnameEnv) > 1) {
            for (const auto& iface : split(',', ifnameEnv)) {
            options->devices.push_back(
                ::c10d::ProcessGroupGloo::createDeviceForInterface(iface));
            }
        } else {
            // If no hostname is specified, this function looks up
            // the machine's hostname and returns a device instance
            // associated with the address that the hostname resolves to.
            options->devices.push_back(
                ::c10d::ProcessGroupGloo::createDefaultDevice());
        }

        options->timeout = timeout;
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        options->threads = options->devices.size() * 2;

        return c10::make_intrusive<ProcessGroupDump>(store, rank, size, options);
    }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("createProcessGroupDump", &ProcessGroupDump::createProcessGroupDump);
}

}