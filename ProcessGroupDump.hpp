#pragma once

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>

#include <c10d/ProcessGroupGloo.hpp>

// 缺少下面两个头文件，C++函数无法导出给Python
#include <torch/extension.h>
#include <pybind11/chrono.h>

namespace c10d {

class ProcessGroupDump: public ProcessGroupGloo {
public:
    ProcessGroupDump(
        const c10::intrusive_ptr<Store>& store,
        int rank,
        int size,
        c10::intrusive_ptr<Options> options
    );

    c10::intrusive_ptr<ProcessGroup::Work> allreduce(
        std::vector<at::Tensor>& tensors,
        const AllreduceOptions& opts = AllreduceOptions()
    ) override;
    
    static c10::intrusive_ptr<ProcessGroup> createProcessGroupDump(
        const c10::intrusive_ptr<Store>& store,
        int rank,
        int size,
        const std::chrono::milliseconds& timeout
    );

    static void ProcessGroupDumpConstructor() __attribute__((constructor)) {
        py::object module = py::module::import("torch.distributed");
        py::object register_backend = module.attr("Backend").attr("register_backend");
        register_backend("dump", py::cpp_function(createProcessGroupDump));
    }
};

}