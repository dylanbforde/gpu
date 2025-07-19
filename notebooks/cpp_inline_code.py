import torch
from torch.utils.cpp_extension import load_inline

cpp_source_code = """
std::string hello_world() {
    return "Hello, World!";
}
"""

test_cpp = load_inline(
    name="test_cpp",
    cpp_sources=[cpp_source_code],
    functions=["hello_world"],
    verbose=True,
    build_directory="./tmp",
)

print(test_cpp.hello_world())
