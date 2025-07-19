import torch


def time_pytorch_function(func, *args, **kwargs):
    """
    Time a PyTorch function.

    Args:
        func: The PyTorch function to time.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        The time taken to execute the function in seconds.
    """
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    func(*args, **kwargs)
    end_time.record()

    torch.cuda.synchronize()  # Wait for the events to be recorded
    return start_time.elapsed_time(end_time) / 1000.0  # Convert milliseconds to seconds


tensor = torch.randn(1000, 1000, device="cuda")

b = torch.randn(1000, 1000, device="cuda")


def square_version_1(tensor):
    return tensor * tensor


def square_version_2(tensor):
    return tensor**2


def square_version_3(tensor):
    return torch.pow(tensor, 2)


time_v0 = time_pytorch_function(torch.square, tensor)
time_v1 = time_pytorch_function(square_version_1, tensor)
time_v2 = time_pytorch_function(square_version_2, tensor)
time_v3 = time_pytorch_function(square_version_3, tensor)

print("===== torch.square =====")
with torch.profiler.profile() as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


print("===== tensor * tensor =====")
with torch.profiler.profile() as prof:
    square_version_1(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


print("===== tensor**2 =====")
with torch.profiler.profile() as prof:
    square_version_2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("===== torch.pow(tensor, 2) =====")
with torch.profiler.profile() as prof:
    square_version_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
