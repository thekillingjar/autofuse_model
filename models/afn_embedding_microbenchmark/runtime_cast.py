"""Optional runtime cast override for the isolated inference experiment."""

from contextlib import contextmanager

import torch


def convert_element_type_via_to_copy(tensor, dtype):
    # _to_copy must allocate even when the input already has the requested dtype.
    return torch.ops.aten._to_copy.default(tensor, dtype=dtype)


@contextmanager
def use_to_copy_for_prims(enabled, *, dispatch_key="PrivateUse1"):
    """Override real-device dispatch; leave FakeTensor/meta and lowering alone.

    The override affects all runtime prims casts on this device in the process
    while active. CPU dispatch is accepted for local semantic tests only.
    """
    if not enabled:
        yield
        return
    operator = "prims::convert_element_type"
    if torch._C._dispatch_has_kernel_for_dispatch_key(operator, dispatch_key):
        raise RuntimeError(
            f"{operator} already has a {dispatch_key} kernel; "
            "refusing to replace an existing device implementation"
        )
    library = torch.library.Library("prims", "IMPL", dispatch_key)
    try:
        library.impl("convert_element_type", convert_element_type_via_to_copy)
        yield
    finally:
        # Library has no context-manager protocol in PyTorch 2.8.
        library._destroy()
