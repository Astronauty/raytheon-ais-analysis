# Test available lazy tensor imports for your GPyTorch version
import gpytorch
import inspect

# Print GPyTorch version
print(f"GPyTorch version: {gpytorch.__version__}")

# Basic import test
print("\n=== Basic import tests ===")
try:
    import gpytorch.lazy
    print("✓ import gpytorch.lazy - SUCCESS")
except ImportError as e:
    print("✗ import gpytorch.lazy - FAILED:", e)

# Check for specific lazy tensor types
lazy_tensor_paths = [
    "gpytorch.lazy.LazyTensor",
    "gpytorch.lazy.lazy_tensor.LazyTensor",
    "gpytorch.lazy.BlockDiagLazyTensor",
    "gpytorch.lazy.block_diag_lazy_tensor.BlockDiagLazyTensor",
    "gpytorch.lazy.KroneckerProductLazyTensor",
    "gpytorch.lazy.DiagLazyTensor",
    "gpytorch.lazy.LazyEvaluatedKernelTensor"
]

print("\n=== Checking specific LazyTensor imports ===")
for path in lazy_tensor_paths:
    try:
        # Try dynamic import using importlib
        components = path.split('.')
        module_path = '.'.join(components[:-1])
        class_name = components[-1]
        
        module = __import__(module_path, fromlist=[class_name])
        lazy_class = getattr(module, class_name)
        print(f"✓ {path} - SUCCESS")
    except (ImportError, AttributeError) as e:
        print(f"✗ {path} - FAILED: {e}")

# List available classes in gpytorch.lazy
print("\n=== Available classes in gpytorch.lazy ===")
for name, obj in inspect.getmembers(gpytorch.lazy):
    if inspect.isclass(obj):
        print(f"- {name}")

# Try to find a working block diagonal method
print("\n=== Testing block diagonal functionality ===")
import torch
matrices = [torch.eye(2) for _ in range(3)]

methods = [
    ("torch.block_diag", lambda: torch.block_diag(*matrices)),
]

# Try different implementations
for name, method in methods:
    try:
        result = method()
        print(f"✓ {name} - SUCCESS: shape {result.shape}")
    except (ImportError, AttributeError, RuntimeError) as e:
        print(f"✗ {name} - FAILED: {e}")

# Custom block diagonal implementation
def custom_block_diag(*matrices):
    """Custom implementation of block_diag"""
    device = matrices[0].device
    rows = sum(m.shape[0] for m in matrices)
    cols = sum(m.shape[1] for m in matrices)
    result = torch.zeros(rows, cols, device=device)
    r, c = 0, 0
    for m in matrices:
        h, w = m.shape
        result[r:r+h, c:c+w] = m
        r += h
        c += w
    return result

try:
    result = custom_block_diag(*matrices)
    print(f"✓ custom_block_diag - SUCCESS: shape {result.shape}")
except Exception as e:
    print(f"✗ custom_block_diag - FAILED: {e}")