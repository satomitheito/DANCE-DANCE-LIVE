import torch
import platform

def validate_pytorch_gpu():
    """Validate that PyTorch is properly configured to use GPU."""
    print("PyTorch version:", torch.__version__)
    print("Python version:", platform.python_version())
    print("Platform:", platform.platform())
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Check MPS (Apple Silicon) availability
    mps_available = torch.backends.mps.is_available()
    print(f"MPS (Apple Silicon GPU) available: {mps_available}")
    
    if mps_available:
        print("✅ Apple Silicon GPU support is available!")
    
    # Test GPU computation
    print("\nTesting GPU computation...")
    
    if cuda_available:
        try:
            device = torch.device('cuda:0')
            print(f"Using CUDA device: {device}")
            
            # Test basic operations
            a = torch.tensor([1.0, 2.0, 3.0], device=device)
            b = torch.tensor([4.0, 5.0, 6.0], device=device)
            c = a + b
            
            print(f"CUDA computation result: {c.cpu().numpy()}")
            print("✅ CUDA GPU computation successful!")
            
        except Exception as e:
            print(f"❌ CUDA GPU test failed: {e}")
    
    elif mps_available:
        try:
            device = torch.device('mps')
            print(f"Using MPS device: {device}")
            
            # Test basic operations
            a = torch.tensor([1.0, 2.0, 3.0], device=device)
            b = torch.tensor([4.0, 5.0, 6.0], device=device)
            c = a + b
            
            print(f"MPS computation result: {c.cpu().numpy()}")
            print("✅ Apple Silicon GPU computation successful!")
            
        except Exception as e:
            print(f"❌ MPS GPU test failed: {e}")
    
    else:
        print("Using CPU for computation...")
        device = torch.device('cpu')
        
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        c = a + b
        
        print(f"CPU computation result: {c.numpy()}")
        print("ℹ️  Running on CPU (no GPU acceleration available)")
    
    # Test model creation and inference
    print("\nTesting model creation...")
    try:
        # Create a simple model
        model = torch.nn.Linear(3, 1).to(device)
        test_input = torch.tensor([[1.0, 2.0, 3.0]], device=device)
        output = model(test_input)
        
        print(f"Model output: {output.item():.4f}")
        print("✅ Model creation and inference successful!")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
    
    # Summary
    print("\n" + "="*50)
    if cuda_available:
        print("🎉 PyTorch is configured with CUDA GPU support!")
    elif mps_available:
        print("🎉 PyTorch is configured with Apple Silicon GPU support!")
    else:
        print("ℹ️  PyTorch is running on CPU")
        print("   For GPU acceleration on Mac, make sure you have Apple Silicon")
        print("   For NVIDIA GPUs, install CUDA toolkit")

if __name__ == "__main__":
    validate_pytorch_gpu()
