try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'🚀 CUDA is available! Version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
        print(f'GPU name: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️ CUDA is NOT available. Check driver or environment.')
except Exception as e:
    print('❌ Torch import failed:', e)