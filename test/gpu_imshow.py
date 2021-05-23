import torch
import ctypes
from d2l import torch as d2l
import matplotlib.pyplot as plt
ctypes.CDLL(r'C:\dev\vcpkg\installed\x64-windows\bin\glfw3.dll')
import lltm_cuda

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    dataiter = iter(test_iter)
    images, labels = dataiter.next()

    image = images[0]
    plt.imshow(image.permute(1, 2, 0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    # dataiter[0].to(device)
    image = image * 255
    cuda_tensor: torch.Tensor = image.to(device)
    # cuda_tensor = cuda_tensor * 255;
    # byte_cuda = cuda_tensor.type(dtype=torch.ByteTensor)
    plt.show()
    # cuda_tensor.permute(1,2,0)
    a = cuda_tensor.permute(1, 2, 0)
    lltm_cuda.gpu_imshow(a)
