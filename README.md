# GPU 图片显示

The project is intended for showing image in GPU memory, you can just invoke the function like `plt` library

```python
gpu_imshow(a)
```





## How ?

Here's some expaination show you what happend under the hood





### 1. OpenGL mode

first , There are two types of OpenGL  mode：

- Immidiate mode 

- Core profile mode

Although Modern OpenGL requires Core Profile Mode, its performance is not well accepted, so Immidiate Mode is used here

### 2. GPU Memory Mapping

the first cool tech used here is GPU Memory Mapping which give the power to OpenGL to access data in GPU Memory directly  

check the two documentations bellow

***\*Graphics InterOperability:\**** 

https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP

 

***\*OpenGL InterOperability:\**** https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL

实现代码如下

### 3. PBO

But even if we do the mapping, we still need to pass the data to the Texture object. If we do it the traditional way, it will still be copied into memory (because it has to be passed to the Texture object, as shown below), so we have to use PBO

![image.png](https://i.loli.net/2021/05/23/DWmXNbKOa9Lorez.png)



PBO allows you to skip the CPU copying to memory and pass the image directly to the TextureObject via DMA

![image.png](https://i.loli.net/2021/05/23/og4OzvCGAxNwJE3.png)