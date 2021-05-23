#ifdef _WIN32
#include <windows.h>
#endif
#include <torch/extension.h>

#include <cuda.h>
// 1 测试头文件包含

#include <glad/glad.h>
// 注意 glad 必须在 glfw 之前包含, 因为 glad.h 包含了 openGL 的库如 GL/gl.h, 包含了他就不必再去特意包含OpenGL
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <vector>


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


GLuint pointer_to_buffer = NULL;
GLuint texture_ID = NULL;

void* gimg;
int full = 0;

int W = 0, H = 0, CH = 0, PICH = 0;

int current_width = 0, current_height = 0;

char* winTitle = "Image";

int waitKeyCalled = 0;
int timeDelay = 0;
int pressed_key = -1;

//OpenGL resource API 用来与OpenGL互操作并且进行映射
cudaGraphicsResource_t resource = 0;



unsigned short channel_type = 0;
unsigned short channel_sequence = 0;
unsigned short image_depth = 0;


void CreateTexture(GLuint* textureId, int width, int height, int channels)
{
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, textureId);
	glBindTexture(GL_TEXTURE_2D, *textureId);
	glTexImage2D(GL_TEXTURE_2D, 0, channel_type, width, height, 0, channel_sequence, image_depth, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void CreatePBO(GLuint* pbo)
{
	if (pbo)
	{
		int data_size = W * H * CH * sizeof(GLubyte);
		glGenBuffers(1, pbo);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, data_size, NULL, GL_DYNAMIC_COPY);

		//缓冲区注册 (新API)
		cudaGraphicsGLRegisterBuffer(&resource, *pbo, cudaGraphicsRegisterFlagsNone);
		//cudaGLRegisterBufferObject(*pbo);
	}
}

void RunCUDA()
{
	void* dptr = NULL;
	size_t size = W * H * CH;
	size_t pitch = W * CH;

	size_t resource_size = 0;



	//--------------new--------------
	CreatePBO(&pointer_to_buffer);
	//映射1个资源
	cudaGraphicsMapResources(1, &resource, 0);
	//获得映射的指针
	cudaGraphicsResourceGetMappedPointer(&dptr, &resource_size, resource);
	cudaMemcpy2D(dptr, pitch, gimg, PICH, pitch, H, cudaMemcpyDeviceToDevice);
	//释放资源使得OpenGL得以访问
	cudaGraphicsUnmapResources(1, &resource, 0);
	cudaGraphicsUnregisterResource(resource);
}




void DisplayFunction()
{
	//glutTimerFunc(timeDelay, TimerFunction, timeDelay != 0);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pointer_to_buffer);
	glBindTexture(GL_TEXTURE_2D, texture_ID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, channel_sequence, image_depth, NULL);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f);	glVertex2f(0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f);	glVertex2f(0.0f, 1.0f);
	glTexCoord2f(1.0f, 0.0f);	glVertex2f(1.0f, 1.0f);
	glTexCoord2f(1.0f, 1.0f);	glVertex2f(1.0f, 0.0f);
	glEnd();
}




void InitCUDA()
{
	//cudaGLSetGLDevice(0); 不在使用
	CreateTexture(&texture_ID, W, H, CH);
	RunCUDA();
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);

}


void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}


//data = image data in GPU Global memory
int showRawImage(void* data, int width, int height, size_t pitch, int channels) {
	winTitle = "Image";

	W = width;
	H = height;
	current_width = W;
	current_height = H;
	CH = channels;
	PICH = pitch;
	gimg = data;

	switch (channels)
	{
	case 1:
		channel_type = GL_LUMINANCE;
		channel_sequence = GL_LUMINANCE;
		image_depth = GL_UNSIGNED_BYTE;
        break;
	case 3:
		channel_type = GL_RGB;
		channel_sequence = GL_RGB;
		image_depth = GL_UNSIGNED_BYTE;
		break;

	case 4:
		channel_type = GL_RGBA;
		channel_sequence = GL_RGBA;
		image_depth = GL_UNSIGNED_BYTE;
		break;
	default:
		return -1;
	}


	glfwInit();

	//使用OpenGL 版本 3.3
	//glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	//glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

	// 告诉 GLFW 我们要使用core-profile.
	// 这意味着我们想要使用OpenGL的特性，并且不想要向后兼容 (glBegin不能在CORE_PROFILE 模式下运行)
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);


	//创建窗口，可以忽略后面两个参数
	GLFWwindow* window = glfwCreateWindow(width, height, "GPU image view", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);



	// 利用glad 获得指定该操作系统的 函数指针，这里是 (glfwGetProcAddress)
	// 初始化 glad
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	//告诉OpenGL 在哪个范围内绘制图像
	// 0,0 代表左下角的点是 0,0
	// 800, 600  代表width=800 , height =600
	glViewport(0, 0, width * 10, height * 10);


	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);



	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);


	InitCUDA();

	//glfwWindowShouldclose 会监测是否关闭窗口
	while (!glfwWindowShouldClose(window))
	{
		processInput(window);
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		// rendering commands here
		// setting color state
		// using and rendering color state
		DisplayFunction();
		glBindTexture(GL_TEXTURE_2D, 0);

		//将内存中缓冲区的内容显示到 window中
		glfwSwapBuffers(window);
		//检测鼠标，键盘事件
		glfwPollEvents();
	}

	glfwTerminate();
}





void showshow(){
	//加载 宽 高 通道数
	int width, height, bpp;

	unsigned char* dev_bitmap;

	unsigned char* rgb_image = stbi_load("C:\\Users\\dell\\Desktop\\testPBO\\container.jpg", &width, &height, &bpp, 0);

	if (!rgb_image) {
		std::cout << "Failed to load texture" << std::endl;
		exit(-1);
	}

	printf("%d, %d, %d\n", width, height, bpp);

	cudaMalloc((void**)&dev_bitmap, width * height * bpp);
	cudaMemcpy(dev_bitmap, rgb_image, width * height * bpp, cudaMemcpyHostToDevice);

	showRawImage(dev_bitmap, width, height, width * bpp, bpp);
}

//先假定类型为 float32 类型
//<typename T>
__global__ void innormalrized(unsigned char * img,void * data,int num_pixels){
    //float 32类型
    float * f_data = (float*)data;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < num_pixels) {
        img[tid] = (unsigned char)(255 * f_data[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

//bpp: bytes per pixels
//一般来说tensor shape(channels, width, height)
void gpu_imshow(torch::Tensor tensor){

    void * data_ptr = tensor.data_ptr();
    // or channel
    int bpp = tensor.size(2);

    int width = tensor.size(0);
    int height = tensor.size(1);
    int pitch = width * bpp;


    if(tensor.device().is_cuda()){
        printf("tensor is cuda tensor!!, device index:%d", tensor.device().index());
        printf("data_ptr: %p", data_ptr);
    }else{
        printf("tensor is not cuda tensor!!, device index:%d", tensor.device().index());
        printf("data_ptr: %p", data_ptr);
    }

    unsigned char * host_mem = new unsigned char[width*height*bpp];

    unsigned char * img_bytes;
    HANDLE_ERROR( cudaMalloc( (void**)&img_bytes, width * height * bpp * sizeof(char) ) );
    innormalrized<<<128, 128>>>(img_bytes, data_ptr, width * height * bpp );

    HANDLE_ERROR( cudaMemcpy(host_mem,img_bytes , width * height * bpp * sizeof(char), cudaMemcpyDeviceToHost));
    for (int i =0; i < bpp; i++){
        for (int j =0 ;j < height; j++){
            for (int k =0 ; k < width; k++){
                    printf("%d ", host_mem[ k + height*j ]);
            }
            printf("\n");
        }
    }


    printf("\n");
    printf("width:%d, height:%d, bpp:%d", width, height, bpp);
    showRawImage(img_bytes, width , height , pitch, bpp);

}











template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmax(0.0, double(z)) + fmin(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}





template <typename scalar_t>
__global__ void lltm_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < gates.size(2)){
    input_gate[n][c] = sigmoid(gates[n][0][c]);
    output_gate[n][c] = sigmoid(gates[n][1][c]);
    candidate_cell[n][c] = elu(gates[n][2][c]);
    new_cell[n][c] =
        old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
    new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
  }
}



std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);
  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

  auto gates = gate_weights.reshape({batch_size, 3, state_size});
  auto new_h = torch::zeros_like(old_cell);
  auto new_cell = torch::zeros_like(old_cell);
  auto input_gate = torch::zeros_like(old_cell);
  auto output_gate = torch::zeros_like(old_cell);
  auto candidate_cell = torch::zeros_like(old_cell);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
    lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}

template <typename scalar_t>
__global__ void lltm_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_old_cell,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> d_gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_h,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_cell,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gate_weights) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < d_gates.size(2)){
    const auto d_output_gate = tanh(new_cell[n][c]) * grad_h[n][c];
    const auto d_tanh_new_cell = output_gate[n][c] * grad_h[n][c];
    const auto d_new_cell =
        d_tanh(new_cell[n][c]) * d_tanh_new_cell + grad_cell[n][c];


    d_old_cell[n][c] = d_new_cell;
    const auto d_candidate_cell = input_gate[n][c] * d_new_cell;
    const auto d_input_gate = candidate_cell[n][c] * d_new_cell;

    d_gates[n][0][c] =
        d_input_gate * d_sigmoid(gate_weights[n][0][c]);
    d_gates[n][1][c] =
        d_output_gate * d_sigmoid(gate_weights[n][1][c]);
    d_gates[n][2][c] =
        d_candidate_cell * d_elu(gate_weights[n][2][c]);
  }
}

std::vector<torch::Tensor> lltm_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gates,
    torch::Tensor weights) {
  auto d_old_cell = torch::zeros_like(new_cell);
  auto d_gates = torch::zeros_like(gates);

  const auto batch_size = new_cell.size(0);
  const auto state_size = new_cell.size(1);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_backward_cuda", ([&] {
    lltm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        d_gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        grad_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));

  auto d_gate_weights = d_gates.reshape({batch_size, 3*state_size});
  auto d_weights = d_gate_weights.t().mm(X);
  auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gate_weights.mm(weights);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}