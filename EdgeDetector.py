import time
import numpy as np
import cv2

def get_gaussian_kernel_size(sigma):
    # Return an integer that represents the size of Gaussian kernel
    threshold = 8e-2
    radius = np.sqrt(-np.log(np.sqrt(2 * np.pi) * sigma * threshold) * (2 * sigma * sigma))
    radius = int(np.ceil(radius))
    kernel_size = radius * 2 + 1
    return kernel_size
def ellipse_gaussian_kernel(sigma_x, sigma_y, theta):
    max_sigma = max(sigma_x, sigma_y)
    kernel_size = get_gaussian_kernel_size(max_sigma)

    centre_x = kernel_size // 2
    centre_y = kernel_size // 2
    sqr_sigma_x = sigma_x ** 2
    sqr_sigma_y = sigma_y ** 2

    a = np.cos(theta) ** 2 / (2 * sqr_sigma_x) + np.sin(theta) ** 2 / (2 * sqr_sigma_y)
    b = - np.sin(2 * theta) / (4 * sqr_sigma_x) + np.sin(2 * theta) / (4 * sqr_sigma_y)
    c = np.sin(theta) ** 2 / (2 * sqr_sigma_x) + np.cos(theta) ** 2 / (2 * sqr_sigma_y)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - centre_x
            y = j - centre_y
            kernel[i, j] = np.exp(- (a * x ** 2 + 2 * b * x * y + c * y ** 2))

    return kernel / kernel.sum()
def TB_side_kernel(num_orients, idx):
    t_kernel = np.zeros([3, 3], dtype=np.float32)
    b_kernel = np.zeros([3, 3], dtype=np.float32)

    theta = idx / num_orients * np.pi
    if(0 <= idx and idx < 3):
        # [0, 45)
        t_kernel[2, 2] = np.tan(theta)
        b_kernel[0, 0] = np.tan(theta)
        t_kernel[1, 2] = 1 - np.tan(theta)
        b_kernel[1, 0] = 1 - np.tan(theta)
    elif(3 <= idx and idx < 6):
        # [45, 90)
        theta = np.pi / 2 - theta
        t_kernel[2, 2] = np.tan(theta)
        b_kernel[0, 0] = np.tan(theta)
        t_kernel[2, 1] = 1 - np.tan(theta)
        b_kernel[0, 1] = 1 - np.tan(theta)
    elif(6 <= idx and idx < 9):
        # [90, 135)
        theta = theta - np.pi / 2
        t_kernel[2, 0] = np.tan(theta)
        b_kernel[0, 2] = np.tan(theta)
        t_kernel[2, 1] = 1 - np.tan(theta)
        b_kernel[0, 1] = 1- np.tan(theta)
    else:
        #[135, 180)
        theta = np.pi - theta
        t_kernel[2, 0] = np.tan(theta)
        b_kernel[0, 2] = np.tan(theta)
        t_kernel[1, 0] = 1 - np.tan(theta)
        b_kernel[1, 2] = 1 - np.tan(theta)

    return t_kernel, b_kernel


class EdgeDetector():
    def __init__(self):
        # These hyper-parameters are used to construct orientation-sensitive neurons in V2
        self.numOrient_estiOri = 12
        self.orientKernels = None
        self.topSideKernels = None
        self.bottomSideKernels = None

        # These hyper-parameters are for discarding weak response of V2 orientation-sensitive neurons
        self.weakEdgeTh = 0.03

        # 当前处理图像信息
        self.priImg = None
        self.imageName = None
        self.H = 0
        self.W = 0

        # 皮层神经元响应
        self.V2Response = None
        self.V2EstiOrient = None
        self.V2ThinnedResponse = None

        # 初始化各类卷积核
        self.initKernels()

    def initKernels(self):
        # Get orientKernels
        t_kernels = []
        for i in range(self.numOrient_estiOri):
            theta = i / self.numOrient_estiOri * np.pi + np.pi / 2
            k = ellipse_gaussian_kernel(0.3, 1.0, theta)
            t_kernels.append(k)
        self.orientKernels = np.array(t_kernels)

        # Get topSideKernels bottomSideKernels
        t_kernels = []
        b_kernels = []
        for i in range(self.numOrient_estiOri):
            t_kernel, b_kernel = TB_side_kernel(self.numOrient_estiOri, i)
            t_kernels.append(t_kernel)
            b_kernels.append(b_kernel)
        self.topSideKernels = np.array(t_kernels)
        self.bottomSideKernels = np.array(b_kernels)


    def loadImage(self, img, imgName):
        # img: 单通道灰度图像
        # imgName: 图像名称
        self.imageName = imgName
        self.priImg = img.astype(np.float32) / 255.  # uint8 to float32
        self.H = img.shape[0]
        self.W = img.shape[1]

        # Edge thinning and orientation estimation
        self.V2()

    def V2(self):
        # 朝向神经元响应并获得最大响应神经元的朝向
        V2OrientColumn = np.zeros([self.H, self.W, self.numOrient_estiOri], dtype=np.float32)
        for i in range(self.numOrient_estiOri):
            V2OrientColumn[:, :, i] = cv2.filter2D(self.priImg, ddepth=-1, kernel=self.orientKernels[i, :, :], borderType=cv2.BORDER_CONSTANT)
        self.V2Response = np.max(V2OrientColumn, axis=2)
        self.V2EstiOrient = np.argmax(V2OrientColumn, axis=2)

        # Prepare the values of two sides
        topSideEstimation = np.zeros([self.H, self.W, self.numOrient_estiOri], dtype=np.float32)
        bottomSideEstimation = np.zeros([self.H, self.W, self.numOrient_estiOri], dtype=np.float32)
        for i in range(self.numOrient_estiOri):
            topSideEstimation[:, :, i] = cv2.filter2D(self.V2Response, ddepth=-1, kernel=self.topSideKernels[i, :, :], borderType=cv2.BORDER_CONSTANT)
            bottomSideEstimation[:, :, i] = cv2.filter2D(self.V2Response, ddepth=-1, kernel=self.bottomSideKernels[i, :, :], borderType=cv2.BORDER_CONSTANT)

        # 将中心同时大于左侧和右侧的地方置 1，否则置 0
        NMS_mask = V2OrientColumn.copy()
        NMS_mask[np.logical_and(V2OrientColumn >= topSideEstimation, V2OrientColumn >= bottomSideEstimation)] = 1
        NMS_mask[NMS_mask != 1] = 0
        # 限制在预估的最优朝向
        V2MaxVal = (V2OrientColumn == V2OrientColumn.max(axis=2)[:, :, None]).astype(int)  # 与上面两行等价
        # 限制在预估的最优朝向，同时该位置大于左右两侧的值
        mask = np.max(NMS_mask * V2MaxVal, axis=2)
        self.V2ThinnedResponse = mask * self.V2Response
