import cv2
from EdgeDetector import EdgeDetector
from EdgeConnection import EdgeSegDetector
from utils import writeImage
from SuperpixelGenerator import SuperpixelGenerator

def getImgName(imgFile):
    idx = imgFile.find(".")
    return imgFile[:idx]

if __name__ == "__main__":

    ed = EdgeDetector()
    ec = EdgeSegDetector()
    sesg = SuperpixelGenerator(spNum=1000)

    # 处理当前图像
    edgeFile = "10081_edge.png"
    priImgFile = "10081.jpg"

    img = cv2.imread(edgeFile, cv2.IMREAD_GRAYSCALE)
    imgName = getImgName(edgeFile)
    # 原图像
    primaryImg = cv2.imread(priImgFile, cv2.IMREAD_COLOR)

    # 朝向估计细化边缘
    ed.loadImage(img, imgName)

    # 连接边缘
    ec.connectEdge(imgName, ed.priImg, ed.V2ThinnedResponse, ed.V2EstiOrient)

    # 基于网格线获得所有边缘段，这些边缘段一定形成封闭小区域
    sesg.generateSmallEdgeSegment(imgName, img.shape, ec.ES, ec.AO)

    # 生成超像素
    sesg.generateSPs(primaryImg)

    # 可视化超像素
    result, spIdx = sesg.visuSP(primaryImg)
    writeImage(result, imgName + "_sp", False, "result")
