import cv2
from utils import computeIdxDis
from EdgeConnectionUtils import *

class EdgeSegDetector():
    def __init__(self):
        self.savedLengthTh = 7

        self.LL = LocalLine()  # LocalLine

        self.imageName = None
        self.edge = None
        self.thinnedEdge = None
        self.AO = None  # argmax of orientation
        self.ES = []  # 未跨越间隙之前的边缘段
        self.ES_wo_gap = []  # 跨越间隙之后的边缘段

    def connectEdge(self, imageName, edge, thinnedEdge, estOri):
        self.imageName = imageName
        self.edge = edge
        self.thinnedEdge = thinnedEdge
        self.AO = estOri
        self.ES.clear()  # 清空上一幅图获得的边缘段
        self.ES_wo_gap.clear()  # 清空上一幅图获得的边缘段

        # 连接边缘点获得边缘段
        self.V4()

    def V4(self):
        ts = self.LL.length
        paddingThinnedEdge = cv2.copyMakeBorder(self.thinnedEdge, ts, ts, ts, ts, borderType=cv2.BORDER_CONSTANT)
        paddingThinnedEdgeMask = paddingThinnedEdge.copy()
        paddingThinnedEdgeMask[paddingThinnedEdgeMask > 0] = 1

        markMask = np.zeros(self.thinnedEdge.shape, dtype=np.int32)  # If one position is marked, its value is set to 1
        for i in range(self.thinnedEdge.shape[0]):
            for j in range(self.thinnedEdge.shape[1]):
                # 如果不是边缘点，跳过
                if self.thinnedEdge[i, j] == 0:  # If it is not the edge point, continue
                    continue
                # 注意，我们不会得到交叉线段，除了 45 度交叉
                if markMask[i, j] == 1:  # If it has been marked, continue
                    continue

                # 从当前点开始生长，获得连续边缘段
                initX = i
                initY = j
                initArgmaxO = self.AO[i, j]

                edgeSeg = EdgeSeg(isRealEdge=True)  # 现在是空线段
                # Add the current position and mark it
                # 将该点加入线段，同时标记该点
                edgeSeg.extend(Point(initX, initY))
                markMask[i, j] = 1
                # Connect the left side
                # 连接左边
                leftPart = self.connectOneSide(True, paddingThinnedEdge, paddingThinnedEdgeMask, markMask, initX, initY, initArgmaxO)
                # Connect the right side
                # 连接右边
                rightPart = self.connectOneSide(False, paddingThinnedEdge, paddingThinnedEdgeMask, markMask, initX, initY, initArgmaxO)
                # Integrate the left side and right side
                #连接左右两侧
                edgeSeg.extend(leftPart)
                edgeSeg.integrateRightPart(rightPart)
                # Save the continuous edge points
                if edgeSeg.getLength() > self.savedLengthTh:
                    # 获得边缘段的显著度
                    salience = self.computeSalience(edgeSeg)
                    edgeSeg.ID = len(self.ES)
                    edgeSeg.salience = salience
                    self.ES.append(edgeSeg)
        # print("ES number: " % len(ES))

    def connectOneSide(self, isLeftSide, paddingThinnedEdge, paddingThinnedEdgeMask, markMask, initX, initY, initArgmaxO):
        edgeSeg = EdgeSeg()
        curX = initX
        curY = initY
        curArgmaxO = initArgmaxO
        HW = self.LL.length * 2 + 1

        while True:
            if isLeftSide:
                # 搜索的偏移位置
                posD = self.LL.leftLinkMasks[curArgmaxO]
            else:
                # 搜索的偏移位置
                posD = self.LL.rightLinkMasks[curArgmaxO]

            localV = 0
            # 邻居点
            neighboringP = Point()  # Adjacent continuous point
            # 邻居点的偏移
            neighboringPointDiff = PosDiff()  # Adjacent continuous point shift
            # 搜索区域
            localSearchRegion = paddingThinnedEdge[curX:curX+HW, curY:curY+HW].copy() * paddingThinnedEdgeMask[curX:curX+HW, curY:curY+HW].copy()

            for pos in posD:
                tV = localSearchRegion[pos.xDiff, pos.yDiff]  # temp value
                # 如果该偏移位置不是边缘，则跳过
                if tV == 0:  # If there is no edge, continue
                    continue
                # 如果该偏移位置是边缘，则进一步处理
                mX = curX + pos.xDiff - self.LL.length  # coordinate in markMask
                mY = curY + pos.yDiff - self.LL.length  # coordinate in markMask
                # 这句似乎无必要，因为既然都有边缘了，表明应该是没有超出图像范围的
                if mX < 0 or mX >= markMask.shape[0] or mY < 0 or mY >= markMask.shape[1]:  # If exceed the image range, continue
                    continue
                tM = markMask[mX, mY]  # temp mark value
                # 如果该偏移位置已经被标注过了，则跳过
                # 可不可以在这里，不仅是跳过，甚至于是直接打断这个循环呢
                # 实验表明，似乎影响不大，原因在于：细化之后主要的边缘是无影响的，而只是在细化效果不好的地方容易出现问题
                if tM == 1:  # If this position has been marked, continue
                    # continue
                    break
                tO = self.AO[mX, mY]  # temp orientation
                # If the difference of two orientations are big, continue
                if computeIdxDis(tO, curArgmaxO, 12) >= 4:
                    continue
                # If there is one edge point, and it is not been marked,
                # and its orientationo is similar to current orientation,
                # it is the continuous edge point.
                # Note: the search process is interrupted
                localV = localSearchRegion[pos.xDiff, pos.yDiff]
                neighboringPointDiff = pos
                break
            if localV == 0:  # If not find the continuous edge point, break
                break
            # Now find the continuous edge point
            neighboringP.x = curX + neighboringPointDiff.xDiff - self.LL.length
            neighboringP.y = curY + neighboringPointDiff.yDiff - self.LL.length
            # Apply DDA algorithm to connect the current point and the continuous point
            ddaSeg = dda(curX, curY, neighboringP.x, neighboringP.y)
            # 为了避免形成环，这里要进行额外处理
            # 如果形成了环，则需要停止循环
            # 如果没有形成环，则继续循环
            isRing = False
            endIdx = 0
            for pIdx, p in enumerate(ddaSeg.points[1:]):
                if markMask[p.x, p.y] == 1:
                    isRing = True
                    endIdx = pIdx
                    break
                else:
                    markMask[p.x, p.y] = 1
            if isRing:
                # Extend edge segment
                edgeSeg.extend(EdgeSeg(ddaSeg.points[:endIdx]))
                # break the loop process
                break
            else:
                # Extend edge segment
                edgeSeg.extend(ddaSeg)
            # # Extend edge segment
            # edgeSeg.extend(ddaSeg)
            # # Update markMask
            # for p in ddaSeg.points:
            #     markMask[p.x, p.y] = 1
            # Update current position
            curX = neighboringP.x
            curY = neighboringP.y
            # 如果经历了最优朝向的跨越，更新左右侧标志
            if curArgmaxO < self.AO[neighboringP.x, neighboringP.y] and curArgmaxO <= 2 and self.AO[neighboringP.x, neighboringP.y] >= 10:
                isLeftSide = not isLeftSide
            if self.AO[neighboringP.x, neighboringP.y] < curArgmaxO and self.AO[neighboringP.x, neighboringP.y] <= 2 and curArgmaxO >= 10:
                isLeftSide = not isLeftSide
            # Update current orientation
            curArgmaxO = self.AO[neighboringP.x, neighboringP.y]
        return edgeSeg

    def computeSalience(self, es):
        sumV = 0.
        for p in es.points:
            sumV += self.edge[p.x, p.y]
        return sumV / len(es.points)
    def visulizeEdgeSeg(self, src, flag="ES"):
        colors = []
        colors.append(np.array([0., 0., 1.], dtype=np.float32))  # 0
        colors.append(np.array([0., .5, 1.], dtype=np.float32))
        colors.append(np.array([0., 1., 1.], dtype=np.float32))  # 60
        colors.append(np.array([0., 1., .5], dtype=np.float32))
        colors.append(np.array([0., 1., 0.], dtype=np.float32))  # 120
        colors.append(np.array([.5, 1., 0.], dtype=np.float32))
        colors.append(np.array([1., 1., 0.], dtype=np.float32))  # 180
        colors.append(np.array([1., .5, 0.], dtype=np.float32))
        colors.append(np.array([1., 0., 0.], dtype=np.float32))  # 240
        colors.append(np.array([1., 0., .5], dtype=np.float32))
        colors.append(np.array([1., 0., 1.], dtype=np.float32))  # 300
        colors.append(np.array([.5, 0., 1.], dtype=np.float32))

        result = np.zeros([src.shape[0], src.shape[1], 3], dtype=np.float32)
        count = 0
        if flag == "ES":
            visu_ES = self.ES
        elif flag == "ES_wo_gap":
            visu_ES = self.ES_wo_gap
        for i in range(len(visu_ES)):
            if i % 2 == 0:
                colorIdx = i % 12
                # colorIdx = 0
            else:
                colorIdx = (i + 6) % 12
                # colorIdx = 8

            tempVec = visu_ES[i]

            if tempVec.getLength() < 3:  # 如果线段长度太短
                continue
            else:
                val = self.computeSalience(tempVec)
                for p in tempVec.getPoints():
                    # result[p.x, p.y, :] = colors[colorIdx]
                    result[p.x, p.y, :] = colors[colorIdx]
                    # result[p.x, p.y, :] = colors[colorIdx] * val
                count += 1
        # print("satisfied edge segments' number is " % count)

        # 将没有边缘的地方换成指定颜色
        gray = np.array([1., 1., 1.], dtype=np.float32)
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                if result[i, j, 0] == 0 and result[i, j, 1] == 0 and result[i, j, 2] == 0:
                    result[i, j, :] = gray
        # result[8::16, 8::16, :] = np.array([0., 0., 1.])
        return result

