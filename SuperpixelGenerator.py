import time
import numpy as np
from EdgeConnectionUtils import Point, EdgeSeg, dda, divideTowSegment
from sortedcontainers import SortedList
import copy
from skimage.segmentation import mark_boundaries

class SPRegion():
    def __init__(self, ID):
        self.ID = ID             # 整数：标号
        self.isValid = True      # 默认当前的超像素是有效的
        self.points = []         # 列表：包含的像素点
        self.size = 0            # 整数：大小，即包含的像素个数
        self.ESSet = None        # 集合：封闭区域的 边界 集合
        self.NeighSet = None     # 集合：当前区域所有邻居区域
        self.Color = None        # 区域的平均颜色

    def extendPoints(self, points):
        self.points.extend(points)
        self.size += len(points)

    def updateESSet(self, t):
        # t：边缘段标号列表，其中有重复值，且包含 -1 (非边缘处值)
        self.ESSet = set(t)
        self.ESSet.discard(-1)


    def updateNeighSet(self, t):
        # t：临近区域标号列表，其中包含 None，自身编号，临近区域标号
        self.NeighSet = set(t)
        self.NeighSet.discard(None)
        self.NeighSet.discard(self.ID)

    def addPoints(self, points):
        self.points.extend(points)
        self.size += len(points)


    def __lt__(self, other):
        return self.size < other.size  # 从小到大排序

    def __repr__(self):
        return f"SPRegion(ID={self.ID})"

class SPSortedIdx():
    def __init__(self, id, size):
        self.ID = id
        self.size = size
class SPCollection():
    def __init__(self):
        self.SPDict = dict()
        self.SPList = SortedList(key=lambda sp: sp.size)

    def addSP(self, sp):
        self.SPDict[sp.ID] = sp
        self.SPList.add(SPSortedIdx(sp.ID, sp.size))

    def createID(self):
        return len(self.SPList) + 1

    def getSize(self):
        return len(self.SPList)

    def clear(self):
        self.SPDict.clear()
        self.SPList.clear()

class SuperpixelGenerator():
    def __init__(self, spNum=500):
        self.spNum = spNum
        self.SPC = SPCollection()
        self.getM()
    def getM(self):
        self.M = np.zeros([3, 3], np.int32)
        self.M[0, 1] = 1
        self.M[1, 0] = 1
        self.M[1, 2] = 1
        self.M[2, 1] = 1
        self.tM = np.zeros([2, 3], np.int32)
        self.tM[0, 0] = 1
        self.tM[1, 1] = 1
        self.tM[0, 2] = 1
        self.bM = np.zeros([2, 3], np.int32)
        self.bM[0, 1] = 1
        self.bM[1, 0] = 1
        self.bM[1, 2] = 1
        self.lM = np.zeros([3, 2], np.int32)
        self.lM[0, 0] = 1
        self.lM[1, 1] = 1
        self.lM[2, 0] = 1
        self.rM = np.zeros([3, 2], np.int32)
        self.rM[0, 1] = 1
        self.rM[1, 0] = 1
        self.rM[2, 1] = 1
        self.searchPoints = []
        self.searchPoints.append(Point(-1, 0))
        self.searchPoints.append(Point(0, 1))
        self.searchPoints.append(Point(1, 0))
        self.searchPoints.append(Point(0, -1))
        self.searchPoints.append(Point(-1, -1))
        self.searchPoints.append(Point(-1, 1))
        self.searchPoints.append(Point(1, 1))
        self.searchPoints.append(Point(1, -1))
        self.searchPoints.append(Point(1, -2))
        self.searchPoints.append(Point(0, -2))
        self.searchPoints.append(Point(-1, -2))
        self.searchPoints.append(Point(-2, -2))
        self.searchPoints.append(Point(-2, -1))
        self.searchPoints.append(Point(-2, 0))
        self.searchPoints.append(Point(-2, 1))
        self.searchPoints.append(Point(-2, 2))
        self.searchPoints.append(Point(-1, 2))
        self.searchPoints.append(Point(0, 2))
        self.searchPoints.append(Point(1, 2))
        self.searchPoints.append(Point(2, 2))
        self.searchPoints.append(Point(2, 1))
        self.searchPoints.append(Point(2, 0))
        self.searchPoints.append(Point(2, -1))
        self.searchPoints.append(Point(2, -2))


    def extendEndpoint(self, isStartP, ep, es, esIdx, searchDis):
        # 若 端点 是 交点，则不用处理（这种情况的出现是由于，前面迭代中有边缘段扩展造成相交，然后相交的边缘段进行了划分，导致出现了交点）
        # 若 端点 不是 交点，则需沿朝向方向延伸，以与其它边缘段相交
        # 注意，第一个点 端点，一定不在另一条边缘段上，这是边缘连接算法保证的
        # 注意，若延长的线段相交于自己(形成圆环)，函数可以将本线段正确划分为两段
        # 注意，优先考虑真实边缘相交，若无法相交，再来考虑与网格相交
        isIntersection = False
        if ep in self.intersectionMark:
            if esIdx not in self.intersectionMark[ep]:
                print("Note: this is unexpected, theoretically, esIdx must be in intersectionPointMark[ep].")
            return
        else:
            # 获得 端点朝向
            orient = self.AO[ep.x, ep.y]
            # 得到 端点朝向 后，沿朝向方向延伸
            # 注意，边缘段序列中的第一个点就是 端点 (由于 边缘段 的长度一定 >3，故 es.pionts[2] 存在)
            # 注意，这个假设有问题，因为我们需要在 真实边缘段 相交时，划分边缘段，这可能造成 边缘段 长度 <3 出现
            if isStartP:
                if es.getLength() < 3:
                    extPs = getPoints(ep, es.points[1], orient, dis=searchDis)
                else:
                    extPs = getPoints(ep, es.points[2], orient, dis=searchDis)
            else:
                if es.getLength() < 3:
                    extPs = getPoints(ep, es.points[-2], orient, dis=searchDis)
                else:
                    extPs = getPoints(ep, es.points[-3], orient, dis=searchDis)
            # addPoints 用于存放需要扩展的边缘点
            addedPoints = []
            # 遍历除 端点 之外剩下的点，以与网格或者其它边缘段相交
            for p in extPs[1:]:
                # 首先，得限定 p 在图像范围内
                if p.x < 0 or p.x >= self.H or p.y < 0 or p.y >= self.W:
                    break

                # 首先，扩展该点(该点一定在图像范围内，原因在于在超出图像范围之前的点，一定会得到处理从而停止循环)
                addedPoints.append(p)
                # 考虑当前扩展的点，有以下几种情况
                # 1. 当扩展到 其它边缘段 时，扩展 端点，更新信息，划分边缘段，记录当前的交点，并停止扩展
                #    注意，若该点是其它边缘段的端点，无需划分
                #    特别注意，若 该点 在当前边缘段上，此时可能扩张之后形成环，需要处理这种特殊情况
                #    特别注意，先扩展。若先断开的话，扩展不知道在哪一段上进行扩展，可能有错误
                # 2. 当上述情况没有达到时，继续循环
                curIdx = self.edgeMark[p.x, p.y]
                # 情形 1
                if curIdx != -1:
                    # c1. 当前点是真实边缘点，若该点是交点，则表明该点处要么是真实边缘段相交处，要么是端点延伸到网格线的点(此时也为真实边缘段的端点)，则更新交点信息
                    #     若相交的真实边缘为自身，即形成环，则如何处理，只能是 0 似乎不用处理
                    # c2. 当前点是真实边缘点，若该点不是交点，且该点是另一条边缘段的点
                    # c3. 当前点是真实边缘点，若该点不是交点，且该点本身就在当前边缘段上，此时形成 6 或 0 这种形状的封闭区域
                    # c1
                    if p in self.intersectionMark:
                        # step1. 扩展 端点
                        # 起点侧，则将扩展点倒序，然后拼接上当前边缘段
                        # 终点侧，当前边缘段直接拼接扩展点
                        self.ES[esIdx].extendP(addedPoints, isStartP)
                        # step2. 更新 edgeMark isRealEdge
                        for tp in addedPoints:
                            self.edgeMark[tp.x, tp.y] = esIdx
                            self.isRealEdge[tp.x, tp.y] = 1
                        # step3. 更新 intersectionMark
                        self.intersectionMark[p].add(esIdx)
                    else:
                        # c2
                        if curIdx != esIdx:
                            # step1. 扩展 端点
                            # 起点侧，则将扩展点倒序，然后拼接上当前边缘段
                            # 终点侧，当前边缘段直接拼接扩展点
                            self.ES[esIdx].extendP(addedPoints, isStartP)
                            # step2. 更新 edgeMark isRealEdge
                            for tp in addedPoints:
                                self.edgeMark[tp.x, tp.y] = esIdx
                                self.isRealEdge[tp.x, tp.y] = 1
                            # step3. 划分边缘段
                            self.updateES(curIdx, p)
                            # step4. 更新 交点 信息 (由 step3，p 一定是交点)
                            self.intersectionMark[p].add(esIdx)
                        # c3，这个情形是最复杂的，此时 curIdx == esIdx
                        else:
                            # 若形成 0
                            if (isStartP and p == self.ES[esIdx].getBack()) or (not isStartP and p == self.ES[esIdx].getFront()):
                                # step1. 扩展 端点
                                # 起点侧，则将扩展点倒序，然后拼接上当前边缘段
                                # 终点侧，当前边缘段直接拼接扩展点
                                self.ES[esIdx].extendP(addedPoints, isStartP)
                                # step2. 更新 edgeMark isRealEdge
                                for tp in addedPoints:
                                    self.edgeMark[tp.x, tp.y] = esIdx
                                    self.isRealEdge[tp.x, tp.y] = 1
                                # step4. 更新 交点 信息 (由 c1，p 一定非交点)
                                self.intersectionMark[p] = set([esIdx])
                            # 若形成 6
                            # 先划分边缘段，后面再去形成 0，进行处理
                            else:
                                # step3. 划分边缘段
                                self.updateES(esIdx, p)
                    isIntersection = True
                    break
                # 情形 2
                else:
                    continue

        # 若上述过程未能扩展到真实边缘点，则现在考虑扩展到网格点
        if isIntersection is False:
            # 若当前点即为网格点，则返回
            if ep.x % self.gridSize == 0 or ep.x == self.H - 1 or ep.y % self.gridSize == 0 or ep.y == self.W - 1:
                self.intersectionMark[ep] = set([esIdx])
                return
            # 2. 当扩展到 网格线 时，扩展 端点，更新信息，记录当前的交点，并停止扩展
            else:
                # 获得 端点朝向
                orient = self.AO[ep.x, ep.y]
                # 得到 端点朝向 后，沿朝向方向延伸
                # 注意，边缘段序列中的第一个点就是 端点 (由于 边缘段 的长度一定 >3，故 es.pionts[2] 存在)
                # 注意，这个假设有问题，因为我们需要在 真实边缘段 相交时，划分边缘段，这可能造成 边缘段 长度 <3 出现
                if isStartP:
                    if es.getLength() < 3:
                        extPs = getPoints(ep, es.points[1], orient, dis=searchDis)
                    else:
                        extPs = getPoints(ep, es.points[2], orient, dis=searchDis)
                else:
                    if es.getLength() < 3:
                        extPs = getPoints(ep, es.points[-2], orient, dis=searchDis)
                    else:
                        extPs = getPoints(ep, es.points[-3], orient, dis=searchDis)
                # addPoints 用于存放需要扩展的边缘点
                addedPoints = []
                # 遍历除 端点 之外剩下的点，以与网格或者其它边缘段相交
                for p in extPs[1:]:
                    # 首先，扩展该点(该点一定在图像范围内，原因在于在超出图像范围之前的点，一定会得到处理从而停止循环)
                    addedPoints.append(p)

                    # 如果到达网格点之后，则扩展，并终止，否则继续
                    if p.x % self.gridSize == 0 or p.x == self.H - 1 or p.y % self.gridSize == 0 or p.y == self.W - 1:
                        # step1. 扩展 端点
                        # 起点侧，则将扩展点倒序，然后拼接上当前边缘段
                        # 终点侧，当前边缘段直接拼接扩展点
                        self.ES[esIdx].extendP(addedPoints, isStartP)
                        # step2. 更新 edgeMark isRealEdge
                        for tp in addedPoints:
                            self.edgeMark[tp.x, tp.y] = esIdx
                            self.isRealEdge[tp.x, tp.y] = 1
                        # step4. 更新 交点 信息
                        if p not in self.intersectionMark:
                            self.intersectionMark[p] = set([esIdx])
                        else:
                            self.intersectionMark[p].add(esIdx)
                        # step5. 停止扩展
                        break
                    else:
                        continue
    def extendTwoEndpoints(self):
        searchDis = int(self.gridSize * np.sqrt(2) * 2 + 1)
        # 使用 s 和 e 来控制循环，而非直接迭代 ES，原因在于，每次操作有可能改变 ES，从而影响迭代
        s = 0
        e = self.ES_num
        while s < e:
            if not self.ES[s].isValid:
                s += 1
                continue
            # 边缘段
            es = self.ES[s]
            # 1. 获得起点，并扩展
            # 注意：若是扩展与自身相交，则边缘段标号有变化，后续的终点处边缘扩展会出现问题
            startP = es.getFront()
            self.extendEndpoint(True, startP, es, s, searchDis)
            # 2. 获得终点，并扩展
            # 注意，这里需要重新判断当前边缘段是否继续有效，与起点处扩展与自身相交相对应
            if self.ES[s].isValid:
                endP = es.getBack()
                self.extendEndpoint(False, endP, es, s, searchDis)
            # 控制循环
            s += 1
            e = len(self.ES)
    def updateES(self, esIdx, p1):
        # 1. p1 为端点
        #    更新交点信息即可
        if p1 == self.ES[esIdx].points[0] or p1 == self.ES[esIdx].points[-1]:
            if p1 not in self.intersectionMark:
                self.intersectionMark[p1] = set([esIdx])
            else:
                self.intersectionMark[p1].add(esIdx)
            return
        # 2. p1 为 中断点，分为两段，对应信息更新
        es1, es2 = divideTowSegment(self.ES[esIdx], p1)
        # 更新 ES ES_num
        self.ES[esIdx].isValid = False  # 废弃这条边缘段
        self.ES.append(es1)  # 添加边缘段
        self.ES.append(es2)  # 添加边缘段
        self.ES_num += 2
        self.ES[-2].ID = self.ES_num - 2
        self.ES[-1].ID = self.ES_num - 1

        # 更新 edgeMark，注意，在 edgeMark 中 交点 的标号是最后一个边缘段的标号
        for p in es1.points:
            self.edgeMark[p.x, p.y] = self.ES_num - 2
        for p in es2.points:
            self.edgeMark[p.x, p.y] = self.ES_num - 1
        # 更新 交点 对应的边缘段值
        # 特别注意的是，若 原边缘段端点处 已经为交点了，则该交点处对应的边缘段标号需要更新
        if p1 not in self.intersectionMark:
            self.intersectionMark[p1] = set([self.ES_num - 2, self.ES_num - 1])
        else:
            self.intersectionMark[p1].add(self.ES_num - 2)
            self.intersectionMark[p1].add(self.ES_num - 1)
        sP = self.ES[esIdx].getFront()
        eP = self.ES[esIdx].getBack()
        if sP in es1.points:
            # 此时 sP 属于 es1，eP 属于 es2
            # 若 sP 是交点，则更新标号，否则不处理，eP 同理 (原因在于 若 sP 是交点的话，该交点一定有对应的 esIdx)
            if sP in self.intersectionMark:
                self.intersectionMark[sP].remove(esIdx)  # 移除原有标号
                self.intersectionMark[sP].add(self.ES_num - 2)  # 更新新标号
            if eP in self.intersectionMark:
                # 特别注意，sP 与 eP 可能相同，这主要是端点扩展造成的，因此，下面使用 discard 而非 remove
                self.intersectionMark[eP].discard(esIdx)  # 移除原有标号
                self.intersectionMark[eP].add(self.ES_num - 1)  # 更新新标号
        else:
            # 此时 sP 属于 es2，eP 属于 es1
            # 若 sP 是交点，则更新标号，否则不处理，eP 同理 (原因在于 若 sP 是交点的话，该交点一定有对应的 esIdx)
            if sP in self.intersectionMark:
                self.intersectionMark[sP].remove(esIdx)  # 移除原有标号
                self.intersectionMark[sP].add(self.ES_num - 1)  # 更新新标号
            if eP in self.intersectionMark:
                # 特别注意，sP 与 eP 可能相同，这主要是端点扩展造成的，因此，下面使用 discard 而非 remove
                self.intersectionMark[eP].discard(esIdx)  # 移除原有标号
                self.intersectionMark[eP].add(self.ES_num - 2)  # 更新新标号
    def updateES2(self, esIdx, p1, p2):
        # 1. p1 或 p2 是 端点
        # 注意，无需更新 交点 信息，在该函数之前执行的代码保证 端点 一定是 交点
        if p1 == self.ES[esIdx].points[0] or p1 == self.ES[esIdx].points[-1]:
            self.updateES(esIdx, p2)
            return
        if p2 == self.ES[esIdx].points[0] or p2 == self.ES[esIdx].points[-1]:
            self.updateES(esIdx, p1)
            return
        # 2. p1 p2 都是 中断点，分为三段，对应信息更新
        # 首先利用 p1 将 边缘段 分为两段，这两段边缘段的标号分别为 ES_num-1 和 ES_num-2
        self.updateES(esIdx, p1)
        # 此时，判断 p2 在哪个 边缘段中，再对该边缘段分段
        if p2 in self.ES[-1].points:
            self.updateES(self.ES_num - 1, p2)
        else:
            self.updateES(self.ES_num - 2, p2)
    def addES(self, points):
        # 添加新的边缘段
        # 注意，新的边缘段 都是网格线上的边缘段，而非真实边缘得到的边缘段
        t_es = EdgeSeg(points[:], salience=0.01, isRealEdge=False)  # 特别注意，必须要用 points[:]，这样才是对 points 的拷贝，否则 points 清除之后，这里也清除
        self.ES.append(t_es)
        self.ES_num += 1
        self.ES[-1].ID = self.ES_num - 1
        # 更新 edgeMark，注意，断点 两个边缘段都拥有，但其标号在 edgeMark 中的标号则是最后一个边缘段的标号
        for p in points:
            self.edgeMark[p.x, p.y] = self.ES_num - 1
        # 更新 交点 对应的边缘段值
        if points[0] not in self.intersectionMark:
            self.intersectionMark[points[0]] = set([self.ES_num - 1])
        else:
            self.intersectionMark[points[0]].add(self.ES_num - 1)
        if points[-1] not in self.intersectionMark:
            self.intersectionMark[points[-1]] = set([self.ES_num - 1])
        else:
            self.intersectionMark[points[-1]].add(self.ES_num - 1)
    def addGridH(self):
        tempEM = []  # tempEM 用于第一遍扫描，记录与 网格线 重叠的 真实边缘段
        tempPoints = []  # tempPoints 用于第二遍扫描，记录 网格线 生成的 新的边缘段
        # 考虑横线，注意 +1
        grid_num_H = 1 + int((self.H + self.gridSize - 1) / self.gridSize)
        for igrid in range(grid_num_H):
            # 横坐标，对最后一个网格，注意处理
            i = igrid * self.gridSize
            if i >= self.H:
                i = self.H - 1

            # 现在先扫描一遍纵坐标，看看是否有 真实边缘段 跟网格线的重叠超过 1 的
            #    若重叠为 1，则表明为交点，以该点为中断点，将 真实边缘段分为 两半 即可
            #    若重叠 >1，则需要将 真实边缘段 分为 三段，中间重叠的部分是单独一段
            for j in range(self.W):
                cur_isRealEdge = self.isRealEdge[i, j]
                cur_em = self.edgeMark[i, j]
                # 1. 当 当前点 不是真实边缘点时，同时 tempEM 中没有点，直接去判断下一个点 (是否为 终点 不影响)
                if not cur_isRealEdge and len(tempEM) == 0:
                    continue
                # 2. 当 当前点 不是真实边缘点时，但 tempEM 中有点，则表明重叠的边缘段到尽头了 (是否为 终点 没有影响)
                if not cur_isRealEdge and len(tempEM) > 0:
                    # 此时，考虑 tempEM 的长度情况，即考虑重叠是单个点还是一段
                    #    若重叠为 1，则 当前点 为交点，以该点为中断点，将 真实边缘段 分为 两半
                    #    若重叠 >1，则将 真实边缘段 分为 三段
                    prev_em = self.edgeMark[tempEM[-1].x, tempEM[-1].y]
                    if len(tempEM) == 1:
                        # 将 真实边缘段 分为两段
                        self.updateES(prev_em, tempEM[-1])
                    else:
                        # 将 真实边缘段 分为三段
                        self.updateES2(prev_em, tempEM[0], tempEM[-1])
                    # tempEM 清空，然后去判断下一个点
                    tempEM.clear()
                    continue
                # 3. 当 当前点 是真实边缘，但 tempEM 中没有点，则表明 真实边缘段 与 网格线 的重叠第一次出现 (是否为 终点 有影响)
                if cur_isRealEdge and len(tempEM) == 0:
                    # 若不是 终点，则添加该点到 tempEM， 然后去判断下一个点
                    if j != self.W - 1:
                        tempEM.append(Point(i, j))
                        continue
                    # 若是 终点，则表明为 交点 且重叠为 1，应以该点为中断点，将 真实边缘段分为 两半 即可， 然后去判断下一个点
                    else:
                        # 将 真实边缘段 分为两段
                        self.updateES(cur_em, Point(i, j))
                        continue
                # 4. 当 当前点 是真实边缘，且 tempEM 中有点，则有两种情况：
                #    若 当前点 与 tempEM 中的点为 同一边缘段 上的点，表明 真实边缘段 与 网格线 的重叠是一段 (是否为 终点 有影响)
                #    若 当前点 与 tempEM 中的点为 不同边缘段 上的点，此时需要对 tempEM 进行判断 并进行处理 (是否为 终点 有影响)
                if cur_isRealEdge and len(tempEM) > 0:
                    # c1. 若 len(tempEM) == 1，则考虑当前点的不同情形
                    # c2. 若 len(tempEM) > 1，则考虑当前点的不同情形
                    if len(tempEM) == 1:
                        # c1 cc1. 前一个点 和 当前点 都是交点，则更新 tempEM
                        if tempEM[-1] in self.intersectionMark and Point(i, j) in self.intersectionMark:
                            # tempEM.clear()
                            tempEM[-1] = Point(i, j)
                            continue
                        # c1 cc2. 若前一个点是交点，当前点不是交点，则判断是否同边缘段
                        #    若是相同边缘段则 tempEM 加入当前点，继续循环 (是否为终点有影响)
                        #    若不同，则 更新 tempEM，继续循环 (是否为终点有影响)
                        if tempEM[-1] in self.intersectionMark and Point(i, j) not in self.intersectionMark:
                            if cur_em in self.intersectionMark[tempEM[-1]]:
                                tempEM.append(Point(i, j))
                                if j != self.W - 1:
                                    continue
                                else:
                                    self.updateES2(cur_em, tempEM[-1], Point(i, j))  # 注意得使用当前点指向的边缘段
                                    tempEM.clear()
                                    continue
                            else:
                                tempEM.clear()
                                if j != self.W - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    self.updateES(cur_em, Point(i, j))
                                    continue
                        # c1 cc3. 若前一个点不是交点，当前点是交点，则判断是否同边缘段
                        #    若同，则加入当前点分段，更新 tempEM，继续循环 (是否为终点有影响)
                        #    若不同，则分段，更新 tempEM，继续循环 (是否为终点有影响)
                        if tempEM[-1] not in self.intersectionMark and Point(i, j) in self.intersectionMark:
                            prev_em = self.edgeMark[tempEM[-1].x, tempEM[-1].y]
                            if prev_em in self.intersectionMark[Point(i, j)]:
                                tempEM.append(Point(i, j))
                                self.updateES2(prev_em, tempEM[0], tempEM[-1])
                                tempEM.clear()
                                if j != self.W - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    continue
                            else:
                                self.updateES(prev_em, tempEM[-1])
                                tempEM.clear()
                                if j != self.W - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    continue
                        # c1 cc4. 若前一个点和当前点都不是交点，则判断是否同边缘
                        #    若同则加入继续 (是否为终点有影响)
                        #    若不同，则分段，更新 tempEM，继续 (是否为终点有影响)
                        if tempEM[-1] not in self.intersectionMark and Point(i, j) not in self.intersectionMark:
                            prev_em = self.edgeMark[tempEM[-1].x, tempEM[-1].y]
                            if cur_em == prev_em:
                                tempEM.append(Point(i, j))
                                if j != self.W - 1:
                                    continue
                                else:
                                    self.updateES2(cur_em, tempEM[0], tempEM[-1])
                                    tempEM.clear()
                                    continue
                            else:
                                self.updateES(prev_em, tempEM[-1])
                                tempEM.clear()
                                if j != self.W - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    self.updateES(cur_em, Point(i, j))
                                    continue
                    # c2 此时 len(tempEM) > 1
                    #  特别注意，此时不用考虑 tempEM 中最后一个点是否为交点，其一定不为交点，因为若其为交点，则等不到现在处理
                    else:
                        # c2 cc1. 若当前点是交点，则判断是否同边缘段
                        #    若同，则加入当前点，分段，更新 tempEM，继续循环 (是否为终点有影响)
                        #    若不同，则分段 (是否为终点有影响)
                        if Point(i, j) in self.intersectionMark:
                            prev_em = self.edgeMark[tempEM[-1].x, tempEM[-1].y]
                            if prev_em in self.intersectionMark[Point(i, j)]:
                                tempEM.append(Point(i, j))
                                self.updateES2(prev_em, tempEM[0], tempEM[-1])
                                tempEM.clear()
                                if j != self.W - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    continue
                            else:
                                self.updateES2(prev_em, tempEM[0], tempEM[-1])
                                tempEM.clear()
                                if j != self.W - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    self.updateES(cur_em, Point(i, j))
                                    continue
                        # c2 cc2. 若当前点不是交点，则判断是否同边缘段
                        #    若同，则加入当前点，继续循环 (是否为终点有影响)
                        #    若不同，则分段，更新 tempEM，继续循环 (是否为终点有影响)
                        else:
                            prev_em = self.edgeMark[tempEM[-1].x, tempEM[-1].y]
                            if prev_em == cur_em:
                                tempEM.append(Point(i, j))
                                if j != self.W - 1:
                                    continue
                                else:
                                    self.updateES2(cur_em, tempEM[0], tempEM[-1])
                                    tempEM.clear()
                                    continue
                            else:
                                self.updateES2(prev_em, tempEM[0], tempEM[-1])
                                tempEM.clear()
                                if j != self.W - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    self.updateES(cur_em, Point(i, j))
                                    continue

            # 再次扫描纵坐标，处理每一段边缘段
            # 实际上，真实边缘点 不用考虑，主要考虑 非真实边缘点，然后要考虑 交点
            for j in range(self.W):
                cur_isRealEdge = self.isRealEdge[i, j]
                # 1. 若 当前点 不是真实边缘点，且其不为 网格交点(不可能是 起点 终点，因为每行 起点 终点 都是网格交点)
                #    则将 当前点 加入 tempPoints，然后去判断下一个点
                if not cur_isRealEdge and j % self.gridSize != 0 and j != self.W - 1:
                    tempPoints.append(Point(i, j))
                    continue
                # 2. 若 当前点 不是真实边缘点，且其为 网格交点(每行 起点 终点 都是网格交点)
                #    则将 当前点 加入 tempPoints，同时，新的边缘段生成了，进行一系列的操作
                # 注意，下面这个判断包含了每行的 起点 终点
                if not cur_isRealEdge and (j % self.gridSize == 0 or j == self.W - 1):
                    tempPoints.append(Point(i, j))
                    # 若 当前点 是 起点，则直接判断下一个点
                    if j == 0:
                        continue
                    # 下面这些情况都是 非起点 的 网格交点，此时对 新获得的边缘段 进行一系列的操作
                    # 添加 新的边缘段
                    self.addES(tempPoints)
                    # 若 当前点 不是 终点 网格交点，则 更新 tempPoints，然后去判断下一个点
                    # 若 当前点 是 终点 网格交点，则 清空 tempPoints，然后去判断下一个点
                    if j != self.W - 1:
                        tempPoints.clear()
                        tempPoints.append(Point(i, j))
                        continue
                    else:
                        tempPoints.clear()
                        continue
                # 3. 若 当前点 是真实边缘点，且其不是 交点 (起点 终点 一定是交点，故不考虑：无论 起点 终点 是否为真实边缘点，其一定是交点)
                #    此种情况表明 当前点 一定与 前一个点 同边缘段，故而我们迭代更新 tempPoints 的第一个点 (tempPoints 也只有一个点)
                if cur_isRealEdge and Point(i, j) not in self.intersectionMark:
                    tempPoints[0] = Point(i, j)
                    continue
                # 4. 若 当前点 是真实边缘点，且其还为 交点 (是否为 终点 不影响处理)(要么是 单个中断点，要么是 中断线 的两个端点)
                # 注意，两个临近点都是交点时，我们会加入由这两个交点构成的 新的边缘段，似乎对结果没有影响
                if cur_isRealEdge and Point(i, j) in self.intersectionMark:
                    # 此时，考虑 tempPoints 的长度
                    #      c1. 若为 0，则表明 tempPoints 中没有点，则加入 当前点，然后去判断下一个点 (此种情形实际只有 起点 为 真实边缘点 的情况下才可能发生)
                    #      c2. 若为 1，则表明 tempPoints 中只有 一个点，该点只有两种可能：1. 非真实边缘点 的 网格交点，2. 真实边缘段与网格线的 交点
                    #             注意：当前点 是否为 终点 影响 tempPoints 的更新
                    #             cc1. 若此唯一的点为 非真实边缘 的 网格交点，则此时应向 tempPoints 加入 当前点生成新的边缘段，然后更新 tempPoints，然后去判断下一个点
                    #                 ccc1.只能是 (起点(非真实边缘点) + 当前点(真实边缘点，且为端点))，只有这种情况需要添加新的边缘段 (当前点不可能为 终点)
                    #                     此情形 即 情形 2 中的 c2 cc1
                    #                 ccc2.或者是 (非起点网格交点(非真实边缘点) + 当前点(真实边缘点，且为端点))，这种情况甚至不需要添加新的边缘段 (当前点可能为 终点)
                    #             cc2. 若此唯一的点为 真实边缘交点 (当前点 是否为 起点 不影响)
                    #                 ccc1.只能是 (该点(真实边缘点) + 当前点(真实边缘点，且为端点)，属于 同一边缘段 时)，更新 tempPoints，然后去判断下一个点
                    #                 ccc2.或者是 (该点(真实边缘点) + 当前点(真实边缘点，且为端点)，不属于 同一边缘段 时)，向 tempPoints 加入 当前点生成新的边缘段，然后更新 tempPoints， 然后去判断下一个点
                    #                     (但似乎不加入此边缘段也没有关系，故可直接更新 tempPoints)
                    #             总结说来，实际上只有 c1 cc1 需要特殊处理，然后，所有情况都只需要更新 tempPoints，同时考虑一下当前点是否为 终点
                    #      c3. 若 >1，则表明 tempPoints 的组成为 ((非真实边缘网格交点 or 真实边缘中断点 or 真实边缘段右端点) + 一段非真实边缘网格点)
                    #            注意：得考虑 当前点 是否为 终点
                    #            此时应向 tempPoints 加入 当前点生成新的边缘段，然后更新 tempPoints，然后去判断下一个点
                    # c1
                    if len(tempPoints) == 0:
                        tempPoints.append(Point(i, j))
                        continue
                    # c2
                    if len(tempPoints) == 1:
                        prev_p = tempPoints[0]
                        prev_isRealEdge = self.isRealEdge[prev_p.x, prev_p.y]
                        # cc1 ccc1
                        if not prev_isRealEdge and prev_p.y == 0:
                            tempPoints.append(Point(i, j))
                            self.addES(tempPoints)
                            tempPoints.clear()
                            tempPoints.append(Point(i, j))
                            continue
                        # 其它情况，只需考虑 当前点 是否为 终点
                        else:
                            if j == self.W - 1:
                                tempPoints.clear()
                                continue
                            else:
                                tempPoints[0] = Point(i, j)
                                continue
                    # c3
                    if len(tempPoints) > 1:
                        tempPoints.append(Point(i, j))
                        self.addES(tempPoints)
                        # 考虑 当前点 是否为 终点
                        if j == self.W - 1:
                            tempPoints.clear()
                            continue
                        else:
                            tempPoints.clear()
                            tempPoints.append(Point(i, j))
                            continue
    def addGridW(self):
        tempEM = []  # tempEM 用于第一遍扫描，记录与 网格线 重叠的 真实边缘段
        tempPoints = []  # tempPoints 用于第二遍扫描，记录 网格线 生成的 新的边缘段
        # 考虑纵线，注意 +1
        grid_num_W = 1 + int((self.W + self.gridSize - 1) / self.gridSize)
        for jgrid in range(grid_num_W):
            # 纵坐标，对最后一个网格，注意处理
            j = jgrid * self.gridSize
            if j >= self.W:
                j = self.W - 1

            # 现在先扫描一遍横坐标，看看是否有 真实边缘段 跟网格线的重叠超过 1 的
            #    若重叠为 1，则 当前点 为交点，以该点为中断点，将 真实边缘段 分为 两半
            #    若重叠 >1，则将 真实边缘段 分为 三段
            for i in range(self.H):
                cur_isRealEdge = self.isRealEdge[i, j]
                cur_em = self.edgeMark[i, j]
                # 1. 当 当前点 不是真实边缘点时，同时 tempEM 中没有点，直接去判断下一个点 (是否为 终点 不影响)
                if not cur_isRealEdge and len(tempEM) == 0:
                    continue
                # 2. 当 当前点 不是真实边缘点时，但 tempEM 中有点，则表明重叠的边缘段到尽头了 (是否为 终点 没有影响)
                if not cur_isRealEdge and len(tempEM) > 0:
                    # 此时，考虑 tempEM 的长度情况，即考虑重叠是单个点还是一段
                    #    若重叠只是为 1，则表明为交点，以该点为中断点，将 真实边缘段分为 两半 即可
                    #    若重叠 >1，则需要将 真实边缘段 分为 三段，中间重叠的部分是单独一段
                    prev_em = self.edgeMark[tempEM[-1].x, tempEM[-1].y]
                    if len(tempEM) == 1:
                        # 将 真实边缘段 分为两段
                        self.updateES(prev_em, tempEM[-1])
                    else:
                        # 将 真实边缘段 分为三段
                        self.updateES2(prev_em, tempEM[0], tempEM[-1])
                    # tempEM 清空，然后去判断下一个点
                    tempEM.clear()
                    continue
                # 3. 当 当前点 是真实边缘，但 tempEM 中没有点，则表明 真实边缘段 与 网格线 的重叠第一次出现 (是否为 终点 有影响)
                if cur_isRealEdge and len(tempEM) == 0:
                    # 若不是 终点，则添加该点到 tempEM， 然后去判断下一个点
                    if i != self.H - 1:
                        tempEM.append(Point(i, j))
                        continue
                    # 若是 终点，则表明为 交点 且重叠为 1，应以该点为中断点，将 真实边缘段分为 两半 即可， 然后去判断下一个点
                    else:
                        # 将 真实边缘段 分为两段
                        self.updateES(cur_em, Point(i, j))
                        continue
                # 4. 当 当前点 是真实边缘，且 tempEM 中有点，则有两种情况：
                #    若 当前点 与 tempEM 中的点为 同一边缘段 上的点，表明 真实边缘段 与 网格线 的重叠是一段 (是否为 终点 有影响)
                #    若 当前点 与 tempEM 中的点为 不同边缘段 上的点，此时需要对 tempEM 进行判断 并进行处理 (是否为 终点 有影响)
                if cur_isRealEdge and len(tempEM) > 0:
                    # c1. 若 len(tempEM) == 1，则考虑当前点的不同情形
                    # c2. 若 len(tempEM) > 1，则考虑当前点的不同情形
                    if len(tempEM) == 1:
                        # c1 cc1. 前一个点 和 当前点 都是交点，则更新 tempEM
                        if tempEM[-1] in self.intersectionMark and Point(i, j) in self.intersectionMark:
                            tempEM[-1] = Point(i, j)
                            continue
                        # c1 cc2. 若前一个点是交点，当前点不是交点，则判断是否同边缘段
                        #    若是相同边缘段则 tempEM 加入当前点，继续循环 (是否为终点有影响)
                        #    若不同，则 更新 tempEM，继续循环 (是否为终点有影响)
                        if tempEM[-1] in self.intersectionMark and Point(i, j) not in self.intersectionMark:
                            if cur_em in self.intersectionMark[tempEM[-1]]:
                                tempEM.append(Point(i, j))
                                if i != self.H - 1:
                                    continue
                                else:
                                    self.updateES2(cur_em, tempEM[-1], Point(i, j))  # 注意得使用当前点指向的边缘段
                                    tempEM.clear()
                                    continue
                            else:
                                tempEM.clear()
                                if i != self.H - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    self.updateES(cur_em, Point(i, j))
                                    continue
                        # c1 cc3. 若前一个点不是交点，当前点是交点，则判断是否同边缘段
                        #    若同，则加入当前点分段，更新 tempEM，继续循环 (是否为终点有影响)
                        #    若不同，则分段，更新 tempEM，继续循环 (是否为终点有影响)
                        if tempEM[-1] not in self.intersectionMark and Point(i, j) in self.intersectionMark:
                            prev_em = self.edgeMark[tempEM[-1].x, tempEM[-1].y]
                            if prev_em in self.intersectionMark[Point(i, j)]:
                                tempEM.append(Point(i, j))
                                self.updateES2(prev_em, tempEM[0], tempEM[-1])
                                tempEM.clear()
                                if i != self.H - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    continue
                            else:
                                self.updateES(prev_em, tempEM[-1])
                                tempEM.clear()
                                if i != self.H - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    continue
                        # c1 cc4. 若前一个点和当前点都不是交点，则判断是否同边缘
                        #    若同则加入继续 (是否为终点有影响)
                        #    若不同，则分段，更新 tempEM，继续 (是否为终点有影响)
                        if tempEM[-1] not in self.intersectionMark and Point(i, j) not in self.intersectionMark:
                            prev_em = self.edgeMark[tempEM[-1].x, tempEM[-1].y]
                            if cur_em == prev_em:
                                tempEM.append(Point(i, j))
                                if i != self.H - 1:
                                    continue
                                else:
                                    self.updateES2(cur_em, tempEM[0], tempEM[-1])
                                    tempEM.clear()
                                    continue
                            else:
                                self.updateES(prev_em, tempEM[-1])
                                tempEM.clear()
                                if i != self.H - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    self.updateES(cur_em, Point(i, j))
                                    continue
                    # c2 此时 len(tempEM) > 1
                    #  特别注意，此时不用考虑 tempEM 中最后一个点是否为交点，其一定不为交点，因为若其为交点，则等不到现在处理
                    else:
                        # c2 cc1. 若当前点是交点，则判断是否同边缘段
                        #    若同，则加入当前点，分段，更新 tempEM，继续循环 (是否为终点有影响)
                        #    若不同，则分段 (是否为终点有影响)
                        if Point(i, j) in self.intersectionMark:
                            prev_em = self.edgeMark[tempEM[-1].x, tempEM[-1].y]
                            if prev_em in self.intersectionMark[Point(i, j)]:
                                tempEM.append(Point(i, j))
                                self.updateES2(prev_em, tempEM[0], tempEM[-1])
                                tempEM.clear()
                                if i != self.H - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    continue
                            else:
                                self.updateES2(prev_em, tempEM[0], tempEM[-1])
                                tempEM.clear()
                                if i != self.H - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    self.updateES(cur_em, Point(i, j))
                                    continue
                        # c2 cc2. 若当前点不是交点，则判断是否同边缘段
                        #    若同，则加入当前点，继续循环 (是否为终点有影响)
                        #    若不同，则分段，更新 tempEM，继续循环 (是否为终点有影响)
                        else:
                            prev_em = self.edgeMark[tempEM[-1].x, tempEM[-1].y]
                            if prev_em == cur_em:
                                tempEM.append(Point(i, j))
                                if i != self.H - 1:
                                    continue
                                else:
                                    self.updateES2(cur_em, tempEM[0], tempEM[-1])
                                    tempEM.clear()
                                    continue
                            else:
                                self.updateES2(prev_em, tempEM[0], tempEM[-1])
                                tempEM.clear()
                                if i != self.H - 1:
                                    tempEM.append(Point(i, j))
                                    continue
                                else:
                                    self.updateES(cur_em, Point(i, j))
                                    continue



                    prev_em = self.edgeMark[tempEM[-1].x, tempEM[-1].y]
                    isSameES = False
                    if prev_em == cur_em:
                        isSameES = True
                    if Point(i, j) in self.intersectionMark and prev_em in self.intersectionMark[Point(i, j)]:
                        isSameES = True
                        self.edgeMark[i, j] = prev_em
                    if tempEM[-1] in self.intersectionMark and cur_em in self.intersectionMark[tempEM[-1]]:
                        isSameES = True
                    # c1. 当前点 与 tempEM 中的点为 同一边缘段 上的点，且 当前点 不是交点
                    #     则添加该点到 tempEM，然后根据是否为 终点 进行操作：非终点继续，终点则分三段
                    if isSameES and Point(i, j) not in self.intersectionMark:
                        tempEM.append(Point(i, j))
                        if i != self.H - 1:
                            continue
                        else:
                            # 将 真实边缘段 分为三段
                            self.updateES2(cur_em, tempEM[0], tempEM[-1])
                            # tempEM 清空，然后去判断下一个点
                            tempEM.clear()
                            continue
                    # c2. 当前点 与 tempEM 中的点为 同一边缘段 上的点，且 当前点 是交点
                    #     则添加该点到 tempEM， 然后根据是否为 终点 进行操作：非终点 分三段，加入 当前点；终点则分三段
                    if isSameES and Point(i, j) in self.intersectionMark:
                        tempEM.append(Point(i, j))
                        # 将 真实边缘段 分为三段
                        self.updateES2(cur_em, tempEM[0], tempEM[-1])
                        # tempEM 清空，然后去判断下一个点
                        tempEM.clear()
                        if i != self.H - 1:
                            # tempEM 加入当前点
                            tempEM.append(Point(i, j))
                            continue
                        else:
                            continue
                    # c3. 当前点 与 tempEM 中的点非 同一边缘段 上的点，且 当前点 不是交点
                    #     则根据 tempEM 的长度来划分：长度为 1，则划分两段；长度 >1，则划分三段
                    #     再根据是否为终点进行操作：非终点，则 tempEM 更新，继续；终点则 tempEM 清空，当前位置处分为两段，继续
                    if not isSameES and Point(i, j) not in self.intersectionMark:
                        if len(tempEM) == 1:
                            # 将 真实边缘段 分为两段
                            self.updateES(prev_em, tempEM[-1])
                        else:
                            # 将 真实边缘段 分为三段
                            self.updateES2(prev_em, tempEM[0], tempEM[-1])
                        if i != self.H - 1:
                            # tempEM 更新，继续
                            tempEM.clear()
                            tempEM.append(Point(i, j))
                            continue
                        else:
                            # tempEM 清空
                            tempEM.clear()
                            # 当前位置处分为两段，继续
                            self.updateES(cur_em, Point(i, j))
                            continue
                    # c4. 当前点 与 tempEM 中的点非 同一边缘段 上的点，且 当前点 是 交点
                    #     则根据 tempEM 的长度来划分：长度为 1，则划分两段；长度 >1，则划分三段
                    #     再根据是否为终点进行操作：非终点，则 tempEM 更新，继续；终点则 tempEM 清空，当前位置处分为两段，继续
                    if not isSameES and Point(i, j) in self.intersectionMark:
                        if len(tempEM) == 1:
                            # 将 真实边缘段 分为两段
                            self.updateES(prev_em, tempEM[-1])
                        else:
                            # 将 真实边缘段 分为三段
                            self.updateES2(prev_em, tempEM[0], tempEM[-1])
                        if i != self.H - 1:
                            # tempEM 更新，继续
                            tempEM.clear()
                            tempEM.append(Point(i, j))
                            continue
                        else:
                            # tempEM 清空
                            tempEM.clear()
                            # 当前位置处分为两段，继续
                            self.updateES(cur_em, Point(i, j))
                            continue

            # 再次扫描横坐标，处理每一段边缘段
            # 实际上，真实边缘点 不用考虑，主要考虑 非真实边缘点，然后要考虑 交点
            for i in range(self.H):
                cur_isRealEdge = self.isRealEdge[i, j]
                if not cur_isRealEdge and i % self.gridSize != 0 and i != self.H - 1:
                    tempPoints.append(Point(i, j))
                    continue
                if not cur_isRealEdge and (i % self.gridSize == 0 or i == self.H - 1):
                    tempPoints.append(Point(i, j))
                    if i == 0:
                        continue
                    self.addES(tempPoints)
                    if i != self.H - 1:
                        tempPoints.clear()
                        tempPoints.append(Point(i, j))
                        continue
                    else:
                        tempPoints.clear()
                        continue
                if cur_isRealEdge and Point(i, j) not in self.intersectionMark:
                    tempPoints[0] = Point(i, j)
                    continue
                if cur_isRealEdge and Point(i, j) in self.intersectionMark:
                    if len(tempPoints) == 0:
                        tempPoints.append(Point(i, j))
                        continue
                    if len(tempPoints) == 1:
                        prev_p = tempPoints[0]
                        prev_isRealEdge = self.isRealEdge[prev_p.x, prev_p.y]
                        if not prev_isRealEdge and prev_p.y == 0:
                            tempPoints.append(Point(i, j))
                            self.addES(tempPoints)
                            tempPoints.clear()
                            tempPoints.append(Point(i, j))
                            continue
                        else:
                            if i == self.H - 1:
                                tempPoints.clear()
                                continue
                            else:
                                tempPoints[0] = Point(i, j)
                                continue
                    if len(tempPoints) > 1:
                        tempPoints.append(Point(i, j))
                        self.addES(tempPoints)
                        if i == self.H - 1:
                            tempPoints.clear()
                            continue
                        else:
                            tempPoints.clear()
                            tempPoints.append(Point(i, j))
                            continue
    def generateSmallEdgeSegment(self, imgName, imgShape, ES, AO):
        ######################################################
        #####  对每一幅图像，进行超像素分割前的准备：准备连接的边缘段  #####
        self.imgName = imgName
        self.H = imgShape[0]
        self.W = imgShape[1]
        self.gridSize = int(np.sqrt(np.floor(self.H * self.W // self.spNum)))  # 不同图像，计算对应的网格大小
        self.ES = ES
        self.ES_num = len(self.ES)  # 当前所用的所有边缘段数量，用于标记 edgeMark intersectionPointMark
        self.AO = AO
        # 使用字典记录某位置是否为交点，且记录是哪些线段的交点
        # 字典的键为：像素点，字典的值为：集合(所有以该点为端点的 边缘段标号)
        self.intersectionMark = {}
        # 遍历边缘段，记录边缘点标号，记录当前位置是否为真实边缘 (包括：真实边缘段上的点 与 两端扩展的点)
        self.edgeMark = -1 * np.ones([self.H, self.W], np.int32)  # 非边缘处，标号全为 -1，边缘段的标号从 0 开始，一直到 self.ES_num - 1
        self.isRealEdge = np.zeros([self.H, self.W], np.int32)  # 真实边缘处为 1，否则为 0
        for esIdx, es in enumerate(self.ES):
            for p in es.points:
                self.edgeMark[p.x, p.y] = esIdx
                self.isRealEdge[p.x, p.y] = 1
        ######################################################

        # 1. 首先，对每一个边缘段，考虑两端，将两端与网格或与其他边缘段相连接
        self.extendTwoEndpoints()

        # 2. 将 网格 的线段和交点加入 ES intersectionPointMark，注意更新 ES_num edgeMark
        self.addGridH()
        self.addGridW()


    def fillRegion(self, startP):
        # 首先创建一个超像素，其 ID 从 1 开始标号
        spregion = SPRegion(self.SPC.createID())

        # 初始生长点为起点，每一轮生长时进行更新
        # 同时，标记 起点，注意，这句 语句 写在这里而非 迭代 中，原因在于判断出四邻域像素点未标记时，立刻标记，与此处逻辑相同
        initSearchPoints = [startP]
        self.isMarked[startP.x, startP.y] = 1
        # 记录当前区域所有封闭边缘段标号和 -1
        esIDs = []
        # 当没有可以进行生长的点时，停止生长
        while len(initSearchPoints) > 0:
            # 扩张区域
            spregion.extendPoints(initSearchPoints)
            # 本轮迭代中，生长的像素点
            extendedPoints = []
            # 对每个搜索点，判断其四邻域，记录扩张区域位置 (即未标记处)
            # 同时，对 边界标号(-1 或者 边缘段标号)，进行记录，注意若该点是边缘段交点，无需记录
            # 注意，搜索点的四邻域不会超过图像范围，这是由我们的边缘段生成算法所保证的(即，图像的四条边界必然是边缘段)
            for p in initSearchPoints:
                # 依次判断 上 下 左 右
                if not self.isMarked[p.x - 1, p.y]:
                    extendedPoints.append(Point(p.x - 1, p.y))
                    self.isMarked[p.x - 1, p.y] = 1
                else:
                    esIDs.append(self.edgeMark[p.x - 1, p.y])

                if not self.isMarked[p.x + 1, p.y]:
                    extendedPoints.append(Point(p.x + 1, p.y))
                    self.isMarked[p.x + 1, p.y] = 1
                else:
                    esIDs.append(self.edgeMark[p.x + 1, p.y])

                if not self.isMarked[p.x, p.y - 1]:
                    extendedPoints.append(Point(p.x, p.y - 1))
                    self.isMarked[p.x, p.y - 1] = 1
                else:
                    esIDs.append(self.edgeMark[p.x, p.y - 1])

                if not self.isMarked[p.x, p.y + 1]:
                    extendedPoints.append(Point(p.x, p.y + 1))
                    self.isMarked[p.x, p.y + 1] = 1
                else:
                    esIDs.append(self.edgeMark[p.x, p.y + 1])

            # 在这一轮迭代之后，更新 初始生长点，开始下一轮迭代
            initSearchPoints = extendedPoints
        # 获得区域对应的轮廓边界标号
        spregion.updateESSet(esIDs)
        return spregion
    def getMiniRegions(self):
        # 获得所有封闭小区域
        # 前提：获得所有的边缘段，这些边缘段可以保证封闭所有区域

        # 首先，标记所有的边缘段像素点，从 edgeMark中直接获得 (比遍历边界获得快得多)
        self.isMarked = copy.copy(self.edgeMark)
        self.isMarked[self.isMarked != -1] = 1
        self.isMarked[self.isMarked == -1] = 0

        # 将交点处的 边缘标记 更新为 -1 可以帮助减少区域扩张对交点判断所花费的时间
        for p in self.intersectionMark.keys():
            self.edgeMark[p.x, p.y] = -1

        # 然后遍历图像，寻找非边缘段点，并从其开始扩张，获得最小区域
        for i in range(self.H):
            for j in range(self.W):
                if not self.isMarked[i, j]:
                    # 从 当前点 开始，生长获得最小封闭区域
                    spregion = self.fillRegion(Point(i, j))
                    # 添加该区域
                    self.SPC.addSP(spregion)
    def FNR(self, pN, M):
        tset = set((pN * M).flatten())
        tset.discard(-1)  # 移除边缘对应的标号，可能上下左右全为区域，并没有边缘
        tset.discard(0)   # 移除不关注区域的标号，或者区域 0(注意，区域 0 需要特别处理，为了处理这种特殊情况，需要将超像素标号从 1 开始标号)
        if len(tset) > 2:
            # print("Test: edge owns more than 2 neighboring regions. It is surprising.")
            pass
        return tset
    def getNeighR(self, p1):
        # 获得当前点的临近区域
        if p1.x == 0:
            tset = self.FNR(self.regionMark[p1.x:p1.x + 2, p1.y - 1:p1.y + 2], self.tM)
        elif p1.x == self.H - 1:
            tset = self.FNR(self.regionMark[p1.x - 1:, p1.y - 1:p1.y + 2], self.bM)
        elif p1.y == 0:
            tset = self.FNR(self.regionMark[p1.x - 1:p1.x + 2, p1.y:p1.y + 2], self.lM)
        elif p1.y == self.W - 1:
            tset = self.FNR(self.regionMark[p1.x - 1:p1.x + 2, p1.y - 1:], self.rM)
        else:
            tset = self.FNR(self.regionMark[p1.x - 1:p1.x + 2, p1.y - 1:p1.y + 2], self.M)
        return tset
    def getAllNeighR(self, esIdx):
        allRSet = set()
        for p in self.ES[esIdx].points[1:-1]:
            tset = self.getNeighR(p)
            allRSet = allRSet | tset
        return allRSet
    def validateES(self):
        # 边界之间没有交点依然可能形成封闭区域 (即，类似边缘并行相贴这种情形)
        # 使用 s 和 e 来控制循环，而非直接迭代 boundaries，原因在于，每次操作有可能改变 boundaries，从而影响迭代
        s = 0
        e = self.ES_num
        count_occurs = 0  # 记录一下看看这种特殊情形出现了多少次
        while s < e:
            if not self.ES[s].isValid:
                s += 1
                continue

            # 边界
            es = self.ES[s]
            # 边界的临近区域集合
            rset = set()
            # 遍历每个点，注意，点一定不会是图片四个角点
            for p1 in es.points[1:-1]:
                # 获得当前点的临近区域集合
                tset = self.getNeighR(p1)
                # 合并
                rset = rset | tset
                # 当三个区域时，停止，这种情形非常少见，故代码繁琐也花费不了太多时间
                # 这里的直觉是：第三个区域的出现，一定是需要出现交点的，此处的交点是两条边缘段上临近的两个点构成，而非一个点
                if len(rset) > 2:
                    count_occurs += 1
                    # 1. 先查找当前边界的所有临近区域
                    # 然后先在该点将边界进行划分
                    updatedRSet = self.getAllNeighR(s)
                    intersections = []
                    intersections.append(p1)
                    # 2. 以该点为中心，考虑其四邻域
                    teSet = set()  # 同一条临近边缘段，只处理一次
                    if p1.x == 0:
                        pass
                    else:
                        cur_em = self.edgeMark[p1.x-1, p1.y]
                        if cur_em != es.ID and cur_em != -1 and Point(p1.x-1, p1.y) not in self.intersectionMark and cur_em not in teSet:
                            teSet.add(cur_em)
                            intersections.append(Point(p1.x-1, p1.y))
                        else:
                            pass
                    if p1.x == self.H - 1:
                        pass
                    else:
                        cur_em = self.edgeMark[p1.x+1, p1.y]
                        if cur_em != es.ID and cur_em != -1 and Point(p1.x+1, p1.y) not in self.intersectionMark and cur_em not in teSet:
                            teSet.add(cur_em)
                            intersections.append(Point(p1.x + 1, p1.y))
                        else:
                            pass
                    if p1.y == 0:
                        pass
                    else:
                        cur_em = self.edgeMark[p1.x, p1.y-1]
                        if cur_em != es.ID and cur_em != -1 and Point(p1.x, p1.y-1) not in self.intersectionMark and cur_em not in teSet:
                            teSet.add(cur_em)
                            intersections.append(Point(p1.x, p1.y-1))
                        else:
                            pass
                    if p1.y == self.W - 1:
                        pass
                    else:
                        cur_em = self.edgeMark[p1.x, p1.y+1]
                        if cur_em != es.ID and cur_em != -1 and Point(p1.x, p1.y-1) not in self.intersectionMark and cur_em not in teSet:
                            teSet.add(cur_em)
                            intersections.append(Point(p1.x, p1.y+1))
                        else:
                            pass
                    for tp in intersections:
                        curRSet = self.getAllNeighR(self.edgeMark[tp.x, tp.y])
                        updatedRSet = updatedRSet | curRSet
                        self.updateES(self.edgeMark[tp.x, tp.y], tp)
                    # 3. 更新区域的临近边缘标号
                    tR = list(updatedRSet)
                    for rID in tR:
                        bSet = set()
                        for p in self.SPC.SPDict[rID].points:
                            if self.edgeMark[p.x - 1, p.y] != -1 and Point(p.x - 1, p.y) not in self.intersectionMark:
                                bSet.add(self.edgeMark[p.x - 1, p.y])
                            if self.edgeMark[p.x + 1, p.y] != -1 and Point(p.x + 1, p.y) not in self.intersectionMark:
                                bSet.add(self.edgeMark[p.x + 1, p.y])
                            if self.edgeMark[p.x, p.y - 1] != -1 and Point(p.x, p.y - 1) not in self.intersectionMark:
                                bSet.add(self.edgeMark[p.x, p.y - 1])
                            if self.edgeMark[p.x, p.y + 1] != -1 and Point(p.x, p.y + 1) not in self.intersectionMark:
                                bSet.add(self.edgeMark[p.x, p.y + 1])
                        self.SPC.SPDict[rID].ESSet = bSet
                    # 4. 停止遍历当前边缘段，开始处理下一个边缘段
                    break
            # 控制循环
            s += 1
            e = self.ES_num
    def getOptimalNeigh_Edge(self, curR):
        # 如果该区域没有边界，则可以想见，该区域应该很小，我们将该区域，随意与附近区域合并
        if len(curR.ESSet) == 0:
            # 搜索半径为 2
            Pp = curR.points[0]
            for p in self.searchPoints:
                # 搜索点在图像范围内
                if 0 <= Pp.x + p.x < self.H and 0 <= Pp.y + p.y < self.W:
                    if self.regionMark[Pp.x+p.x, Pp.y+p.y] != -1:
                        # 返回 区域ID，和需要合并的边缘标号 None
                        return self.regionMark[Pp.x+p.x, Pp.y+p.y], None
                else:
                    continue
            print("Test. Wrong, search out of range.")
        # 现在该区域至少有一条边界，遍历这些边界，寻找最适宜合并的边界和区域
        # 即，合并显著程度最低，尽量长的边缘段
        tESIdx = None
        salience = 1000
        for esIdx in curR.ESSet:
            if not self.ES[esIdx].isValid:
                continue
            if self.ES[esIdx].salience < salience:
                salience = self.ES[esIdx].salience
                tESIdx = esIdx
            elif self.ES[esIdx].salience == salience:
                if len(self.ES[esIdx].points) > len(self.ES[tESIdx].points):
                    tESIdx = esIdx
                else:
                    continue
            else:
                continue
        return self.ES[tESIdx].getNeighRIdx(curR.ID), tESIdx
    def updateRegionMark(self, R):
        for p in R.points:
            self.regionMark[p.x, p.y] = R.ID
    def mergeRegions(self, curR, mergedRID, esIdx):
        # 现在根据不同的情形来合并区域
        # c1. 区域有边界，但该边界只有当前一个临近区域
        #     只需将该边界整合进该区域
        # c2. 区域无边界，此时寻找任意临近区域进行合并
        #     将两个区域融合
        # c3. 区域有边界，该边界有另一临近区域
        #     将区域融合，且将边界整合
        if mergedRID is None and esIdx is not None:
            # 当前的超像素区域失效
            self.SPC.SPDict[curR.ID].isValid = False
            # 创建新的超像素
            r = SPRegion(self.SPC.createID())
            r.points = curR.points[:]
            r.points.extend(self.ES[esIdx].points[1:-1])
            r.size = len(r.points)
            r.ESSet = curR.ESSet
            r.ESSet.remove(esIdx)
            r.NeighSet = curR.NeighSet  # 注意，这里有特殊情况出现，合并该边缘段之后，该区域的另有新的边缘段出现，从而也导致出现新的邻居区域，暂时不处理这个问题
            # 由于新像素产生，导致信息变化，故而更新边缘段信息，更新临近区域的临近区域标号
            self.updateRegionMark(r)
            # 当前区域的合并区域为 None，现在对当前区域的每条边，判断该边的临近区域，是否为当前区域标号，若是，则更新
            # 注意，若该边两侧都为当前区域标号，则都更新为新区域标号
            for ei in r.ESSet:
                if self.ES[ei].firstRIdx == curR.ID:
                    self.ES[ei].firstRIdx = r.ID
                if self.ES[ei].secondRIdx == curR.ID:
                    self.ES[ei].secondRIdx = r.ID
            for tRID in r.NeighSet:
                self.SPC.SPDict[tRID].NeighSet.remove(curR.ID)
                self.SPC.SPDict[tRID].NeighSet.add(r.ID)
            self.ES[esIdx].isValid = False  # 注意该边缘段已经被整合了
            self.intersectionMark[self.ES[esIdx].getFront()].remove(esIdx)  # 交点，更新其相交的边缘段
            self.intersectionMark[self.ES[esIdx].getBack()].discard(esIdx)  # 交点，更新其相交的边缘段，用 dicard 避免环情形
            # 添加新的超像素
            self.SPC.addSP(r)
        elif mergedRID is not None and esIdx is None:
            # 当前的超像素区域失效
            self.SPC.SPDict[curR.ID].isValid = False
            # 需要合并的超像素区域失效
            self.SPC.SPDict[mergedRID].isValid = False
            mergedR = self.SPC.SPDict[mergedRID]
            # 创建新的超像素
            r = SPRegion(self.SPC.createID())
            r.points = curR.points[:] + mergedR.points[:]
            r.size = len(r.points)
            r.ESSet = curR.ESSet | mergedR.ESSet
            r.NeighSet = curR.NeighSet | mergedR.NeighSet
            # 由于新像素产生，导致信息变化，故而更新边缘段信息，更新临近区域的临近区域标号
            self.updateRegionMark(r)
            # 更新新区域的边缘标号
            # 当前区域与合并区域没有合并边缘段，故而来自当前区域的边缘段只有当前区域标号，来自合并区域的边缘段只有合并区域标号
            # 注意，若当前区域或者合并区域存在特殊边缘段：即该边缘段两侧都为该区域，此种情形应该其两侧区域都应更新
            for ei in r.ESSet:
                if self.ES[ei].firstRIdx == curR.ID:
                    self.ES[ei].firstRIdx = r.ID
                if self.ES[ei].secondRIdx == curR.ID:
                    self.ES[ei].secondRIdx = r.ID
                if self.ES[ei].firstRIdx == mergedRID:
                    self.ES[ei].firstRIdx = r.ID
                if self.ES[ei].secondRIdx == mergedRID:
                    self.ES[ei].secondRIdx = r.ID
            for tRID in r.NeighSet:
                self.SPC.SPDict[tRID].NeighSet.discard(curR.ID)
                self.SPC.SPDict[tRID].NeighSet.discard(mergedR.ID)
                self.SPC.SPDict[tRID].NeighSet.add(r.ID)
            # 注意，区域合并之后，可能会出现，两个边缘段该合并的情形，暂时不考虑
            # 添加新的超像素
            self.SPC.addSP(r)
        else:
            # 若当前区域与合并区域为同一区域：即该边缘段在该区域内部
            if curR.ID == mergedRID:
                # 当前的超像素区域失效
                self.SPC.SPDict[curR.ID].isValid = False
                # 创建新的超像素
                r = SPRegion(self.SPC.createID())
                r.points = curR.points[:]
                r.points.extend(self.ES[esIdx].points[1:-1])
                r.size = len(r.points)
                r.ESSet = curR.ESSet
                r.ESSet.remove(esIdx)
                r.NeighSet = curR.NeighSet  # 注意，这里有特殊情况出现，合并该边缘段之后，该区域的另有新的边缘段出现，从而也导致出现新的邻居区域，暂时不处理这个问题
                # 由于新像素产生，导致信息变化，故而更新边缘段信息，更新临近区域的临近区域标号
                self.updateRegionMark(r)
                # 当前区域的合并区域为 None，现在对当前区域的每条边，判断该边的临近区域，是否为当前区域标号，若是，则更新
                # 注意，若该边两侧都为当前区域标号，则都更新为新区域标号
                for ei in r.ESSet:
                    if self.ES[ei].firstRIdx == curR.ID:
                        self.ES[ei].firstRIdx = r.ID
                    if self.ES[ei].secondRIdx == curR.ID:
                        self.ES[ei].secondRIdx = r.ID
                for tRID in r.NeighSet:
                    self.SPC.SPDict[tRID].NeighSet.remove(curR.ID)
                    self.SPC.SPDict[tRID].NeighSet.add(r.ID)
                self.ES[esIdx].isValid = False  # 注意该边缘段已经被整合了
                self.intersectionMark[self.ES[esIdx].getFront()].remove(esIdx)  # 交点，更新其相交的边缘段
                self.intersectionMark[self.ES[esIdx].getBack()].discard(esIdx)  # 交点，更新其相交的边缘段，用 dicard 避免环情形
                # 添加新的超像素
                self.SPC.addSP(r)
            # 此时，当前区域与合并区域是不同的区域
            else:
                # 当前的超像素区域失效
                self.SPC.SPDict[curR.ID].isValid = False
                # 需要合并的超像素区域失效
                self.SPC.SPDict[mergedRID].isValid = False
                mergedR = self.SPC.SPDict[mergedRID]
                # 创建新的超像素
                r = SPRegion(self.SPC.createID())
                r.points = curR.points[:] + mergedR.points[:]
                r.points.extend(self.ES[esIdx].points[1:-1])
                r.size = len(r.points)
                r.ESSet = curR.ESSet | mergedR.ESSet
                r.ESSet.remove(esIdx)  # 需要移除合并的这条边缘段
                r.NeighSet = curR.NeighSet | mergedR.NeighSet
                r.NeighSet.remove(curR.ID)  # 需要移除合并的区域标号
                r.NeighSet.remove(mergedR.ID)  # 需要移除合并的区域标号
                # 由于新像素产生，导致信息变化，故而更新边缘段信息，更新临近区域的临近区域标号
                self.updateRegionMark(r)
                # 更新新区域的边缘标号
                # 注意，若当前区域或者合并区域存在特殊边缘段：即该边缘段两侧都为该区域，此种情形应该其两侧区域都应更新
                # 注意，若当前边缘段只是这两个区域共同边缘段的其中一段，则两侧区域都该更新为新区域标号
                for ei in r.ESSet:
                    if self.ES[ei].firstRIdx == curR.ID:
                        self.ES[ei].firstRIdx = r.ID
                    if self.ES[ei].secondRIdx == curR.ID:
                        self.ES[ei].secondRIdx = r.ID
                    if self.ES[ei].firstRIdx == mergedRID:
                        self.ES[ei].firstRIdx = r.ID
                    if self.ES[ei].secondRIdx == mergedRID:
                        self.ES[ei].secondRIdx = r.ID
                for tRID in r.NeighSet:
                    self.SPC.SPDict[tRID].NeighSet.discard(curR.ID)
                    self.SPC.SPDict[tRID].NeighSet.discard(mergedR.ID)
                    self.SPC.SPDict[tRID].NeighSet.add(r.ID)
                self.ES[esIdx].isValid = False  # 注意该边缘段已经被整合了
                self.intersectionMark[self.ES[esIdx].getFront()].remove(esIdx)  # 交点，更新其相交的边缘段
                self.intersectionMark[self.ES[esIdx].getBack()].discard(esIdx)  # 交点，更新其相交的边缘段，用 dicard 避免环情形
                # 注意，区域合并之后，可能会出现，两个边缘段该合并的情形，暂时不考虑
                # 添加新的超像素
                self.SPC.addSP(r)
    def mergeEdge(self):
        # 注意，这里只是将边缘都给了其第一个分开的区域，并没有认真考虑边缘归属问题
        # 注意，这里将那种两侧都为同一区域的边缘，置为了 False
        for idx, es in enumerate(self.ES):
            if es.isValid:
                if es.firstRIdx is not None and es.firstRIdx == es.secondRIdx:
                    self.SPC.SPDict[es.firstRIdx].addPoints(es.points[1:-1])
                    self.ES[idx].isValid = False
                elif es.firstRIdx is not None:
                    self.SPC.SPDict[es.firstRIdx].addPoints(es.points[1:-1])
    def mergeIntersection(self, regionMark):
        # 这里的思路是，对每个没有分配区域的边缘点，遍历其八邻域，找出其领域区域最多的区域，加入该区域即可
        for v in self.intersectionMark.keys():
            points = []
            if v.x - 1 >= 0 and v.y - 1 >= 0:
                points.append(Point(v.x-1, v.y-1))
            if v.x >= 0 and v.y - 1 >= 0:
                points.append(Point(v.x, v.y-1))
            if v.x + 1 < self.H and v.y - 1 >= 0:
                points.append(Point(v.x + 1, v.y-1))
            if v.x-1 >= 0 and v.y >= 0:
                points.append(Point(v.x-1, v.y))
            if v.x+1 < self.H and v.y >= 0:
                points.append(Point(v.x+1, v.y))
            if v.x-1 >= 0 and v.y + 1 < self.W:
                points.append(Point(v.x-1, v.y+1))
            if v.x >= 0 and v.y + 1 < self.W:
                points.append(Point(v.x, v.y+1))
            if v.x + 1 < self.H and v.y + 1 < self.W:
                points.append(Point(v.x+1, v.y+1))

            logMaxHit = {}
            spSet = set()
            for p in points:
                if regionMark[p.x, p.y] != -1:
                    sp_ID = regionMark[p.x, p.y]
                    if sp_ID not in spSet:
                        spSet.add(sp_ID)
                        logMaxHit[sp_ID] = 1
                    else:
                        logMaxHit[sp_ID] = logMaxHit[sp_ID] + 1

            tempID = None
            maxHit = 0
            for sp_ID in spSet:
                if logMaxHit[sp_ID] > maxHit:
                    maxHit = logMaxHit[sp_ID]
                    tempID = sp_ID

            # 如果8邻域全是交点，则不处理，否则将该点加入对应区域
            if tempID is None:
                print("This is special, if occurs, solve this problem.")
            else:
                self.SPC.SPDict[tempID].addPoints([v])
    def generateSPs(self, primaryImg):
        ##########################
        ##### 准备 #####
        self.SPC.clear()
        ##########################

        ##########
        # 1. 最小区域生成
        #    遍历像素点，若空白点，则创建超像素区域，并区域扩张
        #    区域像素点计数，对应的特征向量整合
        #    同时获得区域 边界集合 交点集合
        #    注意 交点 对应的边界标号不加入 边界集合
        ##########
        # 首先获得最小封闭区域，这些区域除了 临近区域集合 信息未更新，其余所有信息已完整
        Stime = time.time()
        self.getMiniRegions()
        # 接着，获得区域标记，边缘处为 -1
        self.regionMark = - np.ones([self.H, self.W], np.int32)
        for sp in self.SPC.SPDict.values():
            for p in sp.points:
                self.regionMark[p.x, p.y] = sp.ID

        ##########
        # 2. 遍历边界，利用边界拓扑，合理划分边界
        # 此处的假设是：某一条边，其一定临近两个区域
        # 注意：这个假设不一定正确，例如四周边缘，和边缘与网格之间恰好没有区域的情形
        #      故可能只有一个区域，也可能 0 个区域
        #      另外，也可能形成 3 个区域甚至 k 个区域
        #         x      1      x
        #           x    1    x
        #         2   x x x x   3
        #       o o o o o o o o o o
        # 一个边界有三个临近甚至 k个临近区域的原因，如上图所示，边缘该有分段，而实际没有分段
        # 注意：另一种可能是边缘段45度交叉造成
        ##########
        self.validateES()

        ##########
        # 3. 遍历区域
        #    对区域的 边界集合 中的边界，赋予该边界其邻接区域
        ##########
        for sp in self.SPC.SPDict.values():
            for esIdx in sp.ESSet:
                self.ES[esIdx].setRegionIdx(sp.ID)

        ##########
        # 4. 遍历区域的所有边界，通过边界来获取当前区域的所有临近区域
        #    获得当前区域所有临近区域
        ##########
        for sp in self.SPC.SPDict.values():
            # 存放所有区域标号，可能包括 None，包括本区域 标号，邻居区域标号
            t = []
            for esIdx in sp.ESSet:
                t.append(self.ES[esIdx].firstRIdx)
                t.append(self.ES[esIdx].secondRIdx)
            self.SPC.SPDict[sp.ID].updateNeighSet(t)

        ##########
        # 5. 对超像素区域进行合并，合并规则如下：
        #    首先，对小于最小面积要求的超像素区域进行合并 (此时，其实可能合并出现大于要求的超像素)
        #    然后，按照边缘的强弱，对符合大小要求的区域进行合并
        ##########
        # 开始区域合并
        curRIdx = 0
        countMerged = self.SPC.getSize() - self.spNum
        while countMerged > 0:
            # 如果区域标号超过了区域排序列表的大小，停止循环，这种情形应该不会出现
            if curRIdx > self.SPC.getSize():
                print("Test. curRID > SP number.")
                break
            # 考虑当前超像素，如果当前超像素是无效的，则考虑下一个超像素
            curRIndicator = self.SPC.SPList[curRIdx]
            curR = self.SPC.SPDict[curRIndicator.ID]
            if not curR.isValid:
                curRIdx += 1
                continue

            # 现在对当前超像素进行处理，获得应该与该超像素进行合并的最优的区域
            # 根据边缘显著程度来进行合并
            mergedRID, mergedESIdx = self.getOptimalNeigh_Edge(curR)

            # 现在根据不同的情形来合并区域
            # c1. 区域有边界，但该边界只有当前一个临近区域
            #     只需将该边界整合进该区域
            # c2. 区域无边界，此时寻找任意临近区域进行合并
            #     将两个区域融合
            # c3. 区域有边界，该边界有另一临近区域
            #     将区域融合，且将边界整合
            # print(curRIdx)
            self.mergeRegions(curR, mergedRID, mergedESIdx)
            # 处理下一个超像素
            curRIdx += 1
            # 合并了一个区域，需要更新合并数量
            if mergedRID is None or curR.ID == mergedRID:
                continue
            else:
                countMerged -= 1

        # 处理边缘的区域归属问题
        self.mergeEdge()

        # 处理交点的区域归属问题
        regionMark = np.zeros([self.H, self.W], np.int32)
        regionMark[:, :] = -1
        for sp in self.SPC.SPDict.values():
            if not sp.isValid:
                continue
            # 对该区域每个像素点进行标记
            for p in sp.points:
                regionMark[p.x, p.y] = sp.ID
        self.mergeIntersection(regionMark)

    def visuSP(self, priImg):
        spIdx = np.zeros([self.H, self.W], np.int32)
        for sp in self.SPC.SPDict.values():
            if not sp.isValid:
                continue
            # 该区域颜色
            for p in sp.points:
                spIdx[p.x, p.y] = sp.ID
        # visuImg = mark_boundaries(priImg, visuImg, color=(1, 1, 1))
        visuImg = mark_boundaries(priImg, spIdx, color=(1, 1, 0.2))
        return visuImg, spIdx



def getPoints(p, tp, o, dis=23):
    # p: 端点
    # tp: 倒数第三个端点
    # o: 朝向
    # dis: 延伸的距离

    # 第三个端点到端点的向量(dx, dy)
    dx = p.x - tp.x
    dy = p.y - tp.y
    theta = np.arctan2(dy, dx)
    objP = Point()
    objP.x = int(p.x + 0.5 + dis * np.cos(theta))
    objP.y = int(p.y + 0.5 + dis * np.sin(theta))

    es = dda(p.x, p.y, objP.x, objP.y)
    return es.points

