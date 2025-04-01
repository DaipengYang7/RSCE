import numpy as np

class Point():
    def __init__(self, x=0, y=0):
        self.x = x  # H
        self.y = y  # w

    def getX(self):
        return self.x
    def getY(self):
        return self.y

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        # 如果类可能扩展或涉及其他维度的比较，还可以重载 __hash__ 方法以便支持字典或集合中的操作
        return hash((self.x, self.y))

class EdgeSeg():
    def __init__(self, es=None, salience=0, isRealEdge=True):
        self.ID = None
        self.isValid = True
        if es == None:
            self.points = []
        elif isinstance(es, Point):
            self.points = []
            self.points.append(es)
        elif isinstance(es, list):
            self.points = es
        else:
            print("Error: the initial es is not correct, please check your code.")

        self.salience = salience
        self.isRealEdge = isRealEdge
        self.firstRIdx = None
        self.secondRIdx = None

    def setIsRealEdge(self, isRealEdge):
        self.isRealEdge = isRealEdge

    def setSalience(self, salience):
        self.salience = salience

    def getPoints(self):
        return self.points

    def getLength(self):
        return len(self.points)

    def getFront(self):
        return self.points[0] if self.points else None

    def getBack(self):
        return self.points[-1] if self.points else None

    def extend(self, p):
        if isinstance(p, Point):
            self.points.append(p)
        elif isinstance(p, EdgeSeg):
            self.points = self.points + p.points
        else:
            print("Error: the input parameter is not correct, please check your code.")

    def remove(self):
        if len(self.points) >= 1:
            p = self.points.pop()
            return p
        else:
            return None

    def integrateRightPart(self, es):
        # Reverse the points of the left side
        if len(self.points) >= 2:
            self.points.reverse()
        # Extend the points of the right side
        if len(es.points) > 0:
            self.extend(es)
        # Discard the same points
        t_points = []
        if len(self.points) >= 1:
            t_points.append(self.points[0])  # 先将第一个点加入
            for i in range(0, len(self.points)-1):
                # 重复点的特点在于，它们是相邻的
                # Note that if there are same points, they are adjacent
                if t_points[-1].x != self.points[i].x or t_points[-1].y != self.points[i].y:
                    t_points.append(self.points[i])
            self.points = t_points

    def extendP(self, points, isStartP):
        if isStartP:
            # 在起点一侧扩展一段距离
            # points 离 startP 由近到远
            # 先对 points 倒序，然后合并 points 与当前段的 points，完成
            points.reverse()
            self.points = points + self.points
        else:
            # 在终点一侧扩展一段距离
            # points 离 endP 由近到远
            self.points.extend(points)


    def setRegionIdx(self, idx):
        if self.firstRIdx is None:
            self.firstRIdx = idx
        elif self.secondRIdx is None:
            self.secondRIdx = idx
        else:
            print("Test. Wrong setRegionIdx. There has been two neighboring regions, please check codes.")
    def getNeighRIdx(self, RIdx):
        # regionIdx 一定在 RSet 中，如果不在，则表明有问题
        if self.firstRIdx != RIdx and self.secondRIdx != RIdx:
            print("Test. RIdx is either equal to firstRIdx or secondRIdx.")
        if self.firstRIdx == RIdx:
            return self.secondRIdx
        else:
            return self.firstRIdx

    def updateRIdx(self, RIdx_1, RIdx_2, RIdx):
        if self.firstRIdx == RIdx_1 and self.secondRIdx == RIdx_2:
            # self.isValid = False
            self.firstRIdx = RIdx
            self.secondRIdx = RIdx
            return
        if self.firstRIdx == RIdx_2 and self.secondRIdx == RIdx_1:
            # self.isValid = False
            self.firstRIdx = RIdx
            self.secondRIdx = RIdx
            return
        if self.firstRIdx == RIdx_1 or self.firstRIdx == RIdx_2:
            self.firstRIdx = RIdx
        elif self.secondRIdx == RIdx_1 or self.secondRIdx == RIdx_2:
            self.secondRIdx = RIdx
        else:
            print("Test. Wrong, neither firstRIdx nor secondRIdx is equal to RIdx_1.")

def divideTowSegment(es, point):
    # 以点 point 为断点，分割边缘段
    # 其中，到 point 之前的保留，point 之后的返回
    for idxP, p in enumerate(es.points):
        if p == point:
            # 注意，断点是重复的，两个边缘段都有这个点
            return EdgeSeg(es.points[:idxP + 1], es.salience, es.isRealEdge), EdgeSeg(es.points[idxP:], es.salience, es.isRealEdge)

def dda(h1, w1, h2, w2):
    # A naive way of drawing line by utilizing DDA algorithm
    edgePoints = []
    # If these two points are same, add it and return
    if h1 == h2 and w1 == w2:
        edgePoints.append(Point(h1, w1))
        return EdgeSeg(edgePoints)
    # If these two points are neighbors in 8-neighbors
    if abs(h2 - h1) <= 1 and abs(w2 - w1) <= 1:
        edgePoints.append(Point(h1, w1))
        edgePoints.append(Point(h2, w2))
        return EdgeSeg(edgePoints)
    # Apply the DDA algorithm
    dw = float(w2 - w1)
    dh = float(h2 - h1)
    # Slope
    steps = max(abs(dh), abs(dw))
    # One is equal to 1, and another less than 1
    delta_w = dw / steps
    delta_h = dh / steps
    # Round to the nearest, ensuring that the increments of 'w' and 'h' are less than or equal to 1,
    # to make the generated lines as evenly distributed as possible
    w = w1 + 0.5
    h = h1 + 0.5
    for i in range(int(steps + 1)):
        # Add points
        edgePoints.append(Point(int(h), int(w)))
        w += delta_w
        h += delta_h
    # Note that the DDA algorithm might reorder the points on the edge and requires rearrangement
    if edgePoints[0].x != h1 or edgePoints[0].y != w1:
        edgePoints.reverse()
    return EdgeSeg(edgePoints)

class PosDiff():
    def __init__(self, xDiff=0, yDiff=0):
        self.xDiff = xDiff
        self.yDiff = yDiff

class SearchRegion():
    def __init__(self, length, width, direction):
        self.length = length
        self.width = width
        self.direction = direction
        self.PosDiffs = None

        self.constructRegion()

    def _isLeft(self, Ax, Ay, Bx, By, Px, Py):
        return (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax) > 1e-4
    def _isRight(self, Ax, Ay, Bx, By, Px, Py):
        return (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax) < -1e-4

    def constructRegion(self):
        searchHW = self.length * 2 + 1
        centre = self.length
        halfWidth = float(self.width) / 2.

        # line: y = tan(direction) * x  --> sin(d)x - cos(d)y = 0
        # line equation: A*x + B*y + C = 0
        theta = self.direction / 180 * np.pi
        A = np.sin(theta)
        B = - np.cos(theta)
        C = - (A * centre + B * centre)

        shiftDis = 10.  # The shift for determining the left or right sides
        shiftX = centre + np.cos(theta + np.pi / 2) * shiftDis
        shiftY = centre + np.sin(theta + np.pi / 2) * shiftDis

        self.PosDiffs = []
        for i in range(searchHW):
            for j in range(searchHW):
                # Compute the distance to the line
                # d = (A * x0 + B * y0 + C) / sqrt(A**2 + B**2)
                disToLine = abs(A * i + B * j + C)
                # Get the prejection on the line
                # x = (B**2 * x0 - A * B * y0 - A * C) / (A**2 + B**2)
                # y = (- A * B * x0 + A**2 * y0 - B * C) / (A**2 + B**2)
                cpX = (B * B * i - A * B * j - A * C)
                cpY = (- A * B * i + A * A * j - B * C)
                # Compute the distance between "the prejection point" and "the middle point of the line"
                disToLineCentre = np.sqrt((cpX - centre) ** 2 + (cpY - centre) ** 2)

                if disToLineCentre <= self.length and disToLine <= halfWidth:
                    # 只取右侧区域
                    if self._isRight(float(centre), float(centre), float(shiftX), float(shiftY), float(i), float(j)):
                        self.PosDiffs.append(PosDiff(i - self.length, j - self.length))
        # 将这些位置偏移由近到远排列
        self.PosDiffs.sort(key=lambda pd: pd.xDiff ** 2 + pd.yDiff ** 2)

class LocalLine():
    def __init__(self):
        self.numPattern = 12
        self.length = 3
        self.width = 3
        self.construct()
    def construct(self):
        self.leftLinkMasks = []
        self.rightLinkMasks = []
        for i in range(self.numPattern):
            leftSR = SearchRegion(self.length, self.width, 360. - i * 180. / self.numPattern)
            for k in range(len(leftSR.PosDiffs)):
                leftSR.PosDiffs[k].xDiff += self.length
                leftSR.PosDiffs[k].yDiff += self.length
            self.leftLinkMasks.append(leftSR.PosDiffs)

            rightSR = SearchRegion(self.length, self.width, 180. - i * 180 / self.numPattern)
            for k in range(len(rightSR.PosDiffs)):
                rightSR.PosDiffs[k].xDiff += self.length
                rightSR.PosDiffs[k].yDiff += self.length
            self.rightLinkMasks.append(rightSR.PosDiffs)

def getColors():
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
    return colors