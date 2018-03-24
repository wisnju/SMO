import random
import numpy as np

def loadDataSet(fileName):
    """
    loadDataSet(fileName)
    用于从文件中提取数据集
    :param fileName:
    :return datMat, labelMat:
    """
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')      # 移除首尾空格后按照tab分开
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


class OptStructure:
    """
    用于存储训练数据的对象
    """
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.dataMatrix = np.mat(dataMatIn)
        self.labelMat = np.mat(classLabels).transpose()
        self.C = C
        self.tol = toler
        self.m, self.n = np.shape(self.dataMatrix)
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))


def calcEk(oS, k):
    """
    用于计算并第k个样本的误差Ek
    :param oS:
    :param k:
    :return:
    """
    fxk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.dataMatrix * oS.dataMatrix[k,:].T)) + oS.b
    Ek = fxk - float(oS.labelMat[k])
    return Ek


def selectJrand(i, m):
    """
    随机选择第二个alpha值
    :param i:
    :param m:
    :return j:
    """
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def selectJ(oS, i, Ei):
    """
    内循环采用启发式方法选择alphaJ,使得Ei-Ej取得最大值
    如果第一次进行选择，则采用随机选择selectJrand
    :param i:
    :param oS:
    :param Ei:
    :return:
    """
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcache = np.nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcache) > 0:
        for k in validEcache:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxDeltaE = deltaE
                maxK = k
                Ej = Ek
    else:
        maxK = selectJrand(i, oS.m)
        Ej = calcEk(oS, maxK)
    return maxK, Ej


def updateEk(oS, k):
    """
    用于更新Ek并存储
    :param oS:
    :param k:
    :return:
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def clipAlpha(aj, H, L):
    """
    用于限制aj的范围
    :param aj:
    :param H:
    :param L:
    :return:
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def innerloop(i, oS):
    """
    smo内循环优化
    :param i:选定的第一个alpha
    :param oS:
    :return:
    """
    Ei = calcEk(oS, i)
    # 判断是否满足kkt条件和容错范围
    if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or \
            (oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
        # 选取第二个alpha
        j, Ej = selectJ(oS, i, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # 计算上下边界限制
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        # alphaJ更新
        eta = -2.0 * oS.dataMatrix[i, :] * oS.dataMatrix[j, :].T + oS.dataMatrix[i, :] * oS.dataMatrix[i, :].T + \
              oS.dataMatrix[j, :] * oS.dataMatrix[j, :].T
        if eta <= 0:
            print("eta <= 0")
            return 0
        # 更新alphaJ
        oS.alphas[j] += oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j is not moving enough")
            return 0
        # 更新alphaI
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (oS.alphas[j] - alphaJold)
        # 更新b
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.dataMatrix[i, :] * oS.dataMatrix[i, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.dataMatrix[j, :] * oS.dataMatrix[i, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.dataMatrix[i, :] * oS.dataMatrix[j, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.dataMatrix[j, :] * oS.dataMatrix[j, :].T
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.
        return 1
    else:
        return 0


def smoplatt(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    """
    SMO函数(外循环)，返回alpha和b，alpha[i]>0,对应支持向量
    :param dataMatIn:
    :param classLabels:
    :param C:
    :param toler: 容错率
    :param maxIter:
    :return: alpha和b,构成决策函数
    """
    oS = OptStructure(dataMatIn, classLabels, C, toler)
    iter = 0
    entireSet = True; alphaPairChanged = 0
    while iter < maxIter and (alphaPairChanged > 0 or entireSet):
        alphaPairChanged = 0
        #如果上一次没有alpha值变化，则遍历所有样本，进行优化
        if entireSet:
            for i in range(oS.m):
                alphaPairChanged += innerloop(i, oS)
                print("full set, iter: %d, i: %d, pairs changed: %d" % (iter, i, alphaPairChanged))
            iter += 1
        #如果上一轮遍历确定支持向量，则只对支持变量（0<alpha<C）进行优化
        else:
            nonBound = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A <C))[0]  #取出所有满足条件的非零值的i列表
            for i in nonBound:
                alphaPairChanged += innerloop(i, oS)
                print("Non Bound, iter: %d, i: %d, pairs changed: %d" % (iter, i, alphaPairChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairChanged == 0:
            entireSet = True
        print('iterations number: %d' % iter)
        w = np.mat(np.zeros((oS.n, 1)))
        for i in range(oS.m):
            w += np.multiply(oS.alphas[i] * oS.labelMat[i], oS.dataMatrix[i,:].T)
    return w, oS.b, oS.alphas


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    w, b, alphas = smoplatt(dataArr, labelArr, 0.6, 0.001, 40)
    print(dataArr[11] * w + b, labelArr[11])
