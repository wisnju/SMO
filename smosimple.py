import random
import numpy as np


def loadDataSet(fileName):
    """
    loadDataSet(fileName)
    用于从文件中提取数据集
    :param fileName:
    :return datMat, labelMat:
    """
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')  # 移除首尾空格后按照tab分开
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


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


def smosimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    SMO函数，返回alpha和b，alpha[i]>0,对应支持向量
    :param dataMatIn:
    :param classLabels:
    :param C:
    :param toler:
    :param maxIter:
    :return: alpha和b,构成决策函数
    """
    dataMatrix = np.mat(dataMatIn);
    labelMat = np.mat(classLabels).transpose()
    b = 0;
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    print(alphas)
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fxi - float(labelMat[i])
            # 判断是否alpha是否可以优化(是否满足kkt条件 )
            if (labelMat[i] * Ei < -toler and alphas[i] < C) or \
                    (labelMat[i] * Ei > toler and alphas[i] > 0):
                # 内循环找到第二个alpha值
                j = selectJrand(i, m)
                fxj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 计算上下边界限制
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H");
                    continue
                # alphaJ更新
                eta = -2.0 * dataMatrix[i, :] * dataMatrix[j, :].T + dataMatrix[i, :] * dataMatrix[i, :].T + \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta <= 0:
                    print("eta <= 0");
                    continue
                # 更新alphaJ
                alphas[j] += labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j is not moving enough");
                    continue
                # 更新alphaI
                alphas[i] += labelMat[j] * labelMat[i] * (alphas[j] - alphaJold)
                # 更新b
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[i, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            if alphaPairsChanged == 0:
                iter += 1
            else:
                iter = 0
            print("iteration number: %d" % iter)
        return b, alphas
