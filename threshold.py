from __future__ import print_function
import os, sys
import numpy
import matplotlib as mpl
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

MACROWIDTH = 16
MACROHEIGHT = 16

def thresholdImage(image, threshold):
    result = image.point(lambda x: 255 if x > threshold else 0)
    return result

def normalizeImage(im, desiredMean, desiredVariance):
    imarray = numpy.asarray(im)
    mean = numpy.mean(imarray)
    variance = numpy.var(imarray)
    result = imarray
    result.setflags(write=1)
    for i in range(0, len(imarray - 1)):
        for j in range(0, len(imarray[i]) - 1):
            pixVal = numpy.floor((int(imarray[i][j][0]) + imarray[i][j][1] + imarray[i][j][2]) / 3)
            if pixVal > mean:
                newVal = desiredMean + numpy.sqrt((desiredVariance * (numpy.square(pixVal - mean))) / (variance))
                toAppend = [newVal, newVal, newVal]
            else:
                newVal = desiredMean - numpy.sqrt((desiredVariance * (numpy.square(pixVal - mean))) / (variance))
                toAppend = [newVal, newVal, newVal]
            result[i][j] = toAppend
    out = Image.fromarray(result)
    im.paste(out)
    return im

def binarizeMacroblock(im, xMacro, yMacro):
    imarray = numpy.asarray(im)
    macroBlockMean = 0
    result = imarray
    result.setflags(write=1)
    for i in range(MACROWIDTH*xMacro, MACROWIDTH*xMacro + MACROWIDTH):
        for j in range(MACROHEIGHT*yMacro, MACROHEIGHT*yMacro + MACROHEIGHT):
            macroBlockMean += ((int(imarray[i][j][0]) + imarray[i][j][1] + imarray[i][j][2]) / 3)
    macroBlockMean = macroBlockMean / (MACROWIDTH * MACROHEIGHT)
    for i in range(MACROWIDTH*xMacro, MACROWIDTH*xMacro + MACROWIDTH):
        for j in range(MACROHEIGHT*yMacro, MACROHEIGHT*yMacro + MACROHEIGHT):
            if imarray[i][j][0] >= macroBlockMean:
                result[i][j] = [255, 255, 255]
            else:
                result[i][j] = [0, 0, 0]
    out = Image.fromarray(result)
    im.paste(out)
    return im

def verticalSobel(im, xMacro, yMacro):
    imarray = numpy.asarray(im)
    result = []
    cellResult = 0
    operator = [[1, 0, -1],[2, 0, -2],[1, 0, -1]]
    for i in range(MACROWIDTH*xMacro + 1, MACROWIDTH*xMacro + MACROWIDTH-1):
        result.append([])
        for j in range(MACROHEIGHT*yMacro + 1, MACROHEIGHT*yMacro + MACROHEIGHT-1):
            cellResult = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    cellResult += imarray[i + k][j + l][0] * operator[k+1][l+1]
            #cellResult = cellResult / 4
            result[-1].append(cellResult)
    return result




def horizontalSobel(im, xMacro, yMacro):
    imarray = numpy.asarray(im)
    result = []
    cellResult = 0
    operator = [[1, 2, 1],[0, 0, 0],[-1, -2, -1]]
    for i in range(MACROWIDTH*xMacro + 1, MACROWIDTH*xMacro + MACROWIDTH-1):
        result.append([])
        for j in range(MACROHEIGHT*yMacro + 1, MACROHEIGHT*yMacro + MACROHEIGHT-1):
            cellResult = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    cellResult += imarray[i + k][j + l][0] * operator[k+1][l+1]
            #cellResult = cellResult / 4
            result[-1].append(cellResult)
    return result

def ridgeOrientation(vSobel, hSobel):
    result = []
    macroResult = 0
    verticalCells = 0
    vx = 0
    vy = 0
    macroWidth = len(vSobel)
    macroHeight = len(vSobel[0])
    macroInnerWidth = len(vSobel[0][0])
    macroInnerHeight = len(vSobel[0][0][0])
    for i in range(0, macroWidth):
        result.append([])
        
        for j in range(0, macroHeight):
            vx = 0
            vy = 0
            count = 0
            verticalCells = 0
            macroResult = 0
            for k in range (0, macroInnerWidth):
                for l in range(0, macroInnerHeight):
                    #vx += 2 * (vSobel[i][j][k][l]) * (hSobel[i][j][k][l])
                    #vy += numpy.square(vSobel[i][j][k][l]) - numpy.square(hSobel[i][j][k][l])
                    if not (hSobel[i][j][k][l] == 0 and vSobel[i][j][k][l] == 0):
                        count += 1
                    if hSobel[i][j][k][l] != 0:
                        macroResult += (vSobel[i][j][k][l] / hSobel[i][j][k][l])
                    elif hSobel[i][j][k][l] == 0 and vSobel[i][j][k][l] != 0:
                        verticalCells += numpy.abs(vSobel[i][j][k][l]/125)
            #if vx != 0:
                #macroResult = numpy.rad2deg(0.5 * numpy.arctan(vy/vx))
            #else:
                #macroResult = 89
            if macroResult > 0:
                macroResult += (verticalCells * 1)
            elif macroResult < 0:
                macroResult += (verticalCells * -1)
            if count > 0:
                macroResult = macroResult / (count)
            else:
                macroResult = 0
            macroResult = numpy.rad2deg(numpy.arctan(macroResult))
            result[i].append(macroResult)
    return result

def linearRegressionRidgeOrientation(im, xMacro, yMacro):
    imarray = numpy.asarray(im)
    blackRidgesX = []
    blackRidgesY = []
    whiteRidgesX = []
    whiteRidgesY = []
    whiteNum = 0
    blackNum = 0
    result = 0
    for i in range(MACROWIDTH*xMacro + 1, MACROWIDTH*xMacro + MACROWIDTH-1):
        for j in range(MACROHEIGHT*yMacro + 1, MACROHEIGHT*yMacro + MACROHEIGHT-1):
            if imarray[i][j][0] == 0:
                blackRidgesX.append(i % (16))
                blackRidgesY.append(j % (16))
                blackNum += 1
            else:
                whiteRidgesX.append(i % (16))
                whiteRidgesY.append(j % (16))
                whiteNum += 1
    temp1 = [0]
    temp2 = [0]
    if len(blackRidgesX) > 0:
        temp1 = numpy.polyfit(blackRidgesX, blackRidgesY, 1)
    else:
        temp1[0] = 0
    if len(whiteRidgesX) > 0:
        temp2 = numpy.polyfit(whiteRidgesX, whiteRidgesY, 1)
    else:
        temp2[0] = 0
    result = (temp1[0])
    result = numpy.arctan(result)
    result = (numpy.rad2deg(result))
    return result

def printRidgeOrientation(ridgeOrient):
    result = []
    for i in range(len(ridgeOrient)):
        for j in range(len(ridgeOrient[0])):
            m = -(numpy.tan(numpy.deg2rad(ridgeOrient[i][j])))
            #print("SLOPE: ", m, "ANGLE: ", ridgeOrient[i][j])
            x = (j*MACROWIDTH) + MACROWIDTH/2
            y = (i*MACROHEIGHT) + MACROHEIGHT/2
            b = (y) - (m * x)
            d = MACROHEIGHT/2 - 1
            xData = [0, 0]
            xData[0] = ((((2 * x) + (2 * m * y) - (2 * m * b)) - numpy.sqrt(((-2 * x) - (2 * m * y) + (2 * m * b)) ** 2 - (4 * (1 + m**2) * (x**2 + y**2 + b**2 - d**2 - 2*b*y)))) / (2 * (1 + m**2)))
            xData[1] = ((((2 * x) + (2 * m * y) - (2 * m * b)) + numpy.sqrt(((-2 * x) - (2 * m * y) + (2 * m * b)) ** 2 - (4 * (1 + m**2) * (x**2 + y**2 + b**2 - d**2 - 2*b*y)))) / (2 * (1 + m**2)))
            yData = [0, 0]
            yData[0] = (m * xData[0]) + b
            yData[1] = (m * xData[1]) + b
            temp = lines.Line2D(color = "red", xdata = xData, ydata = yData)
            result.append(temp)
    return result

def blackMacroBlock(im, xMacro, yMacro):
    imarray = numpy.asarray(im)
    result = imarray
    result.setflags(write=1)
    for i in range(MACROWIDTH*xMacro, MACROWIDTH*xMacro + MACROWIDTH):
        for j in range(MACROHEIGHT*yMacro, MACROHEIGHT*yMacro + MACROHEIGHT):
            result[i][j] = [0, 0, 0]
    out = Image.fromarray(result)
    im.paste(out)
    return im

def printMacroBlock(im, xMacro, yMacro):
    imarray = numpy.asarray(im)
    for i in range(MACROWIDTH*xMacro, MACROWIDTH*xMacro + MACROWIDTH):
        for j in range(MACROHEIGHT*yMacro, MACROHEIGHT*yMacro + MACROHEIGHT):
            print(imarray[i][j])

def findSingularities(ridgeOrient, threshold):
    result = []
    for i in range(1, len(ridgeOrient)-1):
        for j in range(1, len(ridgeOrient[0])-1):
            tempTotal = 0.0
            temp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            temp[0] = ridgeOrient[i+1][j+1] - ridgeOrient[i+1][j  ]
            temp[1] = ridgeOrient[i  ][j+1] - ridgeOrient[i+1][j+1]
            temp[2] = ridgeOrient[i-1][j+1] - ridgeOrient[i  ][j+1]
            temp[3] = ridgeOrient[i-1][j  ] - ridgeOrient[i-1][j+1]
            temp[4] = ridgeOrient[i-1][j-1] - ridgeOrient[i-1][j  ]
            temp[5] = ridgeOrient[i  ][j-1] - ridgeOrient[i-1][j-1]
            temp[6] = ridgeOrient[i+1][j-1] - ridgeOrient[i  ][j-1]
            temp[7] = ridgeOrient[i+1][j  ] - ridgeOrient[i+1][j-1]
            for t in temp:
                if t > 90:
                    t = t - 180
                elif t <= -90:
                    t = t + 180
                tempTotal += t
            poincare = tempTotal/180
            if poincare >= 1-threshold and poincare <= 1+threshold:
                print ("LOOP FOUND")
                result.append(patches.Circle((j*MACROHEIGHT + MACROHEIGHT/2, i*MACROWIDTH + MACROWIDTH/2), MACROHEIGHT/2, color="blue", fill=False, linewidth=2.0))
            elif  poincare >= 2-threshold and poincare <= 2+threshold:
                print("WHORL FOUND")
                result.append(patches.Circle((j*MACROHEIGHT + MACROHEIGHT/2, i*MACROWIDTH + MACROWIDTH/2), MACROHEIGHT/2, color="yellow", fill=False, linewidth=2.0))
            elif  poincare >= -1-threshold and poincare <= -1+threshold:
                print("DELTA FOUND")
                result.append(patches.Circle((j*MACROHEIGHT + MACROHEIGHT/2, i*MACROWIDTH + MACROWIDTH/2), MACROHEIGHT/2, color="green", fill=False, linewidth=2.0))
    return result

            


for infile in sys.argv[1:]:
    outfile = "test" + ".jpg"
    im = Image.open(infile)
    im.convert("L")
    #im = normalizeImage(im, 100, 100)
    #im = thresholdImage(im, 101)
    (width, height) = im.size
    macroWidth = int(numpy.floor(height/MACROWIDTH))
    macroHeight = int(numpy.floor(width/MACROHEIGHT))
    sobelVert = []
    sobelHor = []
    linearRegression = []
    for i in range(0, macroWidth):
        sobelVert.append([])
        sobelHor.append([])
        linearRegression.append([])
        for j in range(0, macroHeight):
            im = binarizeMacroblock(im, i, j)
            sobelVert[i].append(verticalSobel(im, i, j))
            sobelHor[i].append(horizontalSobel(im, i, j))
            linearRegression[i].append(linearRegressionRidgeOrientation(im, i, j))

    ridgeOr = ridgeOrientation(sobelVert, sobelHor)
    #printMacroBlock(im, 15, 14)
    blackMacroBlock(im, 17, 7)
    print(sobelVert[18][7])
    print(sobelHor[18][7])
    fig, ax = plt.subplots()
    ax.imshow(im)
    #ridgeLines = printRidgeOrientation(linearRegression)
    ridgeLines = printRidgeOrientation(ridgeOr)
    singularityPoints = findSingularities(ridgeOr, 0.05)
    for i in singularityPoints:
        ax.add_patch(i)
    for i in ridgeLines:
        ax.add_line(i)
    plt.show()
    im.save(outfile)

