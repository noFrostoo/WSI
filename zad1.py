# author:  Daniel Lipniacki, 304067

import numpy as np
from random import uniform, choice
import sympy as sp
from math import sqrt


# MAXRANDOM = 9223372036854775807
MAXRANDOM = 50
ACCURANCE = 0.001


def main():
    A = None
    B = None
    EnterMaxRandom()
    print("Enter size of Matrix")
    aSize = int(input())
    print("1. Enter Matrix\n2.Random Matrix")
    while A is None:
        choice = int(input())
        if choice == 1:
            print("Enter Matrix")
            A = EnterMatrix(aSize)
            if not CheckMatrix(A):
                print("Wrong Matrix")
                print(A)
                return
        elif choice == 2:
            A = RandomMatrix(aSize)
        else:
            print("Wrong choice")
    print("1. Enter Vector\n2.Random Vector")
    while B is None:
        choice = int(input())
        if choice == 1:
            print("Enter vector")
            B = EnterVector(aSize)
        elif choice == 2:
            B = RandomVector(aSize)
        else:
            print("Wrong choice")
    gradientAlfa, newontAlfa, sgdAlfa = EnterAlfas()
    print(f"A: {A}")
    print(f"B: {B}")
    exp = ProjectFirstPart(B) + ProjectSecondPart(A)
    print(f'Exp: {exp}')
    print(f"Result gradientD: {GradientDecent(exp, aSize, gradientAlfa)}")
    print(f"Result Newton: {NewtonMethode(exp, aSize,newontAlfa)}")
    print(f"Result SGD: {SGD(exp, aSize,sgdAlfa)}")


def EnterMaxRandom():
    print('Enter max random: ')
    newMaxRandom = int(input())
    global MAXRANDOM
    MAXRANDOM = newMaxRandom


def EnterMatrix(size):
    arr = []
    for i in range(size):
        line = []
        for j in range(size):
            line.append(float(input()))
        arr.append(line)
    return np.array(arr)


def RandomMatrix(size):
    arr = []
    for i in range(size):
        line = []
        for j in range(size):
            line.append(0)
        arr.append(line)
    for i in range(size):
        for j in range(i+1):
            arr[i][j] = uniform(-MAXRANDOM, MAXRANDOM)
    matrix = np.array(arr)
    return matrix.dot(np.transpose(matrix))


def EnterVector(size):
    arr = [0] * size
    for i in range(size):
        arr[i] = float(input())
    return np.array(arr)


def RandomVector(size):
    arr = [0] * size
    for i in range(size):
        arr[i] = uniform(-MAXRANDOM, MAXRANDOM)
    return np.array(arr)


def EnterAlfas():
    print("If your alfa is an expression then use steps as variable")
    print("Enter alfa for Gradient Decent: ")
    gradientDecent = input()
    print("Enter alfa for Netwon Methode: ")
    newton = input()
    print("Enter alfa for SGD: ")
    sgd = input()
    return gradientDecent, newton, sgd


def CheckMatrix(A):
    if not CheckSymetry(A):
        return False
    return CheckPositivity(A)


def CheckSymetry(A) -> bool:
    return (A == np.transpose(A)).all()


def CheckPositivity(A):
    if A[0][0] <= 0:
        return False
    for i in range(2, A.shape[0]+1):
        minor = np.empty((i, i))
        for y in range(i):
            for x in range(i):
                minor[y][x] = A[y][x]
        if np.linalg.det(minor) <= 0:
            print(f"Minor was negative: {minor}")
            return False
    return True


def ProjectSecondPart(A):
    size = A.shape[0]
    arr = [0] * size
    result = 1
    for i in range(size):
        arr[i] = sp.Symbol(f'x{i+1}')
    xVector = np.array(arr)
    for i in range(size):
        for j in range(size):
            result += A[i][j]*xVector[i]*xVector[j]
    return 0.5*(result - 1)


def ProjectFirstPart(B):
    size = B.shape[0]
    arr = [0] * size
    for i in range(size):
        arr[i] = sp.Symbol(f'x{i+1}')
    xVector = np.array(arr)
    return xVector.dot(np.transpose(B))


def CalGradient(exp, x: dict):
    """
    exp - a sympy function
    x - a dict of sympy symbols and with curent x
    """
    gradient = []
    for i in x.keys():
        derivative = sp.diff(exp, i)
        gradient.append(derivative.evalf(subs=x))
    return gradient


def CalHassian(exp, x: dict):
    """
    exp - a sympy function
    x - a dict of sympy symbols and with curent x
    """
    hassian = []
    for first in x.keys():
        hassianLine = []
        for second in x.keys():
            derivativeFirst = sp.diff(exp, second)
            derivativeSecond = sp.diff(derivativeFirst, first)
            hassianLine.append(derivativeSecond.evalf(subs=x))
        hassian.append(hassianLine)
    return hassian


def GradientDecent(exp, size, alfa='1'):
    print("GradientDecent")
    print(f"alfa: {alfa}")
    staringVector = RandomVector(size)
    print(f"Starting vector: {staringVector}")
    # last_x = None
    x = {}
    for i in range(size):
        x[sp.Symbol(f'x{i+1}')] = staringVector[i]
    steps = 0
    while steps < 5000:
        alfaInner = eval(alfa)
        gradient = CalGradient(exp, x)
        # print(f"gradient: {gradient}")
        # print(f"X: {x}, prev_x: {last_x}")
        # last_x = x.copy()
        SubFromX(x, gradient, alfaInner)
        steps += 1
    print(f"Steps {steps}")
    return x


def NewtonMethode(exp, size, alfa='1'):
    print("Newton")
    print(f"alfa: {alfa}")
    staringVector = RandomVector(size)
    print(f"Starting vector: {staringVector}")
    last_x = None
    x = {}
    for i in range(size):
        x[sp.Symbol(f'x{i+1}')] = staringVector[i]
    steps = 0
    while Stop(x, last_x) and steps < 5000:
        alfaInner = eval(alfa)
        hassian = CalHassian(exp, x)
        invHassian = np.linalg.inv(np.array(hassian, dtype='float'))
        gradient = CalGradient(exp, x)
        to_add = invHassian.dot(gradient)
        # print(f"Gradient: {gradient}")
        # print(f"Inv Hassian: {invHassian}")
        # print(f"X: {x}, prev_x: {last_x}")
        last_x = x.copy()
        SubFromX(x, to_add, alfaInner)
        steps += 1
    print(f"Steps: {steps}")
    return x


def SGD(exp, size, alfa='1'):
    print("SGD")
    print(f"alfa: {alfa}")
    staringVector = RandomVector(size)
    print(f"Starting vector: {staringVector}")
    x = {}
    for i in range(size):
        x[sp.Symbol(f'x{i+1}')] = staringVector[i]
    steps = 0
    for i in range(5000):
        alfaInner = eval(alfa)
        chosenElement = choice(exp.args)
        gradient = CalGradient(chosenElement, x)
        # print(f"Chosen Element: {chosenElement}")
        # print(f"Gradient: {gradient}")
        # print(f"X: {x}, prev_x: ")
        SubFromX(x, gradient, alfaInner)
        steps += 1
    return x


def SubFromX(x: dict, to_add: list, alfa: float):
    for i in range(len(to_add)):
        x[sp.Symbol(f'x{i+1}')] -= alfa * to_add[i]


def Stop(x: dict, prev_x: dict) -> bool:
    if prev_x is None:
        return True
    lenFromLastXtoX = GetLenOfVector(SubVectorsInDict(x, prev_x))
    return abs(lenFromLastXtoX) > ACCURANCE


def SubVectorsInDict(dict1, dict2) -> list:
    exitArr = []
    for i in range(len(dict1)):
        exitArr.append(dict1[sp.Symbol(f'x{i+1}')] -
                       dict2[sp.Symbol(f'x{i+1}')])
    return exitArr


def GetLenOfVector(vec: list):
    sum = 0
    for num in vec:
        sum += num**2
    return sqrt(sum)


if __name__ == "__main__":
    main()
