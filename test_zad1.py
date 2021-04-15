import sympy as sp
from zad1 import CalGradient, SubFromX, CalHassian, CheckPositivity, CheckSymetry, RandomMatrix, CheckMatrix, ProjectFirstPart, ProjectSecondPart
import numpy as np


def test_gradient1():
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    exp = x**2 * y
    gradient = CalGradient(exp, {
        x: 3,
        y: 2
    })
    assert gradient == [12, 9]


def test_gradient2():
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    exp = x * y
    gradient = CalGradient(exp, {
        x: 1,
        y: 1
    })
    assert gradient == [1, 1]


def test_SubFromX():
    x = sp.Symbol("x1")
    y = sp.Symbol("x2")
    xDict = {
        x: 3,
        y: 2
    }
    to_add = [1, 1]
    resultExpected = {
        x: 2,
        y: 1,
    }
    SubFromX(xDict, to_add, 1)
    assert resultExpected[x] == xDict[x]
    assert resultExpected[y] == xDict[y]


def test_SubFromX2():
    x = sp.Symbol("x1")
    y = sp.Symbol("x2")
    xDict = {
        x: 3,
        y: 2
    }
    to_add = [1, 1]
    resultExpected = {
        x: 2.5,
        y: 1.5,
    }
    SubFromX(xDict, to_add, 0.5)
    assert resultExpected[x] == xDict[x]
    assert resultExpected[y] == xDict[y]


def test_hassian():
    x = sp.Symbol("x1")
    y = sp.Symbol("x2")
    exp = x**3 - 2*x*y - y**6
    xDict = {
        x: 1,
        y: 2
    }
    assert CalHassian(exp, xDict) == [[6, -2], [-2, -480]]


def test_hassian1_inv():
    x = sp.Symbol("x1")
    y = sp.Symbol("x2")
    exp = x**3 - 2*x*y - y**6
    xDict = {
        x: 1,
        y: 2
    }
    result = np.linalg.inv(np.array(CalHassian(exp, xDict), dtype='float'))
    assert result[0][0] - 0.16643551 < 0.1
    assert result[0][1] - (-0.002080440) < 0.1
    assert result[1][0] - (-0.00069348) < 0.1
    assert result[1][1] - (-0.00208044) < 0.1


# def test_changeintoarr():
#     x = sp.Symbol("x1")
#     y = sp.Symbol("x2")
#     exp = x**3 - 2*x*y - y**6
#     expArr = ChangeExpIntoArr(exp)
#     assert expArr == [x**3, -2*x*y, -y**6]


def test_check_symetry():
    matrix1 = np.array([[2, 1, 3], [1, 6, 7], [3, 7, 9]])
    matrix2 = np.array([[7, 0], [0, 0]])
    assert CheckSymetry(matrix1) == True
    assert CheckSymetry(matrix2) == True


def test_check_symetry_false():
    matrix1 = np.array([[2, 1, 3], [1, 6, 7], [3, 8, 9]])
    matrix2 = np.array([[7, 0], [1, 0]])
    assert CheckSymetry(matrix1) == False
    assert CheckSymetry(matrix2) == False


def test_check_positivity():
    matrix1 = np.array([[3, 1, -1], [1, 1, 0], [-1, 0, 2]])
    matrix2 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    # matrix3 = np.array([[1, 1, 0], [1, 1, -2], [0, -2, 0]])
    assert CheckPositivity(matrix1) == True
    assert CheckPositivity(matrix2) == True
    # assert CheckPositivity(matrix3) == True


def test_check_positivity_false():
    matrix1 = np.array([[-1, 1, 0], [1, 1, -2], [0, -2, 0]])
    assert CheckPositivity(matrix1) == False


def test_random_matrix():
    for i in range(3000):
        assert CheckMatrix(RandomMatrix(5)) == True

def SetUpSymbols():
    pass
