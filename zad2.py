# author:  Daniel Lipniacki, 304067
from math import sin
from random import randint, uniform
import numpy as np


class Evolutionary:

    def __init__(self, populationSize, minBound, maxBound, sigma, dimensions):
        self.populationSize = populationSize
        self.minBound = minBound
        self.maxBound = maxBound
        self.sigma = sigma
        self.dimensions = dimensions

    def main(self):
        self.initiation()
        self.evaluateRun = 0
        self.currentMin = self.min()
        while self.evaluateRun < 100000:
            nextPoulation = self.selection()
            self.mutation(nextPoulation)
            self.sucesion(nextPoulation)
            self.findMinAndSave()
        return self.min()

    def initiation(self):
        self.population = [[uniform(self.minBound, self.maxBound)
                            for i in range(self.dimensions)]
                           for _ in range(self.populationSize)]

    def selection(self):
        t = 0
        nextPoulation = []
        evalArr = self.evaluate()
        while t < self.populationSize/2:
            randA = randint(0, self.populationSize-1)
            randB = randint(0, self.populationSize-1)
            if evalArr[randA] < evalArr[randB]:
                nextPoulation.append(self.population[randA])
            else:
                nextPoulation.append(self.population[randB])
            t += 1
        return nextPoulation

    def mutation(self, nextPoulation):
        nextPopulationSize = len(nextPoulation)
        index = 0
        while nextPopulationSize < self.populationSize:
            goodMutation = False
            subject = None
            while not goodMutation:
                subject = nextPoulation[index] + self.sigma *\
                                    np.random.normal(0, 1, self.dimensions)
                goodMutation = True
                for x in subject:
                    if x > self.maxBound or x < self.minBound:
                        goodMutation = False
            nextPoulation.append(subject)
            nextPopulationSize += 1
            index += 1
        return nextPoulation

    def evaluate(self):
        return [self.expresion(x, 1) for x in self.population]

    def sucesion(self, nextPoulation):
        self.population = nextPoulation

    def expresion(self, x, count):
        p = [7, 6, 0, 4, 0]
        if count:
            self.evaluateRun += 1
        # return sin(2*(x[0]-2))*sin(2*(x[1]-3)) + (x[0]/4)**2 + (x[1]/4)**2
        return 0.05*p[0]*x[0]**2 + 0.04*p[1]*x[1]**2 + 0.03 * p[2] * x[2]**2 +\
            0.02 * p[3] * x[3]**2 + 0.01 * p[4] * x[4]**2 + sin((p[4]+1)*x[0])\
            * sin((p[3]+1)*x[1]) * sin((p[2]+1)*x[2]) * sin((p[1]+1)*x[3]) *\
            sin((p[0]+1)*x[4])
        # return 0.05*7*x[0]**2 + 0.04*6*x[1]**2 + 0.03 * 0 * x[2]**2 + 0.02 *\
        #     4 * x[3]**2 + 0.01 * 0 * x[4]**2 + sin(1*x[0]) * sin(5*x[1]) * \
        #     sin(1*x[2]) * sin(7*x[3]) * sin(8*x[4])

    def min(self):
        xMin = self.population[0]
        m = self.expresion(xMin, 0)
        for x in self.population:
            if self.expresion(x, 0) < m:
                m = self.expresion(x, 0)
                xMin = x
        return xMin

    def findMinAndSave(self):
        minToCheck = self.min()
        if self.expresion(self.currentMin, 0) > self.expresion(minToCheck, 0):
            self.currentMin = minToCheck


if __name__ == "__main__":
    for i in range(50, 800, 50):
        # print(f'pop size{i}')
        i = Evolutionary(100, -5, 5, 0.31, 5)
        i.main()
        # print(i.expresion(i.min(), 0))
        print(F"Min point: {i.currentMin}")
        print(F"J result: {round(i.expresion(i.currentMin, 0), 2)}")
