from pomegranate import DiscreteDistribution, ConditionalProbabilityTable
import numpy as np
from typing import Dict


class BNode:
    def __init__(self, distribution, contionalDistibution, name) -> None:
        super().__init__()
        self.parents = []
        self.childeren = []
        self.distribution = distribution
        self.contionalDistibution = contionalDistibution
        self.value = None
        self.name = name
        self.counterOne = 0
        self.counterZero = 0

    def addChild(self, n):
        self.childeren.append(n)

    def addParent(self, n):
        self.parents.append(n)

    def isCondional(self):
        return self.contionalDistibution

    def sample(self):
        return self.distribution.sample()

    def proba(self, x):
        return self.distribution.probability(x)

    def rand(self):
        parentsVals = [x.value for x in self.parents]
        firstSample = parentsVals.copy()
        firstSample.append(0)
        secondSample = parentsVals.copy()
        secondSample.append(1)
        props = []
        props.append(self.distribution.probability(firstSample))
        props.append(self.distribution.probability(secondSample))
        return np.random.choice([0, 1], p=props)

    def getTableLine(self):
        line = []
        for n in self.parents:
            line.append(n.value)
        line.append(self.value)
        return line

    def parentsNotNones(self):
        for n in self.parents:
            if n.value is None:
                return False
        return True

    def ToggleValue(self):
        if self.value is None:
            return None
        if self.value == 1:
            self.value = 0
            return 0
        else:
            self.value = 1
            return 1

    def record(self):
        if self.value == 1:
            self.counterOne += 1
        elif self.value == 0:
            self.counterZero += 1


class RandomNumberGenerator:
    def __init__(self) -> None:
        super().__init__()
        self.nonConditionsalDist = set()
        self.ConditionsalDist = set()
        self.nodes = set()

    def add_edge(self, fromNode: BNode, toNode: BNode) -> None:
        # fromNode - node that s
        toNode.addParent(fromNode)
        fromNode.addChild(toNode)
        if fromNode.isCondional:
            self.ConditionsalDist.add(fromNode)
        else:
            self.nonConditionsalDist.add(fromNode)
        if toNode.isCondional():
            self.ConditionsalDist.add(toNode)
        else:
            self.nonConditionsalDist.add(toNode)
        self.nodes.add(fromNode)
        self.nodes.add(toNode)

    def finalize(self):
        self.parentsList = sorted(self.par)

    def restart(self):
        for n in self.nodes:
            n.value = None
            n.counterOne = 0
            n.counterZero = 0

    def random(self):
        withValues = 0
        for n in self.nonConditionsalDist:
            n.value = self.sample()
            withValues += 1
        changeHappened = 1
        while(changeHappened == 1):
            changeHappened = 0
            for n in self.nonConditionsalDist:
                if n.parentsNotNones() and n.value is not None:
                    n.value = self.rand()
                    changeHappened = 1
                    withValues += 1
        if withValues != len(self.nodes):
            for n in self.nodes:
                if n.value is None:
                    n.value = n.sample()
        return [x.value for x in self.nodes]

    def MonteCaroGibs(self, evidence: Dict, iterations: int) -> list:
        withValues = 0
        evidenceNodes = set()
        changeHappened = 1
        while(changeHappened == 1):
            changeHappened = 0
            for n in self.nodes:
                if evidence.get(n.name) and n.value is None:
                    n.value = evidence.get(n.name)
                    evidenceNodes.add(n)
                    withValues += 1
                    changeHappened = 1
                else:
                    if not n.isCondional() and n.value is None:
                        n.value = n.sample()
                        withValues += 1
                        changeHappened = 1
                    else:
                        if n.parentsNotNones() and n.value is None:
                            n.value = n.rand()
                            withValues += 1
                            changeHappened = 1
        if not withValues == len(self.nodes):
            for n in self.nodes:
                if n.value is None:
                    n.value = n.sample()

        leftNode = self.nodes - evidenceNodes
        leftNode = [x for x in leftNode]
        for i in range(iterations):
            node = np.random.choice(leftNode)
            firstValue = node.value
            markovBlanket = self.getMarkovBlanket(node)
            p = []
            probs = []
            for n in markovBlanket:
                if not n.isCondional():
                    probs.append(n.proba(n.value))
                else:
                    probs.append(n.proba(n.getTableLine()))
            if not node.isCondional():
                probs.append(node.proba(firstValue))
            else:
                probs.append(node.proba(node.getTableLine()))
            p.append(self.mulProbs(probs))
            # propa for second value
            secondValue = node.ToggleValue()
            probs = []
            for n in markovBlanket:
                if not n.isCondional():
                    probs.append(n.proba(n.value))
                else:
                    probs.append(n.proba(n.getTableLine()))
            if not node.isCondional():
                probs.append(node.proba(secondValue))
            else:
                probs.append(node.proba(node.getTableLine()))
            p.append(self.mulProbs(probs))

            alfa = 1
            summ = p[0] + p[1]
            if summ == 0:
                continue
            alfa = 1/(summ)
            p = [alfa*p[0], alfa*p[1]]

            newValue = np.random.choice([firstValue, secondValue], p=p)
            node.value = newValue
            self.recordeState()

        d = {}
        for n in self.nodes:
            d[n.name] = n.counterZero/iterations
            print(n.counterZero)

        return d

    def recordeState(self):
        for n in self.nodes:
            n.record()

    def mulProbs(self, x):
        exitProb = x[0]
        for i in range(1, len(x)):
            exitProb *= x[i]
        return exitProb

    def getMarkovBlanket(self, node):
        blanket = set()
        for n in node.parents:
            blanket.add(n)

        for n in node.childeren:
            blanket.add(n)
            for parN in n.parents:
                if n != node:
                    blanket.add(parN)
        return blanket


first = DiscreteDistribution({1: 1./2, 0: 1./2})
second = DiscreteDistribution({1: 1./2, 0: 1./2})
mainNode = ConditionalProbabilityTable(
        [[1, 1, 1, 0.4],
         [1, 1, 0, 0.6],
         [1, 0, 1, 0.9],
         [1, 0, 0, 0.1],
         [0, 1, 1, 0.9],
         [0, 1, 0, 0.1],
         [0, 0, 1, 0.4],
         [0, 0, 0, 0.6]], [first, second])

four = ConditionalProbabilityTable(
    [
        [1, 1, 0.4],
        [1, 0, 0.6],
        [0, 0, 0.6],
        [0, 1, 0.4]
    ], [mainNode]
)
five = ConditionalProbabilityTable(
    [
        [1, 1, 0.4],
        [1, 0, 0.6],
        [0, 0, 0.6],
        [0, 1, 0.4]
    ], [mainNode]
)

s1 = BNode(first, False, name="first")
s2 = BNode(second, False, name="second")
s3 = BNode(mainNode, True, name="mainNode")
s4 = BNode(four, True, name="dd")
s5 = BNode(five, True, name='ee')

rng = RandomNumberGenerator()
rng.add_edge(s1, s3)
rng.add_edge(s2, s3)
rng.add_edge(s3, s4)
rng.add_edge(s3, s5)

print(rng.random())
rng.restart()
print(rng.MonteCaroGibs({"first": 1, "second": 0}, 1000))
