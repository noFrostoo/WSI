# !/usr/bin/env python3
import math


PLAYER = "X"
AI = "O"
BLANK = "."
WINNINGAMOUNT = 5


class Board():
    def __init__(self, givenBoard):
        self.board = []
        self.winningBoard = False
        self.winner = None
        for y in range(15):
            line = []
            for x in range(15):
                line.append(givenBoard[y][x])
            self.board.append(line)

    @classmethod
    def Emptyboard(cls):
        b = cls.__createBoard()
        return Board(b)

    @classmethod
    def __createBoard(cls):
        board = []
        for y in range(15):
            line = []
            for x in range(15):
                line.append(BLANK)
            board.append(line)
        return board

    def getboard(self):
        return self.board

    def getEmpty(self):
        emp = set()
        for y in range(15):
            for x in range(15):
                if self.board[y][x] == BLANK:
                    emp.add((x, y))
        return emp

    def placeMarker(self, x, y, marker):
        if self.isValidXY(x, y) and self.board[y][x] == BLANK:
            self.board[y][x] = marker
            wonthisround = False
            if(not self.winningBoard):
                self.winningBoard = self.isWinnerboard(x, y, marker)
                wonthisround = True
            if(wonthisround):
                self.winner = marker
            return True
        else:
            return False

    def isValidXY(self, x, y):
        return x >= 0 and x < 15 and y >= 0 and y < 15

    def getPlacesMarker(self, marker):
        places = []
        for y in range(15):
            for x in range(15):
                if self.board[y][x] != BLANK:
                    places.append((x, y))
        return places

    def blankPlacesAround(self, x, y):
        places = set()
        for i in range(-1, 2):
            for j in range(-1, 2):
                if self.isValidXY(x+j, y+i) and self.board[y+i][x+j] == BLANK:
                    places.add((x+j, y+i))
        return places

    def isWinnerboard(self, x, y, marker):
        return self.isRowWin(x, y, marker) or self.isColWin(x, y, marker) or self.isDiagonalWin(x, y, marker)

    def isRowWin(self, x, y, marker):
        count = 0
        for i in range(15):
            if self.board[y][i] == marker:
                count += 1
            else:
                count = 0
            if count == 5:
                return True
        return False

    def isColWin(self, x, y, marker):
        count = 0
        for i in range(15):
            if self.board[i][x] == marker:
                count += 1
            else:
                count = 0
            if count == 5:
                return True
        return False

    def isDiagonalWin(self, x, y, marker):
        count = 0
        for i in range(-5, 5):
            if self.isValidXY(x+i, y+i):
                if self.board[y+i][x+i] == marker:
                    count += 1
                else:
                    count = 0
                if count == 5:
                    return True
        for i in range(-5, 5):
            if self.isValidXY(x+i, y-i):
                if self.board[y-i][x+i] == marker:
                    count += 1
                else:
                    count = 0
                if count == 5:
                    return True
        return False

    def evaluate(self, marker):
        enemy = AI
        if marker == AI:
            enemy = PLAYER
        isWin = 0
        if self.winningBoard and self.winner == marker:
            isWin = 100000
        elif self.winningBoard and self.winner == enemy:
            isWin = -100000
        off2my = (self.longSegmentsRows(2, marker, enemy) + self.longSegmentsCols(2, marker, enemy) + self.longSegmentsDiag(2, marker, enemy) + self.longSegmentsDiagLeft(2, marker, enemy)) * 2
        off3MY = (self.longSegmentsRows(3, marker, enemy) + self.longSegmentsCols(3, marker, enemy) + self.longSegmentsDiag(3, marker, enemy) + self.longSegmentsDiagLeft(3, marker, enemy)) * 30
        off4MY = (self.longSegmentsRows(4, marker, enemy) + self.longSegmentsCols(4, marker, enemy) + self.longSegmentsDiag(4, marker, enemy) + self.longSegmentsDiagLeft(4, marker, enemy)) * 150
        off2enemy = (self.longSegmentsRows(2, enemy, marker) + self.longSegmentsCols(2, enemy, marker) + self.longSegmentsDiag(2, enemy, marker) + self.longSegmentsDiagLeft(2, enemy, marker)) * 2
        off3enemy = (self.longSegmentsRows(3, enemy, marker) + self.longSegmentsCols(3, enemy, marker) + self.longSegmentsDiag(3, enemy, marker) + self.longSegmentsDiagLeft(3, enemy, marker)) * 30
        off4enemy = (self.longSegmentsRows(4, enemy, marker) + self.longSegmentsCols(4, enemy, marker) + self.longSegmentsDiag(4, enemy, marker) + self.longSegmentsDiagLeft(4, enemy, marker)) * 200
        return off2my + off3MY + off4MY - off2enemy - off3enemy - off4enemy + isWin

    def longSegmentsRows(self, lentght, marker, enemy):
        segLen = 0
        count = 0
        currentSegmeny = []
        for y in range(15):
            currentSegmeny = []
            for cel in range(5):
                currentSegmeny.append(self.board[y][cel])
            for x in range(5, 15, 1):
                segLen = 0
                if enemy not in currentSegmeny:
                    for markerInCell in currentSegmeny:
                        if markerInCell == marker:
                            segLen += 1
                    if segLen == lentght:
                        count += 1
                currentSegmeny.pop(0)
                currentSegmeny.append(self.board[y][x])

        return count

    def longSegmentsCols(self, lentght, marker, enemy):
        segLen = 0
        count = 0
        currentSegmeny = []
        for x in range(15):
            currentSegmeny = []
            for cel in range(5):
                currentSegmeny.append(self.board[cel][x])
            for y in range(5, 15, 1):
                segLen = 0
                if enemy not in currentSegmeny:
                    for markerInCell in currentSegmeny:
                        if markerInCell == marker:
                            segLen += 1
                    if segLen == lentght:
                        count += 1
                currentSegmeny.pop(0)
                currentSegmeny.append(self.board[y][x])
        return count

    def longSegmentsDiag(self, lenght, marker, enemy):
        segLen = 0
        count = 0
        for base in range(15):
            i = 0
            segLen = 0
            currentSegmeny = []
            for cel in range(5):
                if self.isValidXY(cel, base+cel):
                    currentSegmeny.append(self.board[base+cel][cel])
            i = 5
            while self.isValidXY(i, base+i):
                segLen = 0
                if enemy not in currentSegmeny:
                    for markerInCell in currentSegmeny:
                        if markerInCell == marker:
                            segLen += 1
                    if segLen == lenght:
                        count += 1
                currentSegmeny.pop(0)
                currentSegmeny.append(self.board[base+i][i])
                i += 1
            currentSegmeny = []
            segLen = 0
            i = 0
            if base != 0:
                for cel in range(5):
                    if self.isValidXY(base+cel, cel):
                        currentSegmeny.append(self.board[cel][cel+base])
                while self.isValidXY(base+i, i):
                    segLen = 0
                    if enemy not in currentSegmeny:
                        for markerInCell in currentSegmeny:
                            if markerInCell == marker:
                                segLen += 1
                        if segLen == lenght:
                            count += 1
                    currentSegmeny.pop(0)
                    currentSegmeny.append(self.board[i][base+i])
                    i += 1
        return count

    def longSegmentsDiagLeft(self, lenght, marker, enemy):
        segLen = 0
        count = 0
        for base in range(0, 15):
            i = 0
            segLen = 0
            currentSegmeny = []
            for cel in range(5):
                if self.isValidXY(14-cel, base+cel):
                    currentSegmeny.append(self.board[14-cel][base+cel])
            i = 5
            while self.isValidXY(14-i, base+i):
                segLen = 0
                if enemy not in currentSegmeny:
                    for markerInCell in currentSegmeny:
                        if markerInCell == marker:
                            segLen += 1
                    if segLen == lenght:
                        count += 1
                currentSegmeny.pop(0)
                currentSegmeny.append(self.board[14-i][base+i])
                i += 1
            i = 0
            segLen = 0
            currentSegmeny = []
            if base != 0:
                for cel in range(5):
                    if self.isValidXY(14-base-i, cel):
                        currentSegmeny.append(self.board[cel][14-base-i])
                i = 5
                while self.isValidXY(14-base-i, i):
                    segLen = 0
                    if enemy not in currentSegmeny:
                        for markerInCell in currentSegmeny:
                            if markerInCell == marker:
                                segLen += 1
                        if segLen == lenght:
                            count += 1
                    currentSegmeny.pop(0)
                    currentSegmeny.append(self.board[i][14 - base-i])
                    i += 1
        return count

    def showBoard(self):
        print("  ", end=" ")
        for i in range(15):
            print(f"{chr(65+i):>2} ", end="")
        print("")
        print("  ",end=' ')
        for i in range(15):
            print(f"{i:>2} ", end="")
        print()
        for y in range(15):
            print(f'{y:>2}', end='')
            for x in range(15):
                print(f" |{self.board[y][x]}", end='')
            print("\n", end='')


def minimax(borad, depth, alpha, beta, maximanix):
    if(depth == 0 or borad.winningBoard):
        return (borad.evaluate(AI), (0, 0))  # move we return here doesn't matter

    playerCels = borad.getPlacesMarker(PLAYER)
    moves = set()
    for cell in playerCels:
        for eCell in borad.blankPlacesAround(cell[0], cell[1]):
            moves.add((eCell[0], eCell[1]))

    bestMove = None

    if maximanix:
        value = - math.inf
        for move in moves:
            childBorard = Board(borad.getboard())
            childBorard.placeMarker(move[0], move[1], AI)
            value = max(value, minimax(childBorard, depth - 1, alpha, beta, False)[0])
            if value > alpha:
                bestMove = move
                alpha = value
            if alpha >= beta:
                break
        return (alpha, bestMove)
    else:
        value = math.inf
        for move in moves:
            childBorard = Board(borad.getboard())
            childBorard.placeMarker(move[0], move[1], PLAYER)
            value = min(value, minimax(childBorard, depth - 1, alpha, beta, True)[0])
            if value < beta:
                bestMove = move
                beta = value
            if beta <= alpha:
                break
        return (beta, bestMove)


class Game:
    def __init__(self):
        self.curPlayer = AI
        self.board = Board.Emptyboard()

    def swichPlayer(self):
        if self.curPlayer == AI:
            self.curPlayer = PLAYER
        else:
            self.curPlayer = AI

    def playerTurn(self):
        self.board.showBoard()
        print("enter x and y(x,y in numbers)")
        i = input().split(',')
        x = int(i[0])
        y = int(i[1])
        self.board.placeMarker(x, y, PLAYER)

    def takemove(self):
        if self.curPlayer == PLAYER:
            self.playerTurn()
        else:
            self.aiMove()

    def aiMove(self):
        minMaxReturn = minimax(self.board, 3, -math.inf, math.inf, True)
        self.board.placeMarker(minMaxReturn[1][0], minMaxReturn[1][1], AI)
        print(f"ai has done move{minMaxReturn[1]}, value {minMaxReturn[0]}")

    def run(self):
        while(not self.board.winningBoard and len(self.board.getEmpty()) != 0):
            self.swichPlayer()
            self.takemove()

        if len(self.board.getEmpty()) == 0:
            print("DRAW")
        elif self.curPlayer == PLAYER:
            print("player has won")
        else:
            print("Ai has won")
        self.board.showBoard()


if __name__ == "__main__":
    g = Game()
    g.run()