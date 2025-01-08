"""
two agents, placing piece in turn, O or X

environment: 3x3 array as chess board

| X |   |   |
| X | O | O |
| X |   |   |

goal: win the game, have three continuous placement, horizontal, vertical, or diagonal

state: situation of 3x3 chess board, each slot is empty (None) or X or O

action: where to put next piece

reward: immediate value of a action

policy: the function, that action = function(state), here it's where-to-put-piece = policy(board-state)

value: expected long-term cumulative reward from a state or action.

two agents play in turn, and one agent don't receive immediate feedback after one action, its new state come after its opponent place its pice

maybe you are interested about afterstate
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import random


class chessBoard:
    def __init__(self, size):
        self.size = size
        self.board_array = np.array([[None] * size for i in range(size)])  # state
        self.turn = 0

    def check(self):
        for i in range(self.size):
            if self.board_array[i][0] and len(set(self.board_array[i])) == 1:
                return self.board_array[i][0]
        for i in range(self.size):
            if self.board_array[0][i] and len(
                set([row[i] for row in self.board_array]) == 1
            ):
                return self.board_array[0][i]

    def print_board(self):
        for i in range(self.size):
            for j in range(self.size):
                print(
                    self.board_array[i][j] if self.board_array[i][j] else " ", end=" "
                )

            print()

    def isFull(self):
        return all(self.board_array)


class agentPlayer:
    def __init__(self, env, char, epsilon):
        self.env = env
        # self.policy = policy
        self.char = char
        self.model = self.build_model((self.env.size, self.env.size), (1, 3))
        self.epsilon = epsilon

    def play(self, state):
        bestAction = self.actionTaken(self.env.board_array)
        self.env.board_array[bestAction[0]][bestAction[1]] = self.char

    def build_model(state_shape, action_shape):
        input_shape = (state_shape[0] + action_shape[0], state_shape[1])
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=input_shape))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse")
        return model

    def reward(self, state, action):
        model = self.model
        # Assuming state and action are numpy arrays
        input_data = np.concatenate((state.flatten().reshape(1, -1), action))
        reward = model.predict(input_data)
        return reward[0][0]

    def actionTaken(self, state):  # policy
        bestAction = None
        maxReward = 0

        self.valid_actions = self.validActions(state)
        if random.random() < self.epsilon:
            return random.choice(self.valid_actions)
        else:
            for action in self.valid_actions:
                reward = self.reward(state, action)
                if reward > maxReward:
                    maxReward = reward
                    bestAction = action
            return bestAction

    # def value(self, state, action):

    def isValid(self, state, action):
        return True if state[action[0]][action[1]] else False

    def validActions(self, state):
        valid_actions = []
        for i in range(self.size):
            for j in range(self.size):
                if not state[i][j]:
                    action = [i, j, self.char]
                    valid_actions.append(action)

        return valid_actions

    def update(self, result):  # TODO
        if result == "tie":  # change a little
            pass
        elif result == self.char:  # keep going
            pass
        else:  # change
            pass


class Game:
    def __init__(self):
        self.chess_board = chessBoard(3)
        # init_policy = None
        self.player1 = agentPlayer(self.chess_board, "X", 0.1)
        self.player2 = agentPlayer(self.chess_board, "O", 0.1)

    def run(self):
        self.hello()
        while True:
            print(f"turn: {self.chess_board.turn}")
            self.player1.play(self.chess_board.board_array)
            self.chess_board.print_board()
            check_gameover = self.chess_board.check()
            if check_gameover:
                self.gameover(check_gameover)

            if self.chess_board.isFull():
                print("tie!")
                exit()

            self.player2.play(self.chess_board.board_array)
            self.chess_board.print_board()
            check_gameover = self.chess_board.check()
            if check_gameover:
                self.gameover(check_gameover)

    def practise(self, epoch):  # train model
        for i in range(epoch):
            print("-" * 20)
            print(f"practise epoch {i}")
            while True:
                print(f"turn: {self.chess_board.turn}")
                self.player1.play(self.chess_board.board_array)
                # self.chess_board.print_board()
                check_gameover = self.chess_board.check()
                if check_gameover:
                    self.player1.update(check_gameover)
                    self.player2.update(check_gameover)
                    break
                    # self.gameover(check_gameover)

                if self.chess_board.isFull():
                    self.player1.update("tie")
                    self.player2.update("tie")
                    break
                    # print("tie!")
                    # exit()

                self.player2.play(self.chess_board.board_array)
                # self.chess_board.print_board()
                check_gameover = self.chess_board.check()
                if check_gameover:
                    self.player1.update(check_gameover)
                    self.player2.update(check_gameover)
                    break
                    # self.gameover(check_gameover)

    def hello(self):
        print(f"tic tac toe game started!")

    def check_gameover(self, check):
        if check:
            print(f"game over! the winner is {check}")
            exit()

    def gameover(self, winner):
        print(f"game over! the winner is {winner}")
        exit()

    def winner(self, check):
        if check:
            return check


def main():
    mygame = Game()
    mygame.practise()
    mygame.run()


# def test():

# test()
# main()
