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

# refer to doc/how-to-implement-reward

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import random

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging


class chessBoard:
    def __init__(self, size):
        self.size = size
        self.board_array = np.array([[None] * size for i in range(size)])  # state
        self.turn = 0

    def check(self):
        # horizontal
        for i in range(self.size):
            if self.board_array[i][0] and len(set(self.board_array[i])) == 1:
                return self.board_array[i][0]
        # vertical
        for i in range(self.size):
            if (
                self.board_array[0][i]
                and len(set([row[i] for row in self.board_array])) == 1
            ):
                return self.board_array[0][i]
        # diagonal
        tmp1 = set()
        tmp2 = set()
        for i in range(self.size):
            tmp1.add(self.board_array[i][i])
            tmp2.add(self.board_array[i][self.size - 1 - i])
        if self.board_array[0][0] and len(tmp1) == 1:
            return self.board_array[0][0]
        if self.board_array[0][self.size - 1] and len(tmp2) == 1:
            return self.board_array[0][self.size - 1]

    def print_board(self):
        for i in range(self.size):
            line = ""
            for j in range(self.size):
                line += self.board_array[i][j] if self.board_array[i][j] else " "
                line += "|"

            print(line[:-1])

    def isFull(self):
        return self.board_array.all()


class agentPlayer:
    def __init__(self, env, char, epsilon):
        self.env = env
        self.state = self.env.board_array
        # self.policy = policy
        self.char = char
        self.model = self.build_model(self.state.shape, (1, 3))
        self.epsilon = epsilon

    def play(self, state):
        bestAction = self.actionTaken(self.state)
        self.env.board_array[int(bestAction[0])][int(bestAction[1])] = self.char

    def random_play(self, state):
        self.valid_actions = self.validActions(state)
        action = random.choice(self.valid_actions)
        self.env.board_array[int(action[0])][int(action[1])] = self.char

    def build_model(self, state_shape, action_shape):
        input_shape = (state_shape[0] + action_shape[0], state_shape[1])
        input_shape = (input_shape[0] * input_shape[1], 1)
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=input_shape))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse")
        return model

    def reward(self, state, action):  # is this actual RL?
        model = self.model
        # state and action are numpy arrays
        action[:2] = action[:2].astype(int)
        input_data = np.concatenate((state.flatten(), action)).reshape(1, -1)
        input_data = np.where(input_data == None, 0, input_data)
        input_data = np.where(input_data == "X", 1, input_data)
        input_data = np.where(input_data == "O", -1, input_data)
        input_data[0][-3] = int(input_data[0][-3])
        input_data[0][-2] = int(input_data[0][-2])
        input_data = input_data.astype(np.float32)  # tf uses float32!
        reward = model.predict(input_data, verbose=0)
        return reward[0][0]

    def actionTaken(self, state):  # policy
        bestAction = None
        maxReward = float("-inf")

        self.valid_actions = self.validActions(state)
        if random.random() < self.epsilon:
            bestAction = random.choice(self.valid_actions)
            return bestAction
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
        for i in range(self.env.size):
            for j in range(self.env.size):
                if not state[i][j]:
                    action = np.array([i, j, self.char])
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
        self.player1 = agentPlayer(self.chess_board, "X", 0.5)
        self.player2 = agentPlayer(self.chess_board, "O", 0.5)
        self.Xwins = 0
        self.Owins = 0
        self.ties = 0

    def init_state_agents(self):
        self.chess_board = chessBoard(3)
        # init_policy = None
        self.player1 = agentPlayer(self.chess_board, "X", 0.5)
        self.player2 = agentPlayer(self.chess_board, "O", 0.5)

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
            print("-" * 10)
            self.player2.play(self.chess_board.board_array)
            self.chess_board.print_board()
            check_gameover = self.chess_board.check()
            if check_gameover:
                self.gameover(check_gameover)

            self.chess_board.turn += 1
            print()

    def random_run(self):
        self.hello()
        while True:
            print(f"turn: {self.chess_board.turn}")
            self.player1.random_play(self.chess_board.board_array)
            self.chess_board.print_board()
            check_gameover = self.chess_board.check()
            if check_gameover:
                self.gameover(check_gameover)

            if self.chess_board.isFull():
                print("tie!")
                exit()
            print("-" * 10)
            self.player2.random_play(self.chess_board.board_array)
            self.chess_board.print_board()
            check_gameover = self.chess_board.check()
            if check_gameover:
                self.gameover(check_gameover)

            self.chess_board.turn += 1
            print()

    def practise(self, epoch):  # train model # TODO
        for i in range(epoch):
            print("=" * 20)
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
                print("-" * 10)
                self.player2.play(self.chess_board.board_array)
                # self.chess_board.print_board()
                check_gameover = self.chess_board.check()
                if check_gameover:
                    self.player1.update(check_gameover)
                    self.player2.update(check_gameover)
                    break
                    # self.gameover(check_gameover)

                self.chess_board.turn += 1
                print()

    def compare(self, epoch):
        print("compare start!")
        for i in range(epoch):
            print("=" * 20)
            print(f"epoch {i}")
            while True:
                # print(f"turn: {self.chess_board.turn}")
                self.player1.play(self.chess_board.board_array)
                # self.chess_board.print_board()
                check_gameover = self.chess_board.check()
                if check_gameover:
                    # self.player1.update(check_gameover)
                    # self.player2.update(check_gameover)
                    print(f"the winner is {check_gameover}")
                    self.Xwins += 1
                    break
                    # self.gameover(check_gameover)

                if self.chess_board.isFull():
                    # self.player1.update("tie")
                    # self.player2.update("tie")
                    print("tie")
                    self.ties += 1
                    break
                    # print("tie!")
                    # exit()
                # print("-" * 10)
                self.player2.play(self.chess_board.board_array)
                # self.chess_board.print_board()
                check_gameover = self.chess_board.check()
                if check_gameover:
                    # self.player1.update(check_gameover)
                    # self.player2.update(check_gameover)
                    print(f"the winner is {check_gameover}")
                    self.Owins += 1
                    break
                    # self.gameover(check_gameover)

                self.chess_board.turn += 1
                # print()
            self.init_state_agents()
        self.print_wins_and_ties()

    def print_wins_and_ties(self):
        total = self.Xwins + self.Owins + self.ties
        print(f"X wins: {self.Xwins} {self.Xwins/total}")
        print(f"O wins: {self.Owins} {self.Owins/total}")
        print(f"ties: {self.ties} {self.ties/total}")

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


def test_without_RL():
    mygame = Game()
    mygame.random_run()


def test_without_train():
    mygame = Game()
    mygame.run()


def compare():
    mygame = Game()
    mygame.compare(10)


compare()
# test_without_train()
# test_without_RL()
# main()
