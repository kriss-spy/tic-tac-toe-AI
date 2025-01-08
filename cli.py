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


class chessBoard:
    def __init__(self, size):
        self.size = size
        self.board_array = [[None] * size for i in range(size)]  # state
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
    def __init__(self, env, policy, char):
        self.env = env
        self.policy = policy
        self.char = char

    def play(self, state):
        bestAction = None
        maxReward = 0
        for i, j in [[[i, j] for j in range(self.size)] for i in range(self.size)]:
            action = [i, j, self.char]
            if self.isValid(self.env.board_array, action):
                reward = self.reward(self.env.board_array, action)
                if reward > maxReward:
                    maxReward = reward
                    bestAction = action

        self.env.board_array[bestAction[0]][bestAction[1]] = self.char

    def reward(self, state, action):  # critical
        pass

    # def value(self, state, action):

    def isValid(self, state, action):
        return True if state[action[0]][action[1]] else False


class Game:
    def __init__(self):
        self.chess_board = chessBoard(3)
        init_policy = None
        self.player1 = agentPlayer(self.chess_board, init_policy, "X")
        self.player2 = agentPlayer(self.chess_board, init_policy, "O")

    def run(self):
        self.hello()
        while True:
            print(f"turn: {self.chess_board.turn}")
            self.player1.play(self.chess_board.board_array)
            self.chess_board.print_board()
            check = self.chess_board.check()
            self.check_gameover(check)

            if self.chess_board.isFull():
                print("tie!")
                exit()

            self.player2.play(self.chess_board.board_array)
            self.chess_board.print_board()
            check = self.chess_board.check()
            self.check_gameover(check)

    def hello(self):
        print(f"tic tac toe game started!")

    def check_gameover(self, check):
        if check:
            print(f"game over! the winner is {check}")
            exit()


def main():
    mygame = Game()
    mygame.run()


# def test():

# test()
# main()
