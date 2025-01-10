import numpy as np
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class RLParams:
    # Training parameters
    episodes: int = 10000  # Number of training episodes
    eval_interval: int = 200  # How often to evaluate performance
    eval_games: int = 100  # Number of games for each evaluation

    # Learning parameters
    learning_rate: float = 0.1  # How quickly the agent learns (0 to 1)
    initial_epsilon: float = 0.5  # Starting exploration rate (0 to 1)
    min_epsilon: float = 0.01  # Minimum exploration rate
    epsilon_decay: float = 0.999  # Decay rate for exploration

    # Reward structure
    win_reward: float = 1.0  # Reward for winning
    draw_reward: float = 0.5  # Reward for draw
    lose_reward: float = -1.0  # Reward for losing
    living_reward: float = 0.0  # Reward for non-terminal moves


class TicTacToe:
    def __init__(self, params: RLParams):
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.q_table = {}
        self.params = params
        self.epsilon = params.initial_epsilon

    def decay_epsilon(self):
        self.epsilon = max(
            self.params.min_epsilon, self.epsilon * self.params.epsilon_decay
        )

    def get_state(self):
        return str(self.board)

    def get_empty_cells(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == " "]

    def make_move(self, pos, player):
        i, j = pos
        if self.board[i][j] == " ":
            self.board[i][j] = player
            return True
        return False

    def check_winner(self):
        # Check rows, columns and diagonals
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != " ":
                return self.board[i][0]
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != " ":
                return self.board[0][i]

        if self.board[0][0] == self.board[1][1] == self.board[2][2] != " ":
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != " ":
            return self.board[0][2]

        if not any(" " in row for row in self.board):
            return "draw"

        return None

    def get_best_move(self, player, use_epsilon=True):
        state = self.get_state()
        available_moves = self.get_empty_cells()

        if use_epsilon and random.random() < self.epsilon:
            return random.choice(available_moves)

        best_value = float("-inf")
        best_move = random.choice(available_moves)

        for move in available_moves:
            value = self.q_table.get((state, str(move)), 0.0)
            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    def update_q_value(self, state, action, next_state, reward):
        old_value = self.q_table.get((state, str(action)), 0.0)
        next_max = max(
            [
                self.q_table.get((next_state, str(move)), 0.0)
                for move in self.get_empty_cells()
            ],
            default=0.0,
        )

        new_value = old_value + self.params.learning_rate * (
            reward + next_max - old_value
        )
        self.q_table[(state, str(action))] = new_value

    def reset(self):
        self.board = [[" " for _ in range(3)] for _ in range(3)]


def evaluate_performance(game, num_games=100):
    results = {"X": 0, "O": 0, "draw": 0}

    for _ in range(num_games):
        game.reset()
        while True:
            # X's turn (random player)
            x_move = random.choice(game.get_empty_cells())
            game.make_move(x_move, "X")

            winner = game.check_winner()
            if winner:
                results[winner if winner != "draw" else "draw"] += 1
                break

            # O's turn (trained player)
            o_move = game.get_best_move(
                "O", use_epsilon=False
            )  # No exploration during evaluation
            game.make_move(o_move, "O")

            winner = game.check_winner()
            if winner:
                results[winner if winner != "draw" else "draw"] += 1
                break

    return results


def train_and_plot(params: RLParams):
    game = TicTacToe(params)

    evaluation_points = []
    o_win_rates = []
    x_win_rates = []
    draw_rates = []
    epsilon_values = []

    print("Training and evaluating...")

    for episode in range(params.episodes):
        game.reset()
        while True:
            # X's turn (random player)
            x_move = random.choice(game.get_empty_cells())
            game.make_move(x_move, "X")

            if game.check_winner():
                break

            # O's turn (learning player)
            current_state = game.get_state()
            o_move = game.get_best_move("O")
            game.make_move(o_move, "O")

            # Get reward and update Q-value
            winner = game.check_winner()
            if winner is None:
                reward = params.living_reward
            elif winner == "O":
                reward = params.win_reward
            elif winner == "draw":
                reward = params.draw_reward
            else:  # X wins
                reward = params.lose_reward

            game.update_q_value(current_state, o_move, game.get_state(), reward)

            if winner:
                break

        # Decay epsilon
        game.decay_epsilon()

        # Evaluate and record performance at intervals
        if (episode + 1) % params.eval_interval == 0:
            results = evaluate_performance(game, params.eval_games)
            total_games = sum(results.values())

            evaluation_points.append(episode + 1)
            o_win_rates.append(results["O"] / total_games * 100)
            x_win_rates.append(results["X"] / total_games * 100)
            draw_rates.append(results["draw"] / total_games * 100)
            epsilon_values.append(game.epsilon)

            print(f"Episode {episode + 1}/{params.episodes}", end="\r")

    print("\nTraining completed!")

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    # Plot win rates
    ax1.plot(evaluation_points, o_win_rates, "g-", label="O Wins (AI)")
    ax1.plot(evaluation_points, x_win_rates, "r-", label="X Wins (Random)")
    ax1.plot(evaluation_points, draw_rates, "b-", label="Draws")

    ax1.set_title("Training Progress")
    ax1.set_xlabel("Training Episodes")
    ax1.set_ylabel("Rate (%)")
    ax1.legend()
    ax1.grid(True)

    # Plot epsilon decay
    ax2.plot(evaluation_points, epsilon_values, "k-", label="Exploration Rate (ε)")
    ax2.set_xlabel("Training Episodes")
    ax2.set_ylabel("Epsilon")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    # plt.show()
    plt.savefig("dev/visualization/visualization.png")

    return game


def print_params(params: RLParams):
    print("\nTraining Parameters:")
    print(f"Episodes: {params.episodes}")
    print(f"Evaluation Interval: {params.eval_interval}")
    print(f"Games per Evaluation: {params.eval_games}")
    print("\nLearning Parameters:")
    print(f"Learning Rate (α): {params.learning_rate}")
    print(f"Initial Epsilon (ε): {params.initial_epsilon}")
    print(f"Minimum Epsilon: {params.min_epsilon}")
    print(f"Epsilon Decay: {params.epsilon_decay}")
    print("\nReward Structure:")
    print(f"Win Reward: {params.win_reward}")
    print(f"Draw Reward: {params.draw_reward}")
    print(f"Lose Reward: {params.lose_reward}")
    print(f"Living Reward: {params.living_reward}")


if __name__ == "__main__":
    # Default parameters
    params = RLParams()

    # Print parameters being used
    print_params(params)

    # Init evaluation
    print("\nInit evaluation...")
    init_results = evaluate_performance(TicTacToe(params), num_games=1000)

    # Print init results
    total_games = sum(init_results.values())
    print(f"\nResults after {total_games} test games:")
    print(
        f"X wins (Random): {init_results['X']} ({init_results['X']/total_games*100:.1f}%)"
    )
    print(
        f"O wins (AI): {init_results['O']} ({init_results['O']/total_games*100:.1f}%)"
    )
    print(
        f"Draws: {init_results['draw']} ({init_results['draw']/total_games*100:.1f}%)"
    )

    # Train and visualize
    trained_game = train_and_plot(params)

    # Final evaluation
    print("\nFinal evaluation...")
    final_results = evaluate_performance(trained_game, num_games=1000)

    # Print final results
    total_games = sum(final_results.values())
    print(f"\nResults after {total_games} test games:")
    print(
        f"X wins (Random): {final_results['X']} ({final_results['X']/total_games*100:.1f}%)"
    )
    print(
        f"O wins (AI): {final_results['O']} ({final_results['O']/total_games*100:.1f}%)"
    )
    print(
        f"Draws: {final_results['draw']} ({final_results['draw']/total_games*100:.1f}%)"
    )
