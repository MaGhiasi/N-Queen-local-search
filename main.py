import numpy as np
import random


def initial_random_state(width):
    initial_board = np.zeros((width, width), dtype=np.int_)
    states = np.random.randint(low=0, high=width-1, size=(width))

    for i in range(width):
        initial_board[i, states[i]] = 1

    return initial_board


def fitness_function(in_board):
    queens = np.stack(np.nonzero(in_board), axis=-1)

    safe_queens = np.full((in_board.shape[0]), True)

    for i in range(len(queens)):
        for j in range(len(queens)):
            if i != j:
                if queens[i][0] == queens[j][0] or queens[i][1] == queens[j][1]:
                    safe_queens[i] = False

                if abs(queens[i][0] - queens[j][0]) == abs(queens[i][1] - queens[j][1]):
                    safe_queens[i] = False

    return safe_queens.sum()


def n_queen_neighbors_function(current_board):
    n = len(current_board)
    queens = np.stack(np.nonzero(current_board), axis=-1)

    for i in range(n):
        for j in range(1, n):
            neighbor = np.copy(current_board)
            neighbor[queens[i][0], queens[i][1]] = 0
            neighbor[queens[i][0], (queens[i][1]+j) % n] = 1
            yield neighbor


def hill_climbing_steepest_ascent(curr_board):
    board = curr_board.copy()
    cur_fit = fitness_function(board)

    while True:
        best_neighbor = board
        best_neighbor_fit = cur_fit
        for neighbor in n_queen_neighbors_function(board):
            neighbor_fit = fitness_function(neighbor)
            if neighbor_fit > best_neighbor_fit:
                best_neighbor_fit = neighbor_fit
                best_neighbor = neighbor

        if np.array_equal(board, best_neighbor):
            return board, cur_fit
        else:
            board = best_neighbor
            cur_fit = best_neighbor_fit


def hill_climbing_first_choice(curr_board):
    board = curr_board.copy()
    cur_fit = fitness_function(board)

    while True:
        prev_board = board
        for neighbor in n_queen_neighbors_function(board):
            neighbor_fit = fitness_function(neighbor)
            if neighbor_fit > cur_fit:
                board = neighbor
                cur_fit = neighbor_fit
                break

        if np.array_equal(board, prev_board):
            return board, cur_fit


def hill_climbing_stochastic(curr_board):
    board = curr_board.copy()
    cur_fit = fitness_function(board)

    while True:
        better_neighbor_fits = []
        better_neighbors = []
        for neighbor in n_queen_neighbors_function(board):
            neighbor_fit = fitness_function(neighbor)
            if neighbor_fit > cur_fit:
                better_neighbor_fits.append(neighbor_fit - cur_fit)
                better_neighbors.append(neighbor)

        if len(better_neighbors) == 0:
            return board, cur_fit

        best_neighbor = random.choices(better_neighbors, better_neighbor_fits)[0]
        board = best_neighbor
        cur_fit = fitness_function(best_neighbor)


def print_board(input_board, board_fit):
    print('\nFitness = ', board_fit)

    for row in input_board:
        for element in row:
            if element == 1:
                print('\033[92m1 \033[0m', end='')
            else:
                print('0 '.format(), end='')
        print()


if __name__ == '__main__':
    N = int(input('N: '))
    initial = initial_random_state(N)
    steepest_ascent, steepest_ascent_fit = hill_climbing_steepest_ascent(initial)
    first_choice, first_choice_fit = hill_climbing_first_choice(initial)
    stochastic, stochastic_fit = hill_climbing_stochastic(initial)

    print_board(steepest_ascent, steepest_ascent_fit)
    print_board(first_choice, first_choice_fit)
    print_board(stochastic, stochastic_fit)