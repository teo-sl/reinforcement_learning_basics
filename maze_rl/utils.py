import numpy as np

def stringfy(board):
    string = ''
    for i in range(3):
        for j in range(3):
            string += str(board[i, j])
    return string

def check_victory(board):
    # check rows
    for i in range(3):
        if np.all(board[i, :] == 1):
            return 1
        elif np.all(board[i, :] == 2):
            return 2
    # check columns
    for i in range(3):
        if np.all(board[:, i] == 1):
            return 1
        elif np.all(board[:, i] == 2):
            return 2
    # check diagonals
    if np.all(board.diagonal() == 1):
        return 1
    elif np.all(board.diagonal() == 2):
        return 2
    if np.all(np.fliplr(board).diagonal() == 1):
        return 1
    elif np.all(np.fliplr(board).diagonal() == 2):
        return 2
    return 0

def get_legal_actions(board):
        couples = np.argwhere(board == 0)
        return np.array([t[0]*3+t[1] for t in couples])



def get_all_states_tic_tac_toe():# define all 9 tuples of 0,1,2, each tuple is rapresented as a string, like 120001201
    all_states = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            for o in range(3):
                                for p in range(3):
                                    for q in range(3):
                                        all_states.append(str(i)+str(j)+str(k)+str(l)+str(m)+str(n)+str(o)+str(p)+str(q))
    return np.array(all_states)


def get_max_for_specific_indexes(row,idxes):
    max_val = -np.inf
    action = None
    for x in idxes:
        if row[x] > max_val:
            max_val = row[x]
            action = x
    return action

def convert_compact_to_matrix(config):
    board = np.zeros((3,3),dtype=int)
    for i in range(3):
        for j in range(3):
            board[i,j] = config[i*3+j]
    return board
