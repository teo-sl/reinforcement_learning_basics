import numpy as np
import torch

HUMAN = 1
AI = -1
class HadronGame:
    def __init__(self):
        self.board = np.zeros((5,5))
    def legal_moves(self):
        # check for those position (i,j) such that the horizontal and vertical adjacent positions are such that the number of human pieces is equal to the number of AI pieces
        ret = []
        for i in range(5):
            for j in range(5):
                if self.board[i,j] != 0:
                    continue
                count = 0
                if i>0:
                    count += self.board[i-1,j]
                if i<4:
                    count += self.board[i+1,j]
                if j>0:
                    count += self.board[i,j-1]
                if j<4:
                    count += self.board[i,j+1]
                if count == 0:
                    ret.append(i*5+j)
        return ret

    def make_move(self, move, player):
        x,y = move//5, move%5
        self.board[x,y] = player
    def unroll_move(self, move):
        self.board[move] = 0
    def is_over(self):
        return len(self.legal_moves()) == 0
    
    # make a pretty print of the board, make the edged of the board
    def print_board(self):
        print("  0 1 2 3 4")
        for i in range(5):
            print(i, end=" ")
            for j in range(5):
                if self.board[i,j] == HUMAN:
                    print("X", end=" ")
                elif self.board[i,j] == AI:
                    print("O", end=" ")
                else:
                    print("-", end=" ")
            print()



class HadronEnv():
    def __init__(self,opponent):
        self.game = HadronGame()
        self.opponend = None
    def reset(self):
        self.game = HadronGame()
        state = self.preprocess_state()
        return state.reshape(25*3)
    def get_action_space(self):
        return 25
    def get_state_space(self):
        return 25*3
    def preprocess_state(self):
        state = self.game.board
        mat_1 = np.where( state== 1, 1, 0)
        mat_minus_1 = np.where(state == -1, 1, 0)
        mat_0 = np.where(state == 0, 1, 0)
        result_array = np.array([mat_1, mat_minus_1, mat_0])
        return result_array.reshape(25*3)
    
    def step(self, action):

        self.game.make_move(action, AI)

        state = self.preprocess_state()
        if self.game.is_over():
            return state, 1, True, {}
        
        r_action = self.opponent.get_action(self.preproccess_state())

        self.game.make_move(r_action, HUMAN)

        state = self.preprocess_state()

        if self.game.is_over():
            return state, -1, True, {}
        
        return state, 0, False, {}

        
    def render(self):
        print(self.game.board)


class Opponent:
    def __init__(self,net,game):
        self.net = net
        self.game = game
    def get_action(self,state):
        scores = self.net(state)        
        
        selected_tensor = torch.index_select(scores, 1, torch.tensor(mask))
        mask = self.game.legal_moves()
        argmax = torch.argmax(selected_tensor)
        argmax_global = mask[argmax]
        return argmax_global
