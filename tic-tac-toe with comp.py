import numpy as np
import random
from collections import Counter

#Create an empty board and return it.

def create_board():
    return(np.zeros((3,3), dtype=int))

#Function for player movement

def player_move(board, player):
    
    
    if player == 1:
        print("The Present Position is :\n", board)
        row_pos = int(input("Enter the Row position you want to mark (between 0 and 2) :"))
        col_pos = int(input("Enter the Column position you want to mark (between 0 and 2) :"))
        posn = (row_pos,col_pos)
                    
        if posn in possibilities(board):
            place(board, 1, posn)
        else:
            print("Wrong position")
            player_move(board, player)
        print("The Present Position after your move is :\n", board)
        return(board)
    else :
        #random_place(board, player)
        optm_place(board, player)
        
        print("The Present Position after computer's move is :\n", board)
        return(board)

#Function to place computer mark optimally

def optm_place(board, player):
    chk = 0
    
    pos_place1 = []
    pos_place2 = []
    pos_place_all = []
    sim_board = board.copy()
    
    posb = possibilities(board)
    for val1 in posb:
        board1 = board.copy()
        board2 = board.copy()
        place(board1, 1, val1)
        place(board2, 2, val1)
        if row_win(board1, 1) or col_win(board1, 1) or diag_win(board1, 1):
            pos_place1.append(val1)    
        elif row_win(board2, 2) or col_win(board2, 2) or diag_win(board2, 2):
            pos_place2.append(val1)
        else:
            pos_place_all.append(val1)
            


    if len(pos_place2) > 0:
        place(board, 2, random.choice(pos_place2))
    elif len(pos_place1) > 0:
        place(board, 2, random.choice(pos_place1))
    elif (1,1) in posb:
        place(board, 2, (1,1))
    else:
        place(board, 2, random.choice(pos_place_all))
                
    return



#function to get best simulated option

def sim_choice(sim_board, avlb_opt):
    opt_dict={}
    for choice in avlb_opt:
        opt_dict.update({choice : run_sim(sim_board, choice)})
    
    return(max(opt_dict, key=opt_dict.get))


#function to run sim

def run_sim(new_sim_board, move):


    return

#Function to place a mark at an empty place and return the updated board.

def place(board, player, position):
    if board[position] == 0:
        board[position] = player
    return board

#Function to check for available options on board to mark.
def possibilities(board):
    pos = list(zip(*np.where(board == 0)))
    return(pos)

#Function to check for possibilities and place the mark randomly.
def random_place(board, player):
    pos = possibilities(board)
    ch = random.choice(pos)
    return (place(board, player, ch))

#Function to check for winning position in rows/column/diagnoal
def row_win(board, player):
    if np.any(np.all(board==player, axis=1)): # this checks if any row contains all positions equal to player.
        return True
    else:
        return False


def col_win(board, player):
    if np.any(np.all(board==player, axis=0)): # this checks if any row contains all positions equal to player.
        return True
    else:
        return False


def diag_win(board, player):
    if np.all(np.diag(board)==player) or np.all(np.diag(np.fliplr(board))==player):
        return True
    else:
        return False

#Function to evaluate the board after all turns and check for winner
def evaluate(board):
    winner = 0
    for player in [1, 2]:
        # add your code here!
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            winner = player
        pass
    if np.all(board != 0) and winner == 0:
        winner = -1
    return winner

#Main Play Game Function
def play_game():
    board = create_board()
    winner = 0
    while winner == 0:
        for player in [1, 2]:
            player_move(board,player)

            winner = evaluate(board)
            if winner!=0:
                break
    return winner

# Main Function for Playing strategic game.
def play_strategic_game():
    board = create_board()
    winner = 0
    place(board, 1, (1,1))
    random_place(board, 2)
    while winner == 0:
        for player in [1, 2]:
            random_place(board, player)
            winner = evaluate(board)
            if winner!=0:
                break
    return winner


nam = input("Whats your name Player? :")

X = [nam, "Computer", "Its a Tie"]


ind = play_game()
if ind != -1:
    print("And the Winner is : ", X[ind-1])
else:
    print("No Body Wins")
