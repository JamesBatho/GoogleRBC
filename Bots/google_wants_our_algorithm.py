'''
Authors: Connor Finn, Jamie Batho
Date: April 18th, 2021

Description: 

    GoogleWantsOurAlgorithm is a recon blind chess bot designed for use within the Johns Hopkins University, Applied Physics Lab's Reconnaissance blind chess comptetition. The research
project, hosted by the lab's intelligent system's center, provides the bot partial information of a chess board, decided by the sensed square. The goal is to inform decision making based 
on uncertainty. More information on the research can be found at https://rbc.jhuapl.edu/.

Note, at this point, Google definitely does not want our algorithm.

Achnoledgements:
    We leveraged the following github account to for guidance and to compete our bots against https://github.com/wrbernardoni.
'''

import chess.engine
import random
from reconchess import *
import os
import sys
import copy

class GoogleWantsOurAlgorithm(Player):


    def __init__(self):
        STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'                                                                                                  # set up stockfish
        os.environ[STOCKFISH_ENV_VAR] = '/usr/local/Cellar/stockfish/13/bin/stockfish'
        if STOCKFISH_ENV_VAR not in os.environ:
            raise KeyError(
                'GoogleWantsOurAlgorithm requires an environment variable called "{}" pointing to the Stockfish executable'.format(
                    STOCKFISH_ENV_VAR))
        self.stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        if not os.path.exists(self.stockfish_path):
            raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path, setpgrp=True)
        
        self.moved                              = False                            # used to skip handle_opponent_move_result if we are white and its the first turn
        self.board_eval_time                    = 0.1                              # how long should we evaluate boards    
        self.possible_boards                    = []                               # list to keep all posible board fens
        self.possible_scores                    = []                               # list to keep all of the scores
        self.possible_moves                     = []                               # list to keep all engine moves per board

        self.color                              = None                             # our color, WHITE == 1, BLACK == 0
        self.my_piece_captured_square           = None                             # if our oponent captures our square, set the key here, otherwise none          
        self.max_board_score                    = 1000000000                       # score for when we can immediately capture the king (breaks stockfish engine)
        self.L2                                 = [0] * 64                         # list of 64 zeros
        self.board_squares                      = list(range(64))                  # all board square keys, 0, 1, 2, 3, .... 62, 63
        
        l3 = [0] * 7    
        self.lists_of_zeros = []                                                   # create a list, of 64 lists of 7 zeros
        for i in range(len(self.board_squares)):
            self.lists_of_zeros += [copy.deepcopy(l3)]

        self.pieces             = [ 0 , chess.PAWN , chess.KNIGHT , chess.BISHOP , chess.ROOK , chess.QUEEN , chess.KING]     # all the chess pieces, (integers, zero used for empty)
        self.piece_ratings      = {0: 0 ,  chess.PAWN: 1 , chess.KNIGHT : 3 ,  chess.BISHOP : 4  ,   chess.ROOK : 5 \
         ,  chess.QUEEN : 10  ,  chess.KING : 20  }                                                                           # set  piece values: subject to change
        self.sense_squares      = list(range(9 , 15)) + list(range(17, 23)) + list(range(25, 31)) + list(range(33, 39))\
         + list(range(41, 47)) + list(range(49 ,55))                                                                          # list of all keys for internal squares. only sensing from here
            
        self.sense_score        = dict(zip(self.board_squares,self.L2));                                                      # each square gets a score (gain from sensing it)
        list_three              = [0] * 36                                                                                    # 36 internal squares
        self.total_sense_score  = dict(zip(self.sense_squares,list_three));                                                   # total sense score (square and surrounding)
        self.num_king_attackers = dict(zip(self.board_squares,copy.deepcopy(self.L2)));                                       # for each square, how many pieces can attack our king
                                  
        self.sense_map  = {}                                                                                                  # map square to its surrounding square set
        for square in self.sense_squares:
            self.sense_map[square] = self.get_surrounding_squares(square)
        
        self.piece_count        = dict(zip(self.board_squares , copy.deepcopy(self.lists_of_zeros)))                        # dict, squares are keys, list of 7 integers are values
                                                                                                                            # accross all boards, how many of each piece at each square

        self.known_squares      = [];                                                                                       # NOT CURRENTLY IN USE. Could be helpful

    def update_total_sense_score(self):
        # get total sense score for all of the internal squares

        for square in self.sense_squares:                                                                                   # iterate through all internal squares
            self.total_sense_score[square] = sum([self.sense_score[x] for x in self.sense_map[square]])                     # sum the sense score of the current square and all adjacent squares

    def get_board_score( self , board ):
        # ______ Variables: 
        #   - board - chess board object 
        # ______ Output:
        #   - score for the board, relative to the user's color. Evaluated with stockfish
        #   - our recommended move for this board

        try:
            enemy_king_square = board.king(not self.color)                                                                  # check to see if the enemy king is under attack
            if enemy_king_square:                                                                                           # make sure there is a King
                king_attackers = board.attackers(self.color, enemy_king_square)
            else:                                                                                                           
                print('ERROR get_board_score 1: theres no KING?')
                return [None , None]
            
            if king_attackers:                                                                                              # if the enemy King is under attack, set to max score
                val = self.max_board_score
                our_move = chess.Move(attacker_square, enemy_king_square)                                                   # move that takes the King

            else:                                                                                                           # otherwise use stockfish to analyze
                engine_result = self.engine.analyse(board, chess.engine.Limit(time=self.board_eval_time))
                score = engine_result['score']
                if score.is_mate():                                                                                         # mate set max score / 2
                    val = self.max_board_score / 2
                else:                                                                                                       # otherwise get the board score from white perspective
                    val  =score.white().cp

                our_move = engine_result['pv'][0].uci()
            
            if self.color:                                                                                                  # negate the score if we are black
                pass
            else:
                val = val * -1    
        
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:                                         # error handling (castling, checks ect)
            print('ERROR get_board_score 2:  -- Engine bad state at "{}"'.format(board.fen()))
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)                                          # we need to restart our engine
            return [None , None]                                                                                            # just return None for now. (could be informatino here)

        return [val , our_move ]                                                                                                # return the score

    def compute_scores(self , piece_gain = 1 , prob_gain = 10 , captured_bonus = 200 , attackers_gain = 100000 ):
        # computes sensing score for each individual square

        num_boards = len(self.possible_boards)
        for square in self.board_squares:
            attacker_score = self.num_king_attackers[square] / num_boards           # pieces attacking our king

            piece_score = 0                                                         # sum of all pieces: value x probability
            for i in range(len(self.pieces)):
                piece_score += self.piece_ratings[self.pieces[i]] * self.piece_count[square][self.pieces[i]] / num_boards

            max_num = max(self.piece_count[square])                                    
            max_prob = max_num / num_boards                                         # get the max probability of a piece at this square
            if max_prob == 1 or max_prob == 0:                                      # with perfect info, set the score to zero
                self.sense_score[square] = 0        
            else:                                                                   # otherwise, probability score: 1 / max prob 
                prob_score = 1 / max(1 - max_prob  , max_prob )
                self.sense_score[square] = piece_gain * piece_score + prob_gain * prob_score + attackers_gain * attacker_score          # Total score: 3 scores x their gains

        if self.my_piece_captured_square:                       
            self.sense_score[self.my_piece_captured_square] += captured_bonus       # add a bonus for a square where we were captured


    @staticmethod
    def get_surrounding_squares(num):
        # ______ Variables: 
        #   - num - integer key for the board position
        # ______ Output:
        #   - list including the integer keys for  num and all surrounding board squares

        return [num - 9 , num - 8 , num - 7 , num - 1 , num  ,num + 1 , num  + 7 , num + 8 , num + 9]               # one square wrap 

    @staticmethod
    def get_pseudo_legal_moves(board):
        # ______ Variables: 
        #   - board - chess board object
        # ______ Output:
        #   - list of pseudo_legal moves for this board

        moves = []
        for m in board.pseudo_legal_moves:                                          # iterate through all moves and save to list object
            moves = moves +[m]
        return moves

    def update_data_sets(self , board , our_king_square):
        # This method does two things
        #     1. Check how many pieces can attack our king
        #     2. Add to the tally for each piece on the boards squares

        king_attackers = board.attackers(not self.color, our_king_square)           # Squares where a piece is attacking our King
        if king_attackers:
            for pos in king_attackers:
                self.num_king_attackers[pos] += 1

        for i in range(len(self.board_squares)):                                    # For every square, add a tally to the piece held there. eg pawns +=1
            square = self.board_squares[i]
            piece = board.piece_at(square)                                       
            if piece != None and piece.color != self.color:                           
                self.piece_count[square][piece.piece_type] += 1              
            elif piece == None:                                                     # no pieces have key zero
                self.piece_count[square][0] += 1

    def choose_board(self , method):
        # ______ Variables: 
        #   - method - string detailing which way to select the board
        # ______ Output:
        #   - index for the selected board

        if method == 'random':
            return random.randint(len(self.possible_boards))                     # get random integer up to length of boards

        elif method == 'worst':                                                     # get the index of the board where we are worst off
            return self.possible_scores.index(min(self.possible_scores))                

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        # initializes the game for our algorithm

        self.possible_boards    = [board.fen()]                                     # set the board state as our only possible board
        self.possible_scores    = [0]                                               # set the score to be zero
        self.possible_moves     = [None]

        self.color = color                                                          # COLOR: White == 1 , Black == 0
        
        if self.color:                                                              # Useful for the viewer
            print("We are playing White")
        else:
            print("We are playing Black")

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # ______ Variables:  
        #   -  captured_my_piece – If the opponent captured one of your pieces, then True, otherwise False.
        #   -  capture_square – If a capture occurred, then the Square your piece was captured on, otherwise None.

        print('handle_opponent_move_result start ' , len(self.possible_boards))
        
        temp_board = chess.Board()                                                      # Make a temp board for us to use 

        if self.moved == False  and self.color == chess.WHITE:                          # Need to skip this method if its the first turn and we are white
            print('in here')
            self.moved = True
            return

        self.num_king_attackers = dict(zip(self.board_squares,copy.deepcopy(self.L2)));             # reset our datasets to zeros
        self.piece_count = dict(zip(self.board_squares , copy.deepcopy(self.lists_of_zeros) ))      # reser our datasets to zeros
        
        b1 = self.possible_boards[0]                                                    # Get the square our King is on
        temp_board.set_fen(b1) 
        our_king_square = temp_board.king(self.color)

        self.my_piece_captured_square = capture_square                                  # store the square we were captured on
        
        new_boards = []                                                                 # new dictionary for board generation  
        new_scores = []
        new_moves  = []

        for i in range(len(self.possible_boards)):                                      # Would be nice if we made this a function, and then used something more efficient than a loop
            b = self.possible_boards[i]
            temp_board.reset()                                  
            temp_board.set_fen(b)                                                       # set the current board

            if captured_my_piece:                                                       # If they capture, we only need the boards where a piece takes on that square                         
                moves = self.get_pseudo_legal_moves(temp_board)                     
                for move in moves:
                    if move.to_square == capture_square:
                        temp = temp_board.copy()                                        # update the board
                        temp.push(move)
                        if temp.fen() not in new_boards:                                # handle duplicates
                            new_boards = new_boards + [temp.fen()]                      # add the board to our new_boards object
                            new_scores = new_scores + [self.get_board_score(temp)[0]]
                            new_moves  = new_moves + [self.get_board_score(temp)[1]]
                            self.update_data_sets(temp ,our_king_square )               # update our datasets
            
            else:                                                                       # If they don't capture, current boards and all pseudo_legal moves are valid                                       
                temp1 = chess.Board()                       
                temp1.set_fen(b)
                temp1.turn = not temp1.turn                                             # Don't make a move but do change the board's turn
                if temp1.fen() not in new_boards:                                       # handle duplicates
                    new_boards = new_boards + [temp1.fen()]                             # add the board to our new_boards object
                    new_scores = new_scores + [self.get_board_score(temp1)[0]]
                    new_moves  = new_moves + [self.get_board_score(temp1)[1]]
                    self.update_data_sets(temp1 ,our_king_square )                      # update our datasets
                
                moves = self.get_pseudo_legal_moves(temp_board)                         # get all pseudo_legal moves
                for move in moves:                                      
                    if temp_board.piece_at(move.to_square) is None:                     # make sure we don't have a piece at the move endpoint
                        temp = chess.Board()
                        temp.set_fen(b) 
                        temp.push(move)                                                 # update the board
                        if temp.fen() not in new_boards:                                # handle duplicates
                            new_boards = new_boards + [temp.fen()]                      # add the board to our new_boards object
                            new_scores = new_scores + [self.get_board_score(temp)[0]]
                            new_moves  = new_moves + [self.get_board_score(temp)[1]]
                            self.update_data_sets(temp ,our_king_square )               # update our datasets

        self.possible_boards = new_boards                                               # update our possible_boards field to our new_boards object
        self.possible_scores = new_scores
        self.possible_moves  = new_moves

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        #  Variables
        #   * sense_actions – A list containing the valid squares to sense over.
        #   *  move_actions – A list containing the valid moves that can be returned in choose_move().
        #   *  seconds_left – The time in seconds you have left to use in the game.

        print('choose_sense start')

        self.compute_scores()                                                       # get scores for each individual square
        self.update_total_sense_score()                                             # update the net score for all internal squares  
        return max(self.total_sense_score, key=self.total_sense_score.get)          # choose the square with maximum benefit

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # ______ Variables: 
        #  -  sense_result – The result of the sense. A list of Square and an optional chess.Piece. None if no piece
        
        print('handle_sense_result start')

        temp_board = chess.Board()                                                  # Make a temp board for us to use 
        new_boards = []
        new_scores = []
        new_moves = []                                                              

        for i in range(len(self.possible_boards)):                                  # iterate through and remove all boards that conflict with our sense results
            k = self.possible_boards[i]
            temp_board.reset()          
            temp_board.set_fen(k)
            
            good = True
            for square, piece in sense_result:
                if temp_board.piece_at(square) is None and piece is not None:           # sense has a piece, our board has no piece
                    good = False
                    break
                elif temp_board.piece_at(square) is not None and piece is None:         # sense has no piece, our board has a piece
                    good = False
                    break
                elif temp_board.piece_at(square) is not None and piece is not None:     
                    if temp_board.piece_at(square).piece_type != piece.piece_type:      # sense has a different piece type than our board
                        good = False
                        break
                    if temp_board.piece_at(square).color != piece.color:                # sense has a different piece color than our board
                        good = False
                        break

            if good:
                new_boards = new_boards + [k]                                # add the board to our new_boards object
                new_scores = new_scores + [self.possible_scores[i]]
                new_moves  = new_moves  + [self.possible_moves[i]]

        self.possible_boards = new_boards                                                   # update our possible_boards field to our new_boards object
        self.possible_scores = new_scores
        self.possible_moves  = new_moves                                   

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        # ______ Variables: 
        #   -  move_actions – A list containing the valid chess.Move you can choose.
        #   -   This will give bad pawn moves!
        #   -   possibly mocves through oponent pieces? havent tested that yet
        #   -  seconds_left – The time in seconds you have left to use in the game. 

        print('choose_move')

        temp_board = chess.Board()                                                  # Make a temp board for us to use 
        new_boards = []
        new_scores = []
        new_moves = []                                           

        for i in range(len(self.possible_boards)):                                  # if our board has a move that isnt in the move list, it is not valid
            k = self.possible_boards[i]
            temp_board.reset()           
            temp_board.set_fen(k)
            board_moves = self.get_pseudo_legal_moves(temp_board)                   # get the legal moves for the board

            if not all(x in move_actions for x in board_moves):                     # check the two lists
                continue                                                            # if there is a conflict, dont keep the board
            else:
                new_boards = new_boards + [k]                                       #  add the board to our new_boards object
                new_scores = new_scores + [self.possible_scores[i]]
                new_moves  = new_moves  + [self.possible_moves[i]]
        
        self.possible_boards = new_boards                                           # update our possible_boards field to our new_boards object
        self.possible_scores = new_scores
        self.possible_moves  = new_moves


        chosen_board_idx =  self.choose_board('worst')                              # select the board (current options are 'worst' , 'random')
        chosen_board_fen =  self.possible_boards[chosen_board_idx]
        chosen_board = chess.Board()
        chosen_board.set_fen(chosen_board_fen)
        
        enemy_king_square = chosen_board.king(not self.color)                       # if we can take the enemy King, do so
        if enemy_king_square:
            enemy_king_attackers = chosen_board.attackers(self.color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()
                return chess.Move(attacker_square, enemy_king_square)
        
        try:                                                                                                # if we cant take the king, then use the engine 
            chosen_board.turn = self.color
            chosen_board.clear_stack()
            result = self.engine.play(chosen_board, chess.engine.Limit(time=self.board_eval_time))
            return result.move
        except chess.engine.EngineTerminatedError:                                                          # error handling (return none)
            print('Stockfish Engine died in Googles algorithm')
        except chess.engine.EngineError:                                                                    # error handling (return none)
            print('Stockfish Engine bad state at "{}"'.format(chosen_board.fen()))

        return None

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        # ______ Variables: 
        #   -  requested_move – The chess.Move you requested in choose_move().
        #   -  taken_move – The chess.Move that was actually applied by the game if it was a valid move, otherwise None.
        #   -  captured_opponent_piece – If taken_move resulted in a capture, then True, otherwise False.
        #   -  capture_square – If a capture occurred, then the Square that the opponent piece was taken on, otherwise None.

        print('handle_move_result' , taken_move)
        
        temp_board = chess.Board()                                                  # Make a temp board for us to use 
        new_boards = []
        new_scores = []
        new_moves = []    
 

        if requested_move == None:                                                   # if we dont make a move, no need to do anything
            print('we didnt requested a move')
            for k in self.possible_boards:
                temp_board.reset()
                temp_board.set_fen(k)                                                # modify all the turns
                temp_board.turn = not temp_board.turn
                new_boards = new_boards + [temp_board.fen()]                                
                new_scores = new_scores + [self.possible_scores[i]]
                new_moves  = new_moves  + [self.possible_moves[i]]
                self.possible_boards = new_boards                                   # update our possible_boards field to our new_boards object
                self.possible_scores = new_scores
                self.possible_moves  = new_moves
            
            return 
        

        if requested_move == taken_move: # valid move update boards 

            for k in self.possible_boards:
                temp_board.reset()
                temp_board.set_fen(k)
                temp_board.push(taken_move)                                                 # execute the move
                new_boards = new_boards + [temp_board.fen()]                                # add the board to our new_boards object
                new_scores = new_scores + [self.possible_scores[i]]
                new_moves  = new_moves  + [self.possible_moves[i]]
        else: #requested move is not taken move --> we chose an invalid move

        #how to check board to see what type of move we make? 
        ##queen, rook, bishop move is requested
            if qbr_move: 
                for k in self.possible_boards:     
                    temp_board.reset()           
                    temp_board.set_fen(k) 
                    ### now check for piece 
                    if not temp_board.piece_at(capture_square): #is this actual method? 
                        continue
                    

                    temp_board.push(taken_move)                                                 # execute the move
                    new_boards = new_boards + [temp_board.fen()]                                # add the board to our new_boards object
                    new_scores = new_scores + [self.possible_scores[i]]
                    new_moves  = new_moves  + [self.possible_moves[i]]
            if king_move:
                for k in self.possible_boards:     
                    temp_board.reset()           
                    temp_board.set_fen(k) 
                    king_square= temp_board.king(self.color)
                    if:  #kingside castle: 
                        if (temp_board.piece_at(king_square + 1) || temp_board.piece_at(king_square + 2)):
                            continue
                        ### move is now none 
                    temp_board.turn = not temp_board.turn     
                    new_boards = new_boards + [temp_board.fen()]                                # add the board to our new_boards object
                    new_scores = new_scores + [self.possible_scores[i]]
                    new_moves  = new_moves  + [self.possible_moves[i]]
                    else: #queenside 
                        if (temp_board.piece_at(king_square - 1) || temp_board.piece_at(king_square - 2) || temp_board.piece_at(king_square-3)):
                            continue
                        ### move is now none 
                    temp_board.turn = not temp_board.turn     
                    new_boards = new_boards + [temp_board.fen()]                                # add the board to our new_boards object
                    new_scores = new_scores + [self.possible_scores[i]]
                    new_moves  = new_moves  + [self.possible_moves[i]]
            if pawn_move: 
                for k in self.possible_boards:     
                    temp_board.reset()           
                    temp_board.set_fen(k)
                    ## need to get initial pawn square 
                    pawn_square = some_square 
                    if taken_move is None:  ## immediately blocked 
                        if temp_board.piece_at(pawn_square+8):
                            continue
                    elif taken_move is (pawn_square+8): ## blocked after one square
                        if temp_board.piece_at(pawn_square + 16):
                            continue
                    else: #must be attemped capture
                        # if right capture 
                        if attempted_right_capture:
                            if temp_board.piece_at(pawn_square + 9):
                                continue
                        else: #attempted left capture
                            if temp_board.piece_at(pawn_square + 7 ):
                                continue
                    temp_board.push(taken_move)
                    new_boards = new_boards + [temp_board.fen()]                                # add the board to our new_boards object
                    new_scores = new_scores + [self.possible_scores[i]]
                    new_moves  = new_moves  + [self.possible_moves[i]]
        

        self.possible_boards = new_boards                                                   # update our possible_boards field to our new_boards object
        self.possible_scores = new_scores
        self.possible_moves  = new_moves


        print('E handle_move_result:  ', len(self.possible_boards))
        print(' ')



    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        # finish the game within our algorithm

        try:
            self.engine.quit()                                                              # if the engine is already terminated then this call will throw an exception
        except chess.engine.EngineTerminatedError:
            pass





