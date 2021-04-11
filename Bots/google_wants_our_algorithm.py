
# idea is to start with the trout bot, and work to improve it, 

import chess.engine
import random
from reconchess import *
import os
import sys
import copy


class HelperFunctions():

    def __init__(self):
        pass


class GoogleWantsOurAlgorithm(Player):
    """
    TroutBot uses the Stockfish chess engine to choose moves. In order to run TroutBot you'll need to download
    Stockfish from https://stockfishchess.org/download/ and create an environment variable called STOCKFISH_EXECUTABLE
    that is the path to the downloaded Stockfish executable.
    """

    def __init__(self):
        self.possible_boards = {} # keys will be the fen for the board, values will be score or prob (1 for now)

        self.color  = None
        self.my_piece_captured_square = None
        self.moved = False
        # point scores for pieces: I think bishop is worth more  than a knight in this game. ?
        # zero is for no piece there
        self.piece_ratings ={0: 0 ,  chess.PAWN: 1 , chess.KNIGHT : 3 ,  chess.BISHOP : 4  ,   chess.ROOK : 5  ,  chess.QUEEN : 10  ,  chess.KING : 20  }


        # keep track of possible moves to a point
        L1 = list(range(64))
        L2 = [0] * 64

        self.board_squares = L1
        self.opponent_moves_to_square = dict(zip(L1,L2))

        # if we want to limit the boards we keep, we will use this. for now this is not in use
        self.max_boards = sys.maxsize # 750 (breaks)
        
        # internal squares. we will not sense outside of this region
        self.sense_squares = list(range(9 , 15)) + list(range(17, 23)) + list(range(25, 31)) + list(range(33, 39)) + list(range(41, 47)) + list(range(49 ,55))
        list_three = [0] * 36

        # keep track of squares we have perfect knowlege of?
        self.known_squares = [];

        # how about all the squares that could attack the king
        self.num_king_attackers = dict(zip(L1,L2));

        # keep track of the  'benefit' of sensing a certain square. This will inform the sense algorithm
        self.sense_score = dict(zip(L1,L2));

        # score recieved by sensing the square and all surrounding squares
        self.total_sense_score = dict(zip(self.sense_squares,list_three));

        # get the sense mapping
        self.sense_map  = {}
        for square in self.sense_squares:
            self.sense_map[square] = self.get_surrounding_squares(square)

        # would like to know the number of each peice on a square
        # these chess.PAWN , chess.KNIGHT ... are just integers 1-6
        self.pieces = [ 0 , chess.PAWN , chess.KNIGHT , chess.BISHOP , chess.ROOK , chess.QUEEN , chess.KING]
        l3 = [0] * 7
        self.lists_of_zeros = []
        for i in range(len(L1)):
            self.lists_of_zeros += [copy.deepcopy(l3)]

        self.piece_count = dict(zip(self.board_squares , copy.deepcopy(self.lists_of_zeros)))

        self.piece_probabilities = dict(zip(self.board_squares , copy.deepcopy(self.lists_of_zeros)))
        ####  STOCKFISH #### 
        STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
        os.environ[STOCKFISH_ENV_VAR] = '/usr/local/Cellar/stockfish/13/bin/stockfish'
        # make sure stockfish environment variable exists
        if STOCKFISH_ENV_VAR not in os.environ:
            raise KeyError(
                'TroutBot requires an environment variable called "{}" pointing to the Stockfish executable'.format(
                    STOCKFISH_ENV_VAR))

        # make sure there is actually a file
        stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        if not os.path.exists(stockfish_path):
            raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))

        # initialize the stockfish engine
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True)
        #### STOCKFISH ####
    

    def update_total_sense_score(self):
        # this function builds the sense score recieved by looking at the square in quwstion and all surrounding squares
        for square in self.sense_squares:
            self.total_sense_score[square] = sum([self.sense_score[x] for x in self.sense_map[square]])

    def pick_board_to_play(boards):

#    board_value = k1 * variable_one + k2 * variable_two 

        pass

    def compute_scores(self , piece_gain = 1 , prob_gain = 10 , captured_bonus = 50):
        ## TODO ## 
        # add a score for the squares that could attack your peices
        # add a score for stockfish's predicted move


        # we are guarenteed to pick somewhere internal: so let's get the surrounding squares
        num_boards = len(self.possible_boards)
        for square in self.board_squares:
            # get the piece score
            piece_score = 0
            for i in range(len(self.pieces)):
                # prob times the piece score
                piece_score += self.piece_ratings[self.pieces[i]] * self.piece_count[square][self.pieces[i]] / num_boards

            # get the probability score

            max_num = max(self.piece_count[square])
            max_prob = max_num / num_boards
            
            if max_prob == 1 or max_prob == 0:
                self.sense_score[square] = 0
            else:
                prob_score = 1 / max(1 - max_prob  , max_prob )
                
                # take the gains into consideration
                self.sense_score[square] = piece_gain * piece_score + prob_gain * prob_score

                print(piece_gain * piece_score , prob_gain * prob_score)
        # add booster for if piece was captured
        if self.my_piece_captured_square:
            self.sense_score[self.my_piece_captured_square] += captured_bonus



    @staticmethod
    def get_surrounding_squares(num):
        return [num - 9 , num - 8 , num - 7 , num - 1 , num  ,num + 1 , num  + 7 , num + 8 , num + 9]

    @staticmethod
    def get_pseudo_legal_moves(board):
        '''
         get a list of all legal moves for the board. This can be called for both opponent and yourself
        '''
        moves = []
        for m in board.pseudo_legal_moves:
            moves = moves +[m]

        return moves


    @staticmethod
    def get_legal_moves(board):
        '''
         get a list of all legal moves for the board. This can be called for both opponent and yourself
        '''
        moves = []
        for m in board.legal_moves:
            moves = moves +[m]

        return moves


 

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.possible_boards[board.fen()] = 1   # store the first board with a probablilty of one # fen is string of board
        self.color = color   # White  = True, Black = False
        if self.color:
            print("We are playing White")
        else:
            print("We are playing Black")

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # ______ Variables:  
        #   -  captured_my_piece – If the opponent captured one of your pieces, then True, otherwise False.
        #   -  capture_square – If a capture occurred, then the Square your piece was captured on, otherwise None.
        # _______ Logic
        #   -  Capture
        #       - Current boards are no longer vaild
        #       - Create all boards from our board set which result in the capture on that square
        #   -  No capture:
        #       - current boards are valid, and all all moves from each board are valid
        #   -  TODO Get Rid of loops!
        #   -  TODO Make the score (currently 1) reflect some information about the board?

        # make every possible board

        # print('S handle_opponent_move_result:  ', len(self.possible_boards))
        # print(' ')
        # bad implimentation by authors
        if self.moved == False  and self.color == chess.WHITE:
            print('in here')
            self.moved = True
            return




        # if the opponent captured our piece, remove it from our board.
        self.my_piece_captured_square = capture_square
        new_boards = {}    # if he/she/they takes, all old boards are no longer valid

        ########## TODO #############
        # get rid of all these loops!
        ########   TODO #############

        temp_board = chess.Board()
        for b in self.possible_boards:
            # loop through the possible boards. 
            
            temp_board.reset()  
            temp_board.set_fen(b)                                   # create the board object

            if captured_my_piece:                                   # If he captured
                moves = self.get_pseudo_legal_moves(temp_board)

                for move in moves:
                    if move.to_square == capture_square:
                        # for all possible moves of this board, set a new board
                        temp = temp_board.copy()
                        temp.push(move)
                        new_boards[temp.fen()] = 1       
            else:                                                   # no capture, all boards still valid
                # keep the orig board, but change the turn first
                temp1 = chess.Board()
                temp1.set_fen(b)
                temp1.turn = not temp1.turn
                new_boards[temp1.fen()] = 1 
                
                # now evaluate the possible moces
                moves = self.get_pseudo_legal_moves(temp_board) 
                for move in moves:
                    if temp_board.piece_at(move.to_square) is None:   # make sure there is no piece there
                        # for all possible moves of this board, set a new board
                        #temp = temp_board.copy()
                        temp = chess.Board()
                        temp.set_fen(b) 

                        temp.push(move)
                        new_boards[temp.fen()] = 1     

        # set the new boards
        self.possible_boards = new_boards

        # print('E handle_opponent_move_result:  ', len(self.possible_boards))
        # print(' ')

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        #  Variables
        #   * sense_actions – A list containing the valid squares to sense over.
        #   *  move_actions – A list containing the valid moves that can be returned in choose_move().
        #   *  seconds_left – The time in seconds you have left to use in the game.

        # Logic
        #  * right now only restricting the sense to 
        #  *   1. the square that was captured
        #  *   2. a square we are thinking about captureing
        #  *   3. then a random square within the inside border



        # print('S choose_sense:  ', len(self.possible_boards))
        # print(' ')

        # part one: update the score of each location
        # idea here:
        #   * add up the predicted pieces for each board at every square
        # improvements: we can keep track of how many pieces the oponent has
        #   to sace time, once that nummber is reached we stop
        #   we also can know the perfect info squares

        # reset the board count
        temp_board = chess.Board()
        self.piece_count = dict(zip(self.board_squares , copy.deepcopy(self.lists_of_zeros) ))

        # grab the first board
        b1 = list(self.possible_boards.keys())[0]
        temp_board.set_fen(b) 

        # where is our king
        our_king_square = chosen_board.king(self.color)


        for b in self.possible_boards:
            temp_board.reset()  
            temp_board.set_fen(b)  

            # update the enemy attackers on our king
            # king_attackers = temp_board.attackers(not self.color, our_king_square)
            # if king_attackers:
            #     for pos in king_attackers:
            #         self.num_king_attackers
                              
            # check each square  ## TODO THIS IS BROKEN ###
            for i in range(len(self.board_squares)):
                square = self.board_squares[i]
                piece = temp_board.piece_at(square)

                                         # grab a piece
                if piece != None and piece.color != self.color:
                    self.piece_count[square][piece.piece_type] += 1               # add one to our count

                elif piece == None:
                    self.piece_count[square][0] += 1




        # update the scores for each location
        self.compute_scores()

        # update the total sense score
        self.update_total_sense_score()
        

        
        
        return max(self.total_sense_score, key=self.total_sense_score.get)






    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # ______ Variables: 
        #  -  sense_result – The result of the sense. A list of Square and an optional chess.Piece. None if no piece
        # _______ Logic
        #   -  For every board in our possible_boards field, check the sense result for conflicts
        #   -  If a conflict exists, pop the entry from the dictionary. 
        #   - TODO do not check entries where we have perfect information (saves computation)

        # print('S handle_sense_result:  ', len(self.possible_boards))
        # print(' ')
        # print(sense_result)
        # we have sensed, now let's throw away any boards which don't match our sense
        # this is done by checking every sceneario and 
        temp_board = chess.Board()
        newBoards = {}
        for k in self.possible_boards:
            temp_board.reset()           # faster than creting a new board
            temp_board.set_fen(k)
            
            # all the scenarios where the board is not valid
            
        ########## TODO #############
        # we don't need to do this for every sensed item
        # We could have perfect info on some squares:
        #   * our piece is there
        #   * we know the opponent piece is there
        #   * we know the board is empty
        #   * keep track of the squares we know are perfect and don't evaluate them below
        ########   TODO #############

            good = True
            for square, piece in sense_result:

                # board is none, sense is not none
                if temp_board.piece_at(square) is None and piece is not None:  
                    good = False
                    break

                # board is not none, sense is none
                elif temp_board.piece_at(square) is not None and piece is None:
                    good = False
                    break

                # board is not none, sense is not none
                elif temp_board.piece_at(square) is not None and piece is not None:

                    # Wrong piece
                    if temp_board.piece_at(square).piece_type != piece.piece_type:
                        good = False
                        break

                    # Wrong color
                    if temp_board.piece_at(square).color != piece.color:
                        good = False
                        break
            
            if good:
                newBoards[k] = 1

        self.possible_boards = newBoards
        # print('S handle_sense_result:  ', len(self.possible_boards))
        # print(' ')
    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        # ______ Variables: 
        #   -  move_actions – A list containing the valid chess.Move you can choose.
        #   -   This will give bad pawn moves!
        #   -   possibly mocves through oponent pieces? havent tested that yet
        #   -  seconds_left – The time in seconds you have left to use in the game. 
        # ______ Logic: 
        #   - This is good information: we can further restrict our boards by removing ones where the valid moves don't match up
        #   -  For now, randomly select a board and use stockfish to pick the best move.

        ###### TODO ########
        # better algorithm here. currently only considering one randomly selected board
        ###### TODO ########

        # remove boards that have moves not in the legal move list
        # print('S choose_move:  ', len(self.possible_boards))
        # print(' ')
        temp_board = chess.Board()
        newBoards = {}
        for k in self.possible_boards:
            temp_board.reset()           
            temp_board.set_fen(k)
            if temp_board.turn != self.color:
                print('THE BOARD IS ON THE WRONG TURN! THIS IS BAD')

            board_moves = self.get_pseudo_legal_moves(temp_board)      # get the legal moves for the board
            l_board_moves = self.get_pseudo_legal_moves(temp_board)      # get the legal moves for the board
            if board_moves != l_board_moves:
                print('legal is not pseudo_legal_moves  ')
            if not all(x in move_actions for x in board_moves):        # are any moves from our board that are not in the list of possible moves
                continue                           # if so, get rid of that board
            else:
                newBoards[k] = 1
        self.possible_boards = newBoards    
        print('After filtering we now have possible boards:  ', len(self.possible_boards))

        


        chosen_board_idx = random.choice(list(self.possible_boards.keys()))

        chosen_board = chess.Board()
        chosen_board.set_fen(chosen_board_idx)
        enemy_king_square = chosen_board.king(not self.color)
        if enemy_king_square:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = chosen_board.attackers(self.color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()
                return chess.Move(attacker_square, enemy_king_square)

        # otherwise, try to move with the stockfish chess engine
        try:
            chosen_board.turn = self.color
            chosen_board.clear_stack()
            result = self.engine.play(chosen_board, chess.engine.Limit(time=0.5))
            return result.move
        except chess.engine.EngineTerminatedError:
            print('Stockfish Engine died in Googles algorithm')
        except chess.engine.EngineError:
            print('Stockfish Engine bad state at "{}"'.format(chosen_board.fen()))

        # if all else fails, pass
        # print('We return none here')
        return None

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        # ______ Variables: 
        #   -  requested_move – The chess.Move you requested in choose_move().
        #   -  taken_move – The chess.Move that was actually applied by the game if it was a valid move, otherwise None.
        #   -  captured_opponent_piece – If taken_move resulted in a capture, then True, otherwise False.
        #   -  capture_square – If a capture occurred, then the Square that the opponent piece was taken on, otherwise None.
        # _______ Logic
        #   -  In our move logic we have restricted everything to legal moves
        #   -  Taken != requested? This means we captured too early
        #   -  Captured  
        

        # not the same
            # pawmn takes but cant - all boards where you could take no longer valie
            # pawn forward  but cant - any board that doesnt have oponenet piece there is no god
            # pawn forward two - only gets one
            # castling - cant 
            # rook , bishop , queen  go somewhere but stop short. any board with no piece where the rbq ends, not valie



        # print('S handle_move_result:  ', len(self.possible_boards))
        # print(' ')


        if captured_opponent_piece:
            print('captured a piece')

        else:
            print('did not capture')

        if requested_move == None:
            print('we didnt requested a move')
            return 


        else:
            newBoards = {}
            temp_board = chess.Board()
            
            # if the move happens, then let's make the move
            if taken_move is not None:
                # if we made a move
                for k in self.possible_boards:     
                    temp_board.reset()           
                    temp_board.set_fen(k)

                    # if its legal, then make it
                    if taken_move in temp_board.pseudo_legal_moves:

                        # if we take the king, and the game isnt over, we don't want this board
                        if (captured_opponent_piece == True) and (temp_board.piece_at(capture_square) is not None) and (temp_board.piece_at(capture_square).piece_type != chess.KING):
                            temp_board.push(taken_move)
                            newBoards[temp_board.fen()] = 1
                            

                        # we dont capture, and there isn't a piece on our board, keep the board
                        elif (captured_opponent_piece == False) and (temp_board.piece_at(taken_move.to_square) is None):
                            temp_board.push(taken_move)
                            newBoards[temp_board.fen()] = 1


                        # note the two cases we do not use a board
                        #   *  captured_opponent_pierce is true, and there is no piece there
                        #   *  captured_opponent_piece is false, and there is a piece  there

            else:
                # if we tried to make a move but couldnt - don't update any boards 
                newBoards = {}
                temp_board = chess.Board()

                for k in self.possible_boards:     
                    temp_board.reset()           
                    temp_board.set_fen(k)
                    temp_board.turn = not temp_board.turn
                    newBoards[temp_board.fen()] = 1
               
                
                ### TODO ####
                # there is information here
                #  big one is pawns:
                #  if we try to take diagonally and can't.
                # also something about trying to move two forward and only getting one... leave that for later, thats in an eleif above this level
                ### TODO ### 


                # if it was a pawn
                # for k in self.possible_boards:     
                #     temp_board.reset()           
                #     temp_board.set_fen(k)

                #     if (temp_board.piece_at(requested_move.from_square) is not None) and (temp_board.piece_at(requested_move.from_square).piece_type == chess.PAWN):
                #         # if it moved diagonally (modulus 8 is not the same)
                #         if requested_move.to_square % 8 != requested_move.from_square % 8:
                #             continue
                #         else: 
                #             newBoards[temp_board.fen()] = 1
                #  * what information can we get from this scenario
                #  * e.g. a pawn that couldn't take but tried

            self.possible_boards = newBoards

        print('E handle_move_result:  ', len(self.possible_boards))
        print(' ')

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        try:
            # if the engine is already terminated then this call will throw an exception
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass