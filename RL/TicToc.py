# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 18:45:13 2019

@author: cshao
"""
import numpy as np

class Environment:
    def __init__(self, sizeX, sizeY, winLength):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.winLength = winLength
        self.board = np.zeros((self.sizeX, self.sizeY), dtype=int)
        
#        self.board = np.array([[1,1,0],[0,1,0],[0,0,1]])
        self.x = -1 # represents an x on the board, player 1
        self.o = 1 # represents an o on the board, player 2
        self.winner = None
        self.ended = False
        self.num_states = 3**(self.sizeX * self.sizeY)
    
    def is_empty(self, i, j):
        return self.board[i,j] == 0
    
    def is_draw(self):
        return self.ended and self.winner is None

    def reward(self, sym):
        # no reward until game is over
        if not self.game_over():
            return 0
        if self.is_draw():
            return 0.5
        return 1 if self.winner == sym else 0
      
    def get_state(self):
        sum_v=0
        for i in range(sizeX):
            for j in range(sizeY):
                sum_v= sum_v*3 + (self.board[i,j]+1)
        return sum_v
        
    def get_max_state(self):
        self.board=np.ones((self.sizeX, self.sizeY), dtype=int)
        max_state = self.get_state()
        self.board = np.zeros((self.sizeX, self.sizeY), dtype= int)
        return max_state
    
    def game_over(self):
        if self.ended:
            return True
        for i in range(self.sizeX):
            for j in range(self.sizeY):
                p = self.board[i,j]
                bFinished = False
                
                if (i+self.winLength<=self.sizeX):
                    bFinished = True
                    for k in range( self.winLength):
                        if(p* self.board[i+k,j]!=1):
                            bFinished = False
                            break
                        
                if (not bFinished) and (j+self.winLength<=self.sizeY):
                    bFinished = True
                    for k in range(self.winLength):
                        if(p* self.board[i,j+k]!=1):
                            bFinished = False
                            break
                
                if (not bFinished) and (i+self.winLength<=self.sizeX) and (j+self.winLength<=self.sizeY):
                    bFinished = True
                    for k in range(self.winLength):
                        if(p* self.board[i+k,j+k]!=1):
                            bFinished = False
                            break
                
                if bFinished:
                    self.winner= p
                    self.ended = True
        
        if np.all((self.board == 0) == False):
            self.winner= None
            self.ended = True
        
        return self.ended
    
    def display_board(self):
        print("")
        for i in range(self.sizeX):
            for j in range(self.sizeY):
                if(self.board[i,j]==self.o):
                    print('O',end="")
                elif(self.board[i,j]==self.x):
                    print('X',end="")
                else:
                    print(' ',end="")
                print('|',end="") 
            print("")
            for j in range(self.sizeY):
                print('--',end="")
            print("")
        print("")        
            
        
class Agent:
    def __init__(self, sym, eps=0.1, alpha=0.5):
        self.sym = sym
        self.eps = eps # probability of choosing random action instead of greedy
        self.alpha = alpha # learning rate
        self.verbose = False
        self.state_history = []
        
    def set_verbose(self, v):
        # if true, will print values for each position on the board
        self.verbose = v
    
    def setV(self, V):
        self.V = V
        
    def setEps(self, eps):
        self.eps = eps
    
    def reset_history(self):
        self.state_history = []

    def update_state_history(self, s):
        # cannot put this in take_action, because take_action only happens
        # once every other iteration for each player
        # state history needs to be updated every iteration
        # s = env.get_state() # don't want to do this twice so pass it in
        self.state_history.append(s)
        
    def display_V_board(self, env, pos2value):
        print("")
        for i in range(env.sizeX):
            for j in range(env.sizeY):
                if(env.board[i,j]==env.o):
                    print('  o ',end="")
                elif(env.board[i,j]==env.x):
                    print('  x ',end="")
                else:
                    print('%.2f'% pos2value[(i,j)] ,end="")
                print(' |',end="") 
            print("")
            for j in range(env.sizeY):
                print('------',end="")
            print("")
        print("")
            
    def take_action(self, env):
        possible_moves =[]
        for i in range(env.sizeX):
            for j in range(env.sizeY):
                if env.is_empty(i,j):
                    possible_moves.append((i, j))
        
        r=np.random.rand()
        if r < self.eps:
            # take a random action
            if self.verbose:
                print("Taking a random action")
            idx=np.random.choice(len(possible_moves))
            nx_move = possible_moves[idx]
        else:
            pos2value = {} # for debugging
            bestV =-1
            for mv in possible_moves:
                env.board[mv[0],mv[1]]=self.sym
                s=env.get_state()
                env.board[mv[0],mv[1]]=0
                pos2value[mv]=self.V[s]
                if bestV< self.V[s]:
                    bestV = self.V[s]
                    nx_move = mv
            if self.verbose:
                self.display_V_board(env,pos2value)
            
        # make the move
        env.board[nx_move[0], nx_move[1]] = self.sym

    def update(self, env):
        # we want to BACKTRACK over the states, so that:
        # V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
        # where V(next_state) = reward if it's the most current state
        #
        # NOTE: we ONLY do this at the end of an episode
        # not so for all the algorithms we will study
        reward = env.reward(self.sym)
        target = reward
        for prev in reversed(self.state_history):
           self.V[prev] = self.V[prev] + self.alpha*(target - self.V[prev])
           target =self.V[prev]
#          value = self.V[prev] + self.alpha*(target - self.V[prev])
#          self.V[prev] = value
#          target = value
        self.reset_history()


class Human:
    def __init__(self, sym):
        self.sym = sym
        
    def take_action(self, env):
        while True:
          # break if we make a legal move
          move = input("Enter coordinates i,j for your next move (i,j=0..2): ")
          i, j = move.split(',')
          i = int(i)
          j = int(j)
          if env.is_empty(i, j):
            env.board[i,j] = self.sym
            break

    def update(self, env):
        pass

    def update_state_history(self, s):
        pass    

def play_game(p1, p2, env, disp=False):
      # loops until the game is over
      current_player = None
      while not env.game_over():
            # alternate between players
            # p1 always starts first
            if current_player == p1:
                current_player = p2
            else:
                current_player = p1
        
            # draw the board before the user who wants to see it makes a move
            if disp:
                if disp == 1 and current_player == p1:
                    env.display_board()
                if disp == 2 and current_player == p2:
                    env.display_board()
        
            # current player makes a move
            current_player.take_action(env)
        
            # update state histories
            state = env.get_state()
            p1.update_state_history(state)
            p2.update_state_history(state)
        
#            if disp:
#                env.display_board()
      # do the value function update
      p1.update(env)
      p2.update(env)
     
if __name__ == '__main__':
    np.random.seed( 30 )
    sizeX=3
    sizeY=3
    winLen =3
    env = Environment(sizeX, sizeY, winLen)
    max_state = env.get_max_state()
    
    # train the agent
    p1 = Agent(env.x)
    p1.setV(np.zeros(max_state))
    p2 = Agent(env.o)
    p2.setV(np.zeros(max_state))
    
    for t in range(10000):
        if t % 200 == 0:
            print(t)
        play_game(p1, p2, Environment(sizeX, sizeY, winLen), 0)
        
    #switch to Human
    human = Human(env.o)
    p1.set_verbose(True)
    p1.setEps(0)
    while True:
        play_game(p1, human, Environment(sizeX, sizeY, winLen), 2)
        answer = input("Again?")
        if answer.lower() == 'n':
            break
