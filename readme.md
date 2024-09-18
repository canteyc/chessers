# Chessers

Chess bot using a convolutional network on a 6-channel bitboard to predict optimal moves  

## Concept

### Bot Design
On each turn, the board position is converted to a 6-channel bitboard recording the positions of every piece. 
Friendly pieces are coded with a 1, enemy pieces with a -1, and empty squares with a 0. 
The 6 channels encode each piece type: pawns, rooks, knights, bishops, kings, and queens. 
The network output is a 2-channel 8x8 array. The first channel shows how desirable it is to move _away_ from each square. 
The second channel shows how desirable it is to move _to_ each square. 
Together, the output array can be applied to every legal move to find the highest scoring move.  

### Training
Although scoring individual moves could be accomplished with an evaluator like Stockfish, to start with a genetic training approach is used to simply rank generations of agents in a round-robin tournament.
After each tournament, pairs of agents are chosen based on a weighted sampling from victory totals. Each pair is then merged, weighted towards the victor of their specific match.
This merging is run until a new generation is created, and the process repeats until training stops after a set number of epochs.

## Status
Currently, the project has a simple command line ui for playing against the bots. Training is conducted headless for better performance, with every generation of model weights saved.
Every game is deterministic, so running two models together will always produce the same sequence of moves. 
The network itself is just a two convolutional layers with a relu in between, the simplest model that could actually benefit from training. 
It is clear that performance changes based on training, but the network is obviously too small to learn any meaningful strategy.

## Vision
Next steps are to add an evaluation function which pits the champion of each generation to the previous generations, to measure change (hopefully improvement) over time. 
Then the network can be expanded and hyper parameters tuned to find a network that may actually make reasonable choices. 
My current goal is to surpass approximately 800 Elo (me).
