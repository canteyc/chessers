# Chessers

Chess bot using a convolutional network to predict optimal moves.

## Concept

### Bot Design
On each turn, the board position is converted to a 6-channel tensor from the perspective of the current player.
Each of the 6 piece types (Pawn, Knight, Bishop, Rook, Queen, King) gets its own 8x8 channel.
On each channel, friendly pieces are marked as `1.0` and opponent pieces are marked as `-1.0`.
When it is Black's turn, the board is flipped vertically so the model always "sees" the board from the perspective of the player to move.
The network output is a 4097 length array. The last value is a value estimate of how likely this side is to win the game.
The other values are a flattened 64*64 grid of weights for moving from each square to another square.
Together, the output array can be applied to every legal move to find the highest scoring move.

### Training
The network can be trained using PGN data (supervised) or using self-play (unsupervised).
Use the flag `--pgn-file` to set the file path to the pgn file. If you omit that flag, then self-play will be used.

## Status
Currently, the project has a simple gui for playing against the bot. Training is conducted headless for better performance, with only the latest weights saved after each epoch.
Every game is deterministic, so running two models together will always produce the same sequence of moves.
The network itself is a small ResNet, consiting of an opening convolution, two residual blocks, and a final convolution. The winning probability is tacked on as an additional linear layer.
It is clear that performance changes based on training, but the network may be too small to learn any meaningful strategy.

## Vision
Next steps are to add better performance tracking, probably by saving past models and ranking them through round-robin tournaments.
Then the network can be expanded and hyper parameters tuned to find a network that may actually make reasonable choices.
My current goal is to surpass approximately 800 Elo (me).
