use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn::{loss, AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use chess::{Board, ChessMove, Color};
use clap::Parser;
use pgn_reader::{BufferedReader, SanPlus, Visitor};
use rayon::prelude::*;
use std::fs::File;
use std::path::PathBuf;

use chessers::mlp::Mlp;
use chessers::{board_to_tensor, move_to_index};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the PGN file for training.
    #[arg(long)]
    pgn_file: PathBuf,

    /// Path to save the trained model weights.
    #[arg(long, default_value = "chess_mlp.safetensors")]
    output_file: PathBuf,

    /// Number of epochs to train for.
    #[arg(long, default_value_t = 10)]
    epochs: usize,

    /// Learning rate for the optimizer.
    #[arg(long, default_value_t = 1e-4)]
    learning_rate: f64,
}
/// A visitor that collects all board states and the subsequent move for a single game.
/// It only collects positions where it is White's turn to move.
struct GameDataCollector {
    board: Board,
    // Stores (board_state, resulting_move)
    positions: Vec<(Board, ChessMove)>,
}

impl GameDataCollector {
    fn new() -> Self {
        Self {
            board: Board::default(),
            positions: Vec::new(),
        }
    }
}

impl Visitor for GameDataCollector {
    // The return type of the visitor when it's done.
    type Result = Result<Vec<(Board, ChessMove)>>;

    fn san(&mut self, san: SanPlus) {
        // We only train on positions where it's White's turn.
        // A more robust model would handle both perspectives.
        if let Ok(chess_move) = ChessMove::from_san(&self.board, &san.to_string()) {
            if self.board.side_to_move() == Color::White {
                self.positions.push((self.board, chess_move));
            }
            // Apply the move to advance the board state for the next position.
            self.board = self.board.make_move_new(chess_move);
        }
    }

    fn end_game(&mut self) -> Self::Result {
        // Return the collected positions, consuming the vec.
        Ok(std::mem::take(&mut self.positions))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Set the OPENBLAS_NUM_THREADS environment variable to 1.
    // This allows Rayon to manage the parallelism at a higher level.
    // See: https://github.com/huggingface/candle/issues/213
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device);

    // 1. Initialize model and optimizer
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Mlp::new(vb)?;
    let mut optimizer = AdamW::new(model.vars(), ParamsAdamW { lr: args.learning_rate , beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 })?;

    println!(
        "Training with {} epochs and a learning rate of {}.",
        args.epochs, args.learning_rate
    );

    const GAMES_PER_BATCH: usize = 32;

    // 2. Training loop
    for epoch in 0..args.epochs {
        // We re-open the file for each epoch to iterate from the beginning.
        let pgn_file = File::open(&args.pgn_file)?;
        let mut reader = BufferedReader::new(pgn_file);

        let mut total_epoch_loss = 0.0;
        let mut total_moves_in_epoch = 0;
        let mut batch_count = 0;

        loop {
            // --- Data Collection Phase ---
            let mut games_data = Vec::with_capacity(GAMES_PER_BATCH);
            for _ in 0..GAMES_PER_BATCH {
                let mut batch_data = vec![];
                for _ in 0..GAMES_PER_BATCH {
                    let mut collector = GameDataCollector::new();
                    match reader.read_game(&mut collector)? {
                        Some(Ok(game_positions)) if !game_positions.is_empty() => {
                            batch_data.extend(game_positions);
                        }
                        Some(_) => {} // Skip empty or failed games
                        None => break, // End of file
                    }
                }
                games_data.push(batch_data)
            }

            if games_data.is_empty() {
                break; // No more games in the file
            } else {
                println!("Read {} games", games_data.len());
            }

            let moves_in_batch = games_data.len();
            total_moves_in_epoch += moves_in_batch;
            batch_count += 1;

            // --- Parallel Processing and Gradient Accumulation ---
            let batch_losses: Vec<Tensor> = games_data
                .par_iter()
                .map(|batch| {
                    batch
                    .iter()
                    .map(|(board, chess_move)| {
                        let board_tensor = board_to_tensor(board, &device)?;
                        let logits = model.forward(&board_tensor)?; // Shape [4096]
                        Ok(logits.unsqueeze(0)) // Shape [1, 4096]
                    })
                    .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<Vec<_>>>>()?
                .into_iter()
                .flatten()
                .collect::<Result<Vec<_>, candle_core::Error>>()?;

            // Create targets tensor
            let targets: Vec<u32> = games_data
                .iter()
                .flatten()
                .map(|(_, chess_move)| move_to_index(*chess_move) as u32)
                .collect();
            let targets_tensor = Tensor::new(targets.as_slice(), &device)?;

            // Calculate loss
            let logits_tensor = Tensor::cat(&batch_losses, 0)?;
            let total_loss = loss::cross_entropy(&logits_tensor, &targets_tensor)?;
            optimizer.backward_step(&total_loss)?;

            println!("Saving progress to {:?}", args.output_file);
            varmap.save(&args.output_file)?;

            total_epoch_loss += total_loss.to_scalar::<f32>()?;
        }

        let avg_loss = total_epoch_loss / total_moves_in_epoch as f32;
        println!( "Epoch: {:4} | Batches: {:5} | Avg Loss: {:8.5}", epoch + 1, batch_count, avg_loss);
    }

    println!("Training complete. Saving model to {:?}", args.output_file);
    varmap.save(&args.output_file)?;

    Ok(())
}
