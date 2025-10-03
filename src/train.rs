use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn::{loss, AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use chess::{Board, ChessMove, Color};
use clap::Parser;
use pgn_reader::{BufferedReader, SanPlus, Visitor};
use rayon::prelude::*;
use std::fs::File;
use std::path::PathBuf; use std::time::Instant;

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
    // std::env::set_var("OPENBLAS_NUM_THREADS", "1");

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device);

    // 1. Initialize model and optimizer
    let mut varmap = VarMap::new();
    if std::path::PathBuf::from(&args.output_file).exists() {
        println!("Loading model from {:?}", args.output_file);
        varmap.load(&args.output_file)?;
    }
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Mlp::new(vb)?;
    let mut optimizer = AdamW::new(model.vars(), ParamsAdamW { lr: args.learning_rate , beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 })?;

    println!(
        "Training with {} epochs and a learning rate of {}.",
        args.epochs, args.learning_rate
    );

    const BATCH_SIZE: usize = 1024 * 32; // Number of positions per batch
    const EPOCH_SIZE: usize = 16; // Number of batches per epoch

    // 2. Training loop
    for epoch in 0..args.epochs {
        // We re-open the file for each epoch to iterate from the beginning.
        let pgn_file = File::open(&args.pgn_file)?;
        let mut reader = BufferedReader::new(pgn_file);

        let mut total_epoch_loss = 0.0;
        let mut total_moves_in_epoch = 0;
        let mut batch_num = 0;

        loop {
            // --- Data Collection Phase ---
            let data_loading_start = Instant::now();
            let mut batch_positions = Vec::with_capacity(BATCH_SIZE);
            let mut eof = false;
            while batch_positions.len() < BATCH_SIZE {
                let mut collector = GameDataCollector::new();
                match reader.read_game(&mut collector)? {
                    Some(Ok(game_positions)) if !game_positions.is_empty() => {
                        batch_positions.extend(game_positions);
                    }
                    Some(_) => {} // Skip empty or failed games
                    None => {
                        eof = true;
                        break;
                    } // End of file
                }
            }
            let data_loading_duration = data_loading_start.elapsed();

            if batch_positions.is_empty() {
                break; // No more positions to process
            }

            let moves_in_batch = batch_positions.len();
            total_moves_in_epoch += moves_in_batch;
            batch_num += 1;

            // --- Parallel Processing and Gradient Accumulation ---
            let tensor_conv_start = Instant::now();
            let (board_tensors, target_indices): (Vec<_>, Vec<_>) = batch_positions
                .par_iter()
                .map(|(board, chess_move)| {
                    let board_tensor = board_to_tensor(board, &device).expect("Failed to convert board to tensor");
                    let move_index = move_to_index(*chess_move) as u32;
                    (board_tensor, move_index)
                })
                .unzip();
            let tensor_conv_duration = tensor_conv_start.elapsed();

            let forward_pass_start = Instant::now();
            let input_tensor = Tensor::stack(&board_tensors, 0)?;
            let logits = model.forward(&input_tensor)?;
            let forward_pass_duration = forward_pass_start.elapsed();

            let loss_start = Instant::now();
            let targets_tensor = Tensor::new(target_indices.as_slice(), &device)?;
            let total_loss = loss::cross_entropy(&logits, &targets_tensor)?;
            let loss_duration = loss_start.elapsed();

            let loss_backward_start = Instant::now();
            optimizer.backward_step(&total_loss)?;
            let loss_backward_duration = loss_backward_start.elapsed();

            println!(
                "Batch {:<5} | Positions: {:<4} | Data: {:<10?} | Tensors: {:<10?} | Forward: {:<10?} | Loss: {:<10?} | Backward: {:<10?}",
                batch_num, moves_in_batch, data_loading_duration, tensor_conv_duration, forward_pass_duration, loss_duration, loss_backward_duration
            );

            total_epoch_loss += total_loss.to_scalar::<f32>()?;

            if eof || batch_num >= EPOCH_SIZE {
                break;
            }
        }

        let avg_loss = total_epoch_loss / total_moves_in_epoch as f32;
        println!( "Epoch: {:4} | Batches: {:5} | Avg Loss: {:8.5}", epoch + 1, batch_num, avg_loss);

        println!("Saving model after epoch {} to {:?}", epoch + 1, args.output_file);
        varmap.save(&args.output_file)?;
    }

    println!("Training complete. Saving model to {:?}", args.output_file);
    varmap.save(&args.output_file)?;

    Ok(())
}
