use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

use chessers::model::{load_model, ScoringNet, UNet};
use chessers::{board_and_move_to_tensor, find_best_move_with_search};
use chess::{Board, BoardStatus, ChessMove, Color, MoveGen};
use rand::seq::IteratorRandom;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of games to generate and train on per epoch.
    #[arg(long, default_value_t = 1000)]
    games_per_epoch: usize,

    /// Path to save the trained scoring model weights.
    #[arg(long, default_value = "chess_scorer.safetensors")]
    output_file: PathBuf,

    /// Path to the policy/value model used for game generation.
    #[arg(long, default_value = "chess_6.safetensors")]
    generator_model_file: PathBuf,

    /// Number of epochs to train for.
    #[arg(long, default_value_t = 10)]
    epochs: usize,

    /// Learning rate for the optimizer.
    #[arg(long, default_value_t = 1e-4)]
    learning_rate: f64,
}

#[derive(Debug, Clone, Copy)]
enum GameResult {
    WhiteWin,
    BlackWin,
    Draw,
}

/// Plays a full game of chess with the policy/value model playing against itself.
/// Returns the game history and the result.
fn play_game(
    generator_model: &UNet,
    device: &Device,
    exploration_rate: f32,
) -> Result<(Vec<(Board, ChessMove)>, GameResult)> {
    let mut board = Board::default();
    let mut game_history = Vec::new();
    let mut move_count = 0;

    loop {
        if board.status() != BoardStatus::Ongoing || move_count > 200 {
            break;
        }

        let best_move = if rand::random::<f32>() < exploration_rate {
            // Exploration: pick a random move
            let moves = MoveGen::new_legal(&board);
            moves.choose(&mut rand::thread_rng())
        } else {
            // Exploitation: use the generator model to find the best move
            find_best_move_with_search(&board, generator_model, device)
        };

        if let Some(chess_move) = best_move {
            game_history.push((board, chess_move));
            board = board.make_move_new(chess_move);
            move_count += 1;
        } else {
            break;
        }
    }

    let result = match board.status() {
        BoardStatus::Checkmate | BoardStatus::Stalemate if move_count < 200 => {
            if board.side_to_move() == Color::White {
                GameResult::BlackWin
            } else {
                GameResult::WhiteWin
            }
        }
        _ => GameResult::Draw,
    };

    Ok((game_history, result))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device);

    // 1. Load the GENERATOR model (the existing policy/value UNet)
    println!(
        "Loading generator model from {:?}",
        args.generator_model_file
    );
    let generator_model = load_model(&args.generator_model_file, &device)?;
    println!("Generator model loaded successfully.");

    // 2. Initialize the new SCORING model and optimizer
    let mut varmap = VarMap::new();
    if args.output_file.exists() {
        println!("Loading scoring model from {:?}", args.output_file);
        varmap.load(&args.output_file)?;
    }
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let scoring_model = ScoringNet::new(vb)?;
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.learning_rate,
            ..Default::default()
        },
    )?;

    println!(
        "Training scoring model with {} epochs and a learning rate of {}.",
        args.epochs, args.learning_rate
    );

    const BATCH_SIZE: usize = 1024;

    for epoch in 0..args.epochs {
        let epoch_start_time = Instant::now();
        println!("--- Starting Self-Play Epoch {}/{} ---", epoch + 1, args.epochs);

        // --- Data Generation Phase ---
        let data_gen_start = Instant::now();
        let games_data: Vec<_> = (0..args.games_per_epoch)
            .into_par_iter()
            .map(|_| {
                let exploration_rate = (0.5 * (1.0 - (epoch as f32 / args.epochs as f32))).max(0.1);
                play_game(&generator_model, &device, exploration_rate).unwrap()
            })
            .collect();
        let data_gen_duration = data_gen_start.elapsed();

        let training_data: Vec<_> = games_data
            .into_iter()
            .flat_map(|(history, result)| {
                let outcome_value = match result {
                    GameResult::WhiteWin => 1.0f32,
                    GameResult::BlackWin => -1.0f32,
                    GameResult::Draw => 0.0f32,
                };
                history.into_iter().map(move |(b, m)| {
                    (
                        b,
                        m,
                        if b.side_to_move() == Color::White {
                            outcome_value
                        } else {
                            -outcome_value
                        },
                    )
                })
            })
            .collect();

        println!(
            "Generated {} games ({} positions) in {:.3?}",
            args.games_per_epoch,
            training_data.len(),
            data_gen_duration
        );

        // --- Training Phase ---
        let mut total_epoch_loss = 0.0;
        for (batch_num, batch) in training_data.chunks(BATCH_SIZE).enumerate() {
            let (input_tensors, target_values): (Vec<_>, Vec<_>) = batch
                .par_iter()
                .map(|(board, chess_move, value)| {
                    let input_tensor = board_and_move_to_tensor(board, *chess_move, &device)
                        .expect("Failed to convert board and move to tensor");
                    (input_tensor, *value)
                })
                .unzip();

            let input_tensor = Tensor::stack(&input_tensors, 0)?;
            let target_tensor = Tensor::from_vec(target_values, (batch.len(), 1), &device)?;

            let predictions = scoring_model.forward_is_training(&input_tensor, true)?;
            let loss = loss::mse(&predictions, &target_tensor)?;

            optimizer.backward_step(&loss)?;

            total_epoch_loss += loss.to_scalar::<f32>()?;

            if (batch_num + 1) % 10 == 0 {
                println!(
                    "  Batch {:<5} | Avg Loss: {:.5}",
                    batch_num + 1,
                    total_epoch_loss / ((batch_num + 1) * BATCH_SIZE) as f32
                );
            }
        }

        let epoch_duration = epoch_start_time.elapsed();
        println!(
            "Epoch: {:4} | Avg Loss: {:8.5} | Duration: {:.3?}",
            epoch + 1,
            total_epoch_loss / training_data.len() as f32,
            epoch_duration
        );

        println!("Saving scoring model to {:?}", args.output_file);
        varmap.save(&args.output_file)?;
    }

    Ok(())
}