use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{loss, AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;
use pgn_reader::{BufferedReader, Color as PgnColor, Outcome as PgnOutcome, SanPlus, Visitor};
use rand::seq::IteratorRandom;
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

use chessers::model::UNet;
use chessers::{board_to_tensor, move_to_index};
use chess::{Board, BoardStatus, ChessMove, Color, MoveGen};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of games to generate and train on per epoch.
    #[arg(long, default_value_t = 1000)]
    games_per_epoch: usize,

    /// Path to save the trained model weights.
    #[arg(long, default_value = "chess_mlp.safetensors")]
    output_file: PathBuf,

    /// Number of epochs to train for.
    #[arg(long, default_value_t = 10)]
    epochs: usize,

    /// Learning rate for the optimizer.
    #[arg(long, default_value_t = 1e-4)]
    learning_rate: f64,

    /// Path to a PGN file for training. If not provided, self-play will be used.
    #[arg(long)]
    pgn_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
enum GameResult {
    WhiteWin,
    BlackWin,
    Draw,
}

/// A visitor for the PGN parser to extract game data.
#[derive(Default)]
struct GameVisitor {
    games: Vec<(Vec<ChessMove>, GameResult)>,
    board: Board,
}

impl Visitor for GameVisitor {
    type Result = ();

    fn begin_game(&mut self) {
        // A new game is starting. Reset the board and add a new game entry.
        self.board = Board::default();
        self.games.push((Vec::new(), GameResult::Draw)); // Placeholder result
    }
    
    fn san(&mut self, san_plus: SanPlus) {
        // Convert the SAN to a string and parse it with the `chess` crate's board.
        let san_str = san_plus.san.to_string();
        if let Ok(chess_move) = ChessMove::from_san(&self.board, &san_str) {
            if let Some(last_game) = self.games.last_mut() {
                last_game.0.push(chess_move);
            }
            // Play the move on our internal board to keep it in sync.
            self.board = self.board.make_move_new(chess_move);
        }
    }

    fn outcome(&mut self, outcome: Option<PgnOutcome>) {
        if let Some(last_game) = self.games.last_mut() {
            last_game.1 = match outcome {
                Some(PgnOutcome::Decisive { winner: PgnColor::White }) => GameResult::WhiteWin,
                Some(PgnOutcome::Decisive { winner: PgnColor::Black }) => GameResult::BlackWin,
                _ => GameResult::Draw, // Includes draws and unknown results.
            };
        }
    }

    fn end_game(&mut self) -> Self::Result {
        // This method is called at the end of each game. We don't need to do anything special here
        // since the outcome is handled by the `outcome` method.
    }
}

/// Plays a full game of chess with the model playing against itself.
/// Returns the game history and the result.
fn play_game(model: &UNet, device: &Device, exploration_rate: f32) -> Result<(Vec<(Board, ChessMove)>, GameResult)> {
    let mut board = Board::default();
    let mut game_history = Vec::new();
    let mut move_count = 0;

    loop {
        if board.status() != BoardStatus::Ongoing || move_count > 200 { // Max moves to prevent infinite games
            break;
        }

        let best_move = if rand::random::<f32>() < exploration_rate {
            // Exploration: pick a random move
            let moves = MoveGen::new_legal(&board);
            moves.choose(&mut rand::thread_rng())
        } else {
            // Exploitation: use the model to find the best move
            find_model_move(&board, model, device)
        };

        if let Some(chess_move) = best_move {
            game_history.push((board, chess_move));
            board = board.make_move_new(chess_move);
            move_count += 1;
        } else {
            // No legal moves, game is over.
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
        },
        _ => GameResult::Draw, // Stalemate, insufficient material, etc.
    };

    Ok((game_history, result))
}

/// Uses the MLP to find the best legal move from a given board state.
fn find_model_move(board: &Board, model: &UNet, device: &Device) -> Option<ChessMove> {
    let board_tensor = board_to_tensor(board, device).ok()?;
    let output = match model.forward(&board_tensor) {
        Ok(tensor) => tensor,
        Err(e) => {
            eprintln!("Error during forward pass: {:?}", e);
            return None;
        }
    };

    // The first 4096 elements are policy logits
    let logits = output.i((.., ..4096)).ok()?;
    let mut best_move: Option<ChessMove> = None;
    let mut max_logit = f32::NEG_INFINITY;

    for m in MoveGen::new_legal(board) {
        let move_index = move_to_index(m);
        let move_logit = logits.get(0).ok()?.get(move_index).ok()?.to_scalar::<f32>().ok()?;
        if move_logit > max_logit {
            max_logit = move_logit;
            best_move = Some(m);
        }
    }

    best_move
}

/// Loads training data from a PGN file.
fn load_training_data_from_pgn(
    pgn_path: &PathBuf,
) -> Result<Vec<(Board, ChessMove, f32)>> {
    println!("Loading training data from PGN: {}", pgn_path.display());
    let pgn_file = std::fs::File::open(pgn_path)?;
    let mut reader = BufferedReader::new(pgn_file);

    let mut visitor = GameVisitor::default();
    reader.read_all(&mut visitor)?;

    let training_data: Vec<_> = visitor
        .games
        .into_par_iter()
        .flat_map(|(moves, result)| {
            let mut board = Board::default();
            let outcome_value = match result {
                GameResult::WhiteWin => 1.0f32,
                GameResult::BlackWin => -1.0f32,
                GameResult::Draw => 0.0f32,
            };

            moves.into_iter().filter_map(move |m| {
                let current_board = board;
                if board.legal(m) {
                    board = board.make_move_new(m);
                    Some((current_board, m, if current_board.side_to_move() == Color::White { outcome_value } else { -outcome_value }))
                } else { None }
            }).collect::<Vec<_>>()
        }).collect();
    Ok(training_data)
}

fn main() -> Result<()> {
    let args = Args::parse();

    // // --- Profiling Setup (pprof-rs) ---
    // // This guard starts the profiler and will generate the report when it's dropped at the end of main.
    // let guard = pprof::ProfilerGuardBuilder::default()
    //     .frequency(1000) // Sample at 1000Hz
    //     .blocklist(&["libc", "libgcc", "pthread", "vdso"]) // Exclude some low-level noise
    //     .build()?;

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
    let model = UNet::new(vb)?;
    let mut optimizer = AdamW::new(varmap.all_vars(), ParamsAdamW { lr: args.learning_rate , beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 })?;

    println!(
        "Training with {} epochs and a learning rate of {}.",
        args.epochs, args.learning_rate
    );

    const BATCH_SIZE: usize = 1024 * 4; // Number of positions per training batch

    // 2. Load data or start training loop for self-play
    if let Some(pgn_path) = &args.pgn_file {
        // --- PGN Training ---
        let training_data = load_training_data_from_pgn(pgn_path)?;
        println!("Loaded {} positions from PGN file.", training_data.len());
        train_on_data(&training_data, &model, &mut optimizer, &device, args.epochs, BATCH_SIZE, &args.output_file, &mut varmap)?;
    } else {
        // --- Self-Play Training ---
        for epoch in 0..args.epochs {
            let epoch_start_time = Instant::now();
            println!("--- Starting Self-Play Epoch {}/{} ---", epoch + 1, args.epochs);

            // --- Data Generation Phase ---
            let data_gen_start = Instant::now();
            let games_data: Vec<_> = (0..args.games_per_epoch)
                .into_par_iter()
                .map(|_| {
                    // Anneal exploration rate over epochs
                    let exploration_rate = (0.5 * (1.0 - (epoch as f32 / args.epochs as f32))).max(0.1);
                    play_game(&model, &device, exploration_rate).unwrap()
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
                    history
                        .into_iter()
                        .map(move |(b, m)| (b, m, if b.side_to_move() == Color::White { outcome_value } else { -outcome_value }))
                })
                .collect();

            println!("Generated {} games ({} positions) in {:?}", args.games_per_epoch, training_data.len(), data_gen_duration);
            train_on_data(&training_data, &model, &mut optimizer, &device, 1, BATCH_SIZE, &args.output_file, &mut varmap)?;
            let epoch_duration = epoch_start_time.elapsed();
            println!( "Epoch: {:4} | Duration: {:?}", epoch + 1, epoch_duration);
        }
    }

    fn train_on_data(training_data: &[(Board, ChessMove, f32)], model: &UNet, optimizer: &mut AdamW, device: &Device, epochs: usize, batch_size: usize, output_file: &PathBuf, varmap: &mut VarMap) -> Result<()> {
        if training_data.is_empty() {
            println!("No training data provided. Skipping training.");
            return Ok(());
        }

        for epoch in 0..epochs {
            let epoch_start_time = Instant::now();
            println!("--- Starting Training Epoch {}/{} ---", epoch + 1, epochs);

            let mut total_epoch_loss = 0.0;
            for (batch_num, batch) in training_data.chunks(batch_size).enumerate() {
                let batch_start_time = Instant::now();

                // --- Parallel Processing and Gradient Accumulation ---
                let (board_tensors, target_indices, target_values): (Vec<_>, Vec<_>, Vec<_>) = batch
                    .par_iter()
                    .map(|(board, chess_move, value)| {
                        let board_tensor = board_to_tensor(board, &device).expect("Failed to convert board to tensor");
                        let move_index = move_to_index(*chess_move) as u32;
                        (board_tensor, move_index, *value)
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .fold((Vec::new(), Vec::new(), Vec::new()), |mut acc, (b, m, v)| {
                        acc.0.push(b);
                        acc.1.push(m);
                        acc.2.push(v);
                        acc
                    });

                let input_tensor = Tensor::stack(&board_tensors, 0)?;
                let output = model.forward(&input_tensor)?;
                let policy_logits = output.i((.., ..4096))?;
                let value_preds = output.i((.., 4096..))?.squeeze(1)?;

                let policy_targets = Tensor::new(target_indices.as_slice(), &device)?;
                let value_targets = Tensor::new(target_values.as_slice(), &device)?;

                let policy_loss = loss::cross_entropy(&policy_logits, &policy_targets)?;
                let value_loss = loss::mse(&value_preds, &value_targets)?;
                let total_loss = (policy_loss.to_device(&device)? + value_loss.to_device(&device)?)?;

                optimizer.backward_step(&total_loss)?;

                let batch_loss = total_loss.to_scalar::<f32>()?;
                total_epoch_loss += batch_loss;

                if (batch_num + 1) % 10 == 0 {
                    println!(
                        "  Batch {:<5} | Positions: {:<4} | Loss: {:.5} | Duration: {:?}",
                        batch_num + 1, batch.len(), batch_loss / batch.len() as f32, batch_start_time.elapsed()
                    );
                }
            }

            let avg_loss = total_epoch_loss / training_data.len() as f32;
            let epoch_duration = epoch_start_time.elapsed();
            println!( "Epoch: {:4} | Avg Loss: {:8.5} | Duration: {:?}", epoch + 1, avg_loss, epoch_duration);

            println!("Saving model after epoch {} to {:?}", epoch + 1, output_file);
            varmap.save(output_file)?;
        }
        Ok(())
    }

    println!("Training complete. Saving model to {:?}", args.output_file);
    varmap.save(&args.output_file)?;

    // // --- Profiling Report Generation ---
    // if let Ok(report) = guard.report().build() {
    //     println!("Generating flamegraph to flamegraph.svg...");
    //     let file = std::fs::File::create("flamegraph.svg")?;
    //     let mut options = pprof::flamegraph::Options::default();
    //     options.image_width = Some(2400); // Set a custom width
    //     options.title = "Chessers Training Flamegraph".to_string();
    //     report.flamegraph_with_options(file, &mut options)?;
    // }

    Ok(())
}