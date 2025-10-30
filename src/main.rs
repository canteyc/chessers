use candle_core::{Device};
use chess::{Board, BoardStatus, ChessMove, Color, File, MoveGen, Piece, Rank, Square};
use eframe::egui::{self, Color32, Rect, Sense, Vec2};
use rand::seq::IteratorRandom;
use std::path::Path;
use std::fs::OpenOptions;
use std::io::Write;
 
use chessers::{find_best_move_with_search, model::{load_model, UNet}, simple_bot::find_simple_move};

fn main() {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Chessers",
        native_options,
        Box::new(|_cc| Box::new(ChessApp::new())),
    )
    .expect("Failed to run eframe");
}

#[derive(Debug, PartialEq)]
enum Player {
    Human,
    // We can have different bot types
    Bot(BotModel),
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BotModel {
    Random,
    UNet,
    Simple,
}

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
struct ChessApp {
    board: Board,
    selected_square: Option<Square>,
    white_player: Player,
    black_player: Player,
    model: UNet,
    promotion_move: Option<(Square, Square)>,
    model_status: String,
    game_history: Vec<ChessMove>,
}

impl ChessApp {
    fn new() -> Self {
        let device = Device::Cpu;
        let model_path = Path::new("chess_6.safetensors");

        let (model, model_status) = match load_model(model_path, &device) {
            Ok(model) => (model, format!("Loaded model from: {}", model_path.display())),
            Err(e) => panic!("Failed to load model: {}. Please run the training script first.", e),
        };

        Self {
            board: Board::default(),
            selected_square: None,
            white_player: Player::Bot(BotModel::Simple), // Default to Human vs Bot
            // white_player: Player::Human, // Default to Human vs Bot
            // black_player: Player::Bot(BotModel::Random),
            black_player: Player::Human,
            model,
            promotion_move: None,
            model_status,
            game_history: Vec::new(),
        }
    }

    /// Gets the `Player` whose turn it is.
    fn current_player(&self) -> &Player {
        match self.board.side_to_move() {
            Color::White => &self.white_player,
            Color::Black => &self.black_player,
        }
    }

    /// Dispatches to the correct bot implementation.
    fn make_bot_move(&mut self, model: BotModel) {
        let best_move = match model {
            BotModel::Random => self.find_random_move(),
            BotModel::UNet => self.find_model_move(),
            BotModel::Simple => find_simple_move(&self.board),
        };

        if let Some(chess_move) = best_move {
            self.board = self.board.make_move_new(chess_move);
            self.game_history.push(chess_move);
        }
    }

    /// Saves the current game history to a PGN file.
    fn save_game_to_pgn(&self) -> std::io::Result<()> {
        let mut file = OpenOptions::new().append(true).create(true).open("human_games.pgn")?;

        let result_str = match self.board.status() {
            BoardStatus::Checkmate => if self.board.side_to_move() == Color::White { "0-1" } else { "1-0" },
            BoardStatus::Stalemate => "1/2-1/2",
            _ => "*", // Game is ongoing or drawn by other means
        };

        // Write PGN headers
        writeln!(file, "[Event \"Human vs Bot\"]")?;
        writeln!(file, "[Site \"Local\"]")?;
        writeln!(file, "[Date \"{}\"]", chrono::Local::now().format("%Y.%m.%d"))?;
        writeln!(file, "[Round \"-\"]")?;
        writeln!(file, "[White \"{}\"]", self.white_player)?;
        writeln!(file, "[Black \"{}\"]", self.black_player)?;
        writeln!(file, "[Result \"{}\"]", result_str)?;
        writeln!(file)?;

        let mut board = Board::default();
        let mut move_line = String::new();
        for (i, &chess_move) in self.game_history.iter().enumerate() {
            if i % 2 == 0 {
                move_line.push_str(&format!("{}. ", i / 2 + 1));
            }
            let san = chess_move.to_string();
            println!("SAN: {}", san);
            move_line.push_str(&san);
            move_line.push(' ');
            board = board.make_move_new(chess_move);
        }

        writeln!(file, "{} {}", move_line.trim(), result_str)?;
        writeln!(file)?;
        Ok(())
    }

    /// Bot implementation: picks a random legal move.
    fn find_random_move(&self) -> Option<ChessMove> {
        let moves = MoveGen::new_legal(&self.board);
        moves.choose(&mut rand::thread_rng())
    }

    /// Bot implementation: uses a UNet  to pick a move.
    fn find_model_move(&self) -> Option<ChessMove> {
        find_best_move_with_search(&self.board, &self.model, &Device::Cpu)
    }

    /// Draws the board and handles user input.
    fn draw_board_and_handle_input(&mut self, ui: &mut egui::Ui) {
        const SQUARE_SIZE: f32 = 60.0;
        const BOARD_SIZE: f32 = 8.0 * SQUARE_SIZE;
        const LIGHT_SQUARE_COLOR: Color32 = Color32::from_rgb(240, 217, 181);
        const DARK_SQUARE_COLOR: Color32 = Color32::from_rgb(181, 136, 99);
        const SELECTED_SQUARE_COLOR: Color32 = Color32::from_rgb(135, 152, 106);

        // Allocate space for the board
        let (response, painter) =
            ui.allocate_painter(Vec2::new(BOARD_SIZE, BOARD_SIZE), Sense::click());
        let board_rect = response.rect;

        // --- Draw squares and pieces ---
        for rank_idx in 0..8 {
            for file_idx in 0..8 {
                let square = Square::make_square(Rank::from_index(rank_idx), File::from_index(file_idx));
                let is_light = (rank_idx + file_idx) % 2 != 0;

                let mut square_color = if is_light { LIGHT_SQUARE_COLOR } else { DARK_SQUARE_COLOR };
                if self.selected_square == Some(square) {
                    square_color = SELECTED_SQUARE_COLOR;
                }

                let top_left = board_rect.left_top() + Vec2::new(file_idx as f32 * SQUARE_SIZE, (7 - rank_idx) as f32 * SQUARE_SIZE);
                let rect = Rect::from_min_size(top_left, Vec2::new(SQUARE_SIZE, SQUARE_SIZE));
                painter.rect_filled(rect, 0.0, square_color);

                // Draw piece
                if let Some(piece) = self.board.piece_on(square) {
                    let piece_color = self.board.color_on(square).unwrap();
                    let piece_char = get_piece_char(piece, piece_color);
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        piece_char,
                        egui::FontId::monospace(SQUARE_SIZE * 0.8),
                        if piece_color == Color::White { Color32::WHITE } else { Color32::BLACK },
                    );
                }
            }
        }

        // --- Handle Clicks ---
        if response.clicked() {
            if let Some(click_pos) = response.interact_pointer_pos() {
                let file_idx = ((click_pos.x - board_rect.left()) / SQUARE_SIZE).floor() as usize;
                let rank_idx = 7 - ((click_pos.y - board_rect.top()) / SQUARE_SIZE).floor() as usize;

                if file_idx < 8 && rank_idx < 8 {
                    let clicked_square = Square::make_square(Rank::from_index(rank_idx), File::from_index(file_idx));

                    if let Some(start_square) = self.selected_square {
                        // This is the second click (destination)
                        let piece = self.board.piece_on(start_square).unwrap();
                        let rank = clicked_square.get_rank();

                        // Check if it's a promotion move
                        if piece == Piece::Pawn && (rank == Rank::First || rank == Rank::Eighth) {
                            // It's a promotion, so we set the state to ask the user for the piece.
                            self.promotion_move = Some((start_square, clicked_square));
                        } else {
                            // It's a regular move.
                            let chess_move = ChessMove::new(start_square, clicked_square, None);
                            if self.board.legal(chess_move) {
                                self.board = self.board.make_move_new(chess_move);
                                self.game_history.push(chess_move);
                            }
                        }

                        // Deselect after attempting a move
                        self.selected_square = None;
                    } else if self.promotion_move.is_some() {
                        // A promotion choice is pending, don't allow other moves.
                        // You could add feedback to the user here if desired.
                        self.selected_square = None;
                    } else {
                        // This is the first click (source)
                        // Only select if it's our piece
                        if self.board.color_on(clicked_square) == Some(self.board.side_to_move()) {
                            self.selected_square = Some(clicked_square);
                        }
                    }
                }
            }
        }
    }
}

impl std::fmt::Display for Player {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Player::Human => write!(f, "Human"),
            Player::Bot(model) => write!(f, "Bot ({:?})", model),
        }
    }
}

impl std::fmt::Display for BotModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

fn get_piece_char(piece: Piece, color: Color) -> char {
    // Using standard unicode chess characters
    let char = match piece {
        Piece::Pawn => '♙',
        Piece::Knight => '♘',
        Piece::Bishop => '♗',
        Piece::Rook => '♖',
        Piece::Queen => '♕',
        Piece::King => '♔',
    };
    if color == Color::White { char } else { char.to_lowercase().next().unwrap() }
}

impl eframe::App for ChessApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // If it's a bot's turn and the game is not over, make a move.
            if let Player::Bot(model) = self.current_player() {
                if self.board.status() == BoardStatus::Ongoing {
                    // Make a copy of the model to pass to the function
                    let bot_model = *model;
                    self.make_bot_move(bot_model);
                    // Request a repaint to show the bot's move immediately
                    ctx.request_repaint();
                }
            }

            ui.heading("Chessers");
            ui.label(&self.model_status);

            // --- Game Status ---
            ui.horizontal(|ui| {
                ui.label("Turn:");
                ui.label(format!("{:?}", self.board.side_to_move()));
            });

            if self.board.status() != BoardStatus::Ongoing {
                ui.label(format!("Game Over: {:?}", self.board.status()));
            }

            // --- Board Rendering and Input ---
            ui.add_space(10.0);
            self.draw_board_and_handle_input(ui);
            ui.add_space(10.0);

            ui.horizontal(|ui| {
                if ui.button("New Game").clicked() {
                    self.selected_square = None;
                    self.board = Board::default();
                    self.game_history.clear();
                }

                if ui.button("Save Game").clicked() {
                    if let Err(e) = self.save_game_to_pgn() {
                        self.model_status = format!("Error saving game: {}", e);
                    } else {
                        self.model_status = "Game saved to human_games.pgn".to_string();
                    }
                }
            });
        });

        // --- Promotion UI ---
        if let Some((start, end)) = self.promotion_move {
            egui::Window::new("Pawn Promotion")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label("Promote to:");
                    ui.horizontal(|ui| {
                        let pieces = [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight];
                        for piece in pieces {
                            if ui.button(get_piece_char(piece, self.board.side_to_move()).to_string()).clicked() {
                                let chess_move = ChessMove::new(start, end, Some(piece));
                                if self.board.legal(chess_move) {
                                    self.board = self.board.make_move_new(chess_move);
                                    self.game_history.push(chess_move);
                                }
                                // Close the promotion window
                                self.promotion_move = None;
                                break;
                            }
                        }
                    });
                });
        }
    }
}
