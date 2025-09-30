mod mlp_model;

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use chess::{Board, BoardStatus, ChessMove, Color, File, MoveGen, Piece, Rank, Square};
use eframe::egui::{self, Color32, Rect, Sense, Vec2};
use mlp_model::Mlp;
use rand::seq::IteratorRandom;

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
    Mlp,
}

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
struct ChessApp {
    board: Board,
    selected_square: Option<Square>,
    white_player: Player,
    black_player: Player,
    mlp_model: Mlp,
}

impl ChessApp {
    fn new() -> Self {
        // Create a VarBuilder for model initialization
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let mlp_model = Mlp::new(vb).expect("Failed to create MLP model");

        Self {
            board: Board::default(),
            selected_square: None,
            white_player: Player::Human, // Default to Human vs Bot
            black_player: Player::Bot(BotModel::Mlp),
            mlp_model,
        }
    }

    /// Dispatches to the correct bot implementation.
    fn make_bot_move(&mut self, model: BotModel) {
        let best_move = match model {
            BotModel::Random => self.find_random_move(),
            BotModel::Mlp => self.find_mlp_move(),
        };

        if let Some(chess_move) = best_move {
            self.board = self.board.make_move_new(chess_move);
        }
    }

    /// Bot implementation: picks a random legal move.
    fn find_random_move(&self) -> Option<ChessMove> {
        let moves = MoveGen::new_legal(&self.board);
        moves.choose(&mut rand::thread_rng())
    }

    /// Bot implementation: uses a (placeholder) MLP to pick a move.
    fn find_mlp_move(&self) -> Option<ChessMove> {
        // 1. Convert board to tensor
        let board_tensor = board_to_tensor(&self.board, &Device::Cpu).ok()?;

        // 2. --- FORWARD PASS ---
        let logits = self.mlp_model.forward(&board_tensor).ok()?;

        // 3. Find the best legal move according to the logits
        let mut best_move: Option<ChessMove> = None;
        let mut max_logit = f32::NEG_INFINITY;

        let legal_moves = MoveGen::new_legal(&self.board);
        for m in legal_moves {
            let move_index = move_to_index(m);
            let move_logit = logits.get(move_index).unwrap().to_scalar::<f32>().unwrap();

            if move_logit > max_logit {
                max_logit = move_logit;
                best_move = Some(m);
            }
        }

        println!("MLP Bot chose move: {:?} with score: {}", best_move, max_logit);
        best_move
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
                        // Note: We don't handle promotions here yet.
                        let chess_move = ChessMove::new(start_square, clicked_square, None);

                        // Check if the move is legal and make it
                        if self.board.legal(chess_move) {
                            self.board = self.board.make_move_new(chess_move);
                        }

                        // Deselect after attempting a move
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

/// Converts a `chess::Board` to a `candle_core::Tensor`.
fn board_to_tensor(board: &Board, device: &Device) -> Result<Tensor> {
    let mut planes = [0.0f32; 13 * 8 * 8];
    let active_color = board.side_to_move();

    // Piece planes
    for i in 0..64 {
        // This is safe because the loop guarantees `i` is always in the range 0..64.
        let sq = unsafe {
            Square::new(i)
        };
        if let Some(piece) = board.piece_on(sq) {
            let color = board.color_on(sq).unwrap();
            let piece_idx = piece.to_index();
            let plane_idx = if color == Color::White { 0 } else { 6 };
            let idx = (plane_idx + piece_idx) * 64 + (i as usize);
            planes[idx] = 1.0;
        }
    }

    // Active color plane
    if active_color == Color::White {
        for i in 0..64 {
            planes[12 * 64 + i] = 1.0;
        }
    }

    // Create a (13, 8, 8) tensor and then flatten it
    Tensor::from_slice(&planes, (13, 8, 8), device)?.flatten_all()
}

/// Maps a `ChessMove` to a unique index from 0 to 4095.
fn move_to_index(m: ChessMove) -> usize {
    let from = m.get_source().to_index();
    let to = m.get_dest().to_index();
    // Note: This simple mapping doesn't account for promotions.
    // A more advanced mapping would be needed for a full-featured engine.
    from * 64 + to
}

impl eframe::App for ChessApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // --- Bot Move Logic ---
            // Check whose turn it is and if they are a bot.
            let current_player = match self.board.side_to_move() {
                Color::White => &self.white_player,
                Color::Black => &self.black_player,
            };

            // If it's a bot's turn and the game is not over, make a move.
            if let Player::Bot(model) = current_player {
                if self.board.status() == BoardStatus::Ongoing {
                    // Make a copy of the model to pass to the function
                    let bot_model = *model;
                    self.make_bot_move(bot_model);
                    // Request a repaint to show the bot's move immediately
                    ctx.request_repaint();
                }
            }

            ui.heading("Chessers");

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

            // Example button to reset the game
            if ui.button("New Game").clicked() {
                self.selected_square = None;
                // Note: This doesn't reset the model's weights.
                self.board = Board::default();
            }
        });
    }
}
