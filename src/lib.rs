pub mod model;
pub mod simple_bot;

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::Module;
use chess::{Board, ChessMove, Color, MoveGen, Square};

use crate::model::UNet;

/// Converts a `chess::Board` to a `candle_core::Tensor`.
pub fn board_to_tensor(board: &Board, device: &Device) -> Result<Tensor> {
    let mut planes = [0.0f32; 13 * 8 * 8];
    let active_color = board.side_to_move();

    // Piece planes
    for i in 0..64 {
        // This is safe because the loop guarantees `i` is always in the range 0..64.
        let sq = unsafe { Square::new(i) };
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
    Tensor::from_slice(&planes, (13, 8, 8), device)
}

pub fn board_to_small_tensor(board: &Board, device: &Device) -> Result<Tensor> {
    let mut planes = [0.0f32; 6 * 8 * 8];
    let perspective = board.side_to_move();

    for i in 0..64 {
        let sq = unsafe { Square::new(i) };
        if let Some(piece) = board.piece_on(sq) {
            let piece_idx = piece.to_index();
            let color = board.color_on(sq).unwrap();

            // Flip board for Black's perspective
            let square_index = if perspective == Color::White {
                i as usize
            } else {
                ((7 - (i / 8)) * 8 + (i % 8)) as usize
            };

            let idx = piece_idx * 64 + square_index;
            planes[idx] = if color == perspective { 1.0 } else { -1.0 };
        }
    }

    Tensor::from_slice(&planes, (6, 8, 8), device)
}

/// Converts a `chess::Board` and a `ChessMove` to a tensor for a move-scoring model.
pub fn board_and_move_to_tensor(board: &Board, m: ChessMove, device: &Device) -> Result<Tensor> {
    let mut planes = [0.0f32; 8 * 8 * 8]; // 8 channels now
    let perspective = board.side_to_move();

    // Channels 0-5: Piece positions (same as board_to_small_tensor)
    for i in 0..64 {
        let sq = unsafe { Square::new(i) };
        if let Some(piece) = board.piece_on(sq) {
            let piece_idx = piece.to_index();
            let color = board.color_on(sq).unwrap();
            let square_index = if perspective == Color::White { i as usize } else { 63 - i as usize };
            let idx = piece_idx * 64 + square_index;
            planes[idx] = if color == perspective { 1.0 } else { -1.0 };
        }
    }

    // Channel 6: "From" square plane
    let from_idx = m.get_source().to_index();
    let from_square_index = if perspective == Color::White { from_idx } else { 63 - from_idx };
    planes[6 * 64 + from_square_index] = 1.0;

    // Channel 7: "To" square plane
    let to_idx = m.get_dest().to_index();
    let to_square_index = if perspective == Color::White { to_idx } else { 63 - to_idx };
    planes[7 * 64 + to_square_index] = 1.0;

    Tensor::from_slice(&planes, (8, 8, 8), device)
}

/// Maps a `ChessMove` to a unique index from 0 to 4095.
pub fn move_to_index(m: ChessMove) -> usize {
    let from = m.get_source().to_index();
    let to = m.get_dest().to_index();
    // Note: This simple mapping doesn't account for promotions.
    // A more advanced mapping would be needed for a full-featured engine.
    from * 64 + to
}

/// Maps an index from 0 to 4095 back to a potential `ChessMove`.
/// Note: This does not guarantee the move is legal.
pub fn index_to_move(index: usize) -> ChessMove {
    let from_sq_idx = (index / 64) as u8;
    let to_sq_idx = (index % 64) as u8;

    // These are safe because the index calculation ensures they are within 0-63.
    let from_square = unsafe { Square::new(from_sq_idx) };
    let to_square = unsafe { Square::new(to_sq_idx) };

    // This doesn't handle promotions. For now, we assume no promotion.
    // A real implementation would need to check the piece and rank.
    ChessMove::new(from_square, to_square, None)
}

/// Recursive minimax search function.
fn minimax_search(
    board: &Board,
    model: &UNet,
    device: &Device,
    depth: usize,
    is_maximizing_player: bool,
    mut alpha: f32,
    mut beta: f32,
) -> f32 {
    // If we've reached the desired depth or the game is over, return the static evaluation.
    if depth == 0 || board.status() != chess::BoardStatus::Ongoing {
        let board_tensor = match board_to_small_tensor(board, device) {
            Ok(t) => t,
            Err(_) => return 0.0, // Return neutral score on error
        };
        let output = match model.forward(&board_tensor) {
            Ok(t) => t,
            Err(_) => return 0.0,
        };
        // The value is from the perspective of the current player on the board.
        let value = output.i((0, 4096)).and_then(|t| t.to_scalar::<f32>()).unwrap_or(0.0);

        // Minimax needs the score relative to the initial player of the search.
        // If we are the maximizing player, and it's our turn on this board, the value is good.
        // If we are the maximizing player, and it's the opponent's turn, a high value for them is bad for us.
        let perspective_multiplier = if is_maximizing_player { 1.0 } else { -1.0 };
        return value * perspective_multiplier;
    }

    let mut best_value = if is_maximizing_player { f32::NEG_INFINITY } else { f32::INFINITY };

    // In a real implementation, we would also use the policy head here to prune the search.
    // For simplicity in this example, we iterate all legal moves.
    for m in MoveGen::new_legal(board) {
        let new_board = board.make_move_new(m);
        let eval = minimax_search(&new_board, model, device, depth - 1, !is_maximizing_player, alpha, beta);

        if is_maximizing_player {
            best_value = best_value.max(eval);
            alpha = alpha.max(eval);
            if beta <= alpha {
                break; // Beta cutoff
            }
        } else {
            best_value = best_value.min(eval);
            beta = beta.min(eval);
            if beta <= alpha {
                break; // Alpha cutoff
            }
        }
    }
    best_value
}

/// Uses a 1-ply search to find the best move.
/// It evaluates the board state after each legal move using the model's value head.
pub fn find_best_move_with_search(board: &Board, model: &UNet, device: &Device) -> Option<ChessMove> {
    const SEARCH_WIDTH: usize = 5; // Only look ahead on the top 5 policy moves.
    const SEARCH_DEPTH: usize = 2; // How many moves to look ahead (e.g., 2 = 1 full move)

    // --- 1. Policy Head Pruning ---
    // First, find the most promising moves according to the policy head.
    let initial_board_tensor = board_to_small_tensor(board, device).ok()?;
    let initial_output = model.forward(&initial_board_tensor).ok()?;
    let policy_logits = initial_output.i((.., ..4096)).ok()?;

    let mut promising_moves = Vec::new();
    for m in MoveGen::new_legal(board) {
        let move_index = move_to_index(m);
        if let Ok(logits_for_move) = policy_logits.i((0, move_index)) {
            if let Ok(logit) = logits_for_move.to_scalar::<f32>() {
                promising_moves.push((m, logit));
            }
        } else {
            // This case might happen if move_index is out of bounds, which would be a bug.
            // Logging this would be a good idea.
        }
    }

    // Sort moves by their policy logit, descending.
    promising_moves.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // --- 2. Minimax Evaluation on Pruned Moves ---
    let mut best_move: Option<ChessMove> = None;
    let mut max_value = f32::NEG_INFINITY;

    for (m, _logit) in promising_moves.iter().take(SEARCH_WIDTH) {
        let new_board = board.make_move_new(*m);
        // We made a move, so now it's the opponent's turn (minimizing player).
        let value = minimax_search(&new_board, model, device, SEARCH_DEPTH - 1, false, f32::NEG_INFINITY, f32::INFINITY);

        if value > max_value {
            max_value = value;
            best_move = Some(*m);
        }
    }

    // If the search yields no move (e.g., all top moves lead to errors),
    // fall back to the best policy move from the initial set.
    best_move.or_else(|| promising_moves.first().map(|(m, _)| *m))
}
