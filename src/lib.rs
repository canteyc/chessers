pub mod model;
pub mod simple_bot;

use candle_core::{Device, IndexOp, Result, Tensor};
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

pub fn move_to_index(m: ChessMove) -> (usize, usize) {
    (m.get_source().to_index(), m.get_dest().to_index())
}

/// Maps an index from 0 to 63 back to a potential `ChessMove`.
/// Note: This does not guarantee the move is legal and does not specify the source square.
pub fn index_to_move(index: usize) -> ChessMove {
    let to_square = unsafe { Square::new(index as u8) };
    // The from square is unknown, so we'll use a placeholder.
    // This function is not critical for training, but is updated for consistency.
    ChessMove::new(to_square, to_square, None)
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
        let (_policy_output, value_output) = match model.forward_all(&board_tensor) {
            Ok(t) => t,
            Err(_) => return 0.0,
        };
        // The value is from the perspective of the current player on the board.
        let value = value_output.i((0, 0)).and_then(|t| t.to_scalar::<f32>()).unwrap_or(0.0);

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
    let (policy_logits, _value_logits) = model.forward_all(&initial_board_tensor).ok()?;

    let mut promising_moves = Vec::new();
    for m in MoveGen::new_legal(board) {
        let move_index = m.get_source().to_index() * 64 + m.get_dest().to_index();
        if let Ok(logits_for_move) = policy_logits.i((0, move_index)) {
            if let Ok(logit) = logits_for_move.to_scalar::<f32>() {
                promising_moves.push((m, logit));
            }
        } else {
            // This case might happen if move_index is out of bounds, which would be a bug.
            // Logging this would be a good idea.
            continue;
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

pub fn find_best_move(board: &Board, model: &UNet, device: &Device) -> Option<ChessMove> {
    let board_tensor = board_to_small_tensor(board, device).ok()?;
    let (policy, _value) = model.forward_all(&board_tensor).ok()?;
    let from_logits = policy.i((.., 0)).ok()?.flatten_from(1).ok()?;
    let to_logits = policy.i((.., 1)).ok()?.flatten_from(1).ok()?;
    let mut best_move: Option<ChessMove> = None;
    let mut max_logit = f32::NEG_INFINITY;

    for m in MoveGen::new_legal(board) {
        let (from_index, to_index) = move_to_index(m);
        if let (Ok(from_logit_tensor), Ok(to_logit_tensor)) = (from_logits.i((0, from_index)), to_logits.i((0, to_index))) {
            if let (Ok(from_logit), Ok(to_logit)) = (from_logit_tensor.to_scalar::<f32>(), to_logit_tensor.to_scalar::<f32>()) {
                let mut logit = from_logit + to_logit;
                logit += rand::random::<f32>();
                if logit > max_logit {
                    max_logit = logit;
                    dbg!(max_logit);
                    best_move = Some(m);
                }
            }
        }
    }
    best_move
}
