pub mod model;

use candle_core::{Device, Result, Tensor};
use chess::{Board, ChessMove, Color, Square};

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