use chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves, Board,
    ChessMove, MoveGen,
};
use rayon::prelude::*;

fn calculate_score(board: &Board) -> f32 {
    let mut legal_moves = MoveGen::new_legal(board);
    let mut total_score = legal_moves.len() as f32;

    let color = board.side_to_move();
    let opponent_pieces = board.color_combined(!color);
    legal_moves.set_iterator_mask(*opponent_pieces);
    let attacking_moves = legal_moves.len();
    total_score += attacking_moves as f32 * 0.5;

    let my_pieces = board.color_combined(color);
    let my_pawns = board.pieces(chess::Piece::Pawn) & *my_pieces;
    let my_knights = board.pieces(chess::Piece::Knight) & *my_pieces;
    let my_bishops = board.pieces(chess::Piece::Bishop) & *my_pieces;
    let my_rooks = board.pieces(chess::Piece::Rook) & *my_pieces;
    let my_queens = board.pieces(chess::Piece::Queen) & *my_pieces;
    let my_king = board.pieces(chess::Piece::King) & *my_pieces;

    let mut defender_score = 0;
    let all_pieces = board.combined();

    for p in my_pawns {
        defender_score += (get_pawn_attacks(p, color, *my_pieces) & *my_pieces).popcnt();
    }
    for p in my_knights {
        defender_score += (get_knight_moves(p) & *my_pieces).popcnt();
    }
    for p in my_bishops {
        defender_score += (get_bishop_moves(p, *all_pieces) & *my_pieces).popcnt();
    }
    for p in my_rooks {
        defender_score += (get_rook_moves(p, *all_pieces) & *my_pieces).popcnt();
    }
    for p in my_queens {
        let moves = get_bishop_moves(p, *all_pieces) | get_rook_moves(p, *all_pieces);
        defender_score += (moves & *my_pieces).popcnt();
    }
    for p in my_king {
        defender_score += (get_king_moves(p) & *my_pieces).popcnt();
    }
    total_score += defender_score as f32;

    total_score
}

fn lookahead(board: &Board, depth: usize, mut alpha: f32, beta: f32) -> f32 {
    if depth == 0 {
        return calculate_score(board);
    }

    let legal_moves = MoveGen::new_legal(board);
    if legal_moves.len() == 0 {
        return calculate_score(board);
    }

    for m in legal_moves {
        let next_board = board.make_move_new(m);
        let score = -lookahead(&next_board, depth - 1, -beta, -alpha);
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

pub fn find_simple_move(board: &Board) -> Option<ChessMove> {
    let moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
    moves
        .into_par_iter()
        .map(|m| {
            let score = -lookahead(&board.make_move_new(m), 4, f32::MIN, f32::MAX);
            (m, score)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(m, _)| m)
}
