use chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves, Board,
    ChessMove, MoveGen,
};
use dashmap::DashMap;
use rayon::prelude::*;

#[derive(Clone, Copy)]
enum Flag {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct Entry {
    depth: usize,
    score: f32,
    flag: Flag,
}

type TranspositionTable = DashMap<u64, Entry>;


fn calculate_score(board: &Board) -> f32 {
    let mut legal_moves = MoveGen::new_legal(board);
    let mut total_score = legal_moves.len() as f32;

    let color = board.side_to_move();
    let opponent_pieces = board.color_combined(!color);
    legal_moves.set_iterator_mask(*opponent_pieces);
    let attacking_moves = legal_moves.len();
    total_score += attacking_moves as f32 * 0.5;

    let my_pieces = board.color_combined(color);
    let all_pieces = board.combined();
    let mut defender_score = 0;

    for p in board.pieces(chess::Piece::Pawn) & *my_pieces {
        defender_score += (get_pawn_attacks(p, color, *my_pieces) & *my_pieces).popcnt();
    }
    for p in board.pieces(chess::Piece::Knight) & *my_pieces {
        defender_score += (get_knight_moves(p) & *my_pieces).popcnt();
    }
    for p in board.pieces(chess::Piece::Bishop) & *my_pieces {
        defender_score += (get_bishop_moves(p, *all_pieces) & *my_pieces).popcnt();
    }
    for p in board.pieces(chess::Piece::Rook) & *my_pieces {
        defender_score += (get_rook_moves(p, *all_pieces) & *my_pieces).popcnt();
    }
    for p in board.pieces(chess::Piece::Queen) & *my_pieces {
        let moves = get_bishop_moves(p, *all_pieces) | get_rook_moves(p, *all_pieces);
        defender_score += (moves & *my_pieces).popcnt();
    }
    for p in board.pieces(chess::Piece::King) & *my_pieces {
        defender_score += (get_king_moves(p) & *my_pieces).popcnt();
    }
    total_score += defender_score as f32;

    total_score
}

fn lookahead(
    board: &Board,
    depth: usize,
    mut alpha: f32,
    beta: f32,
    table: &TranspositionTable,
) -> f32 {
    let original_alpha = alpha;
    if let Some(entry) = table.get(&board.get_hash()) {
        if entry.depth >= depth {
            match entry.flag {
                Flag::Exact => return entry.score,
                Flag::Lower => alpha = alpha.max(entry.score),
                Flag::Upper => return entry.score,
            }
            if alpha >= beta {
                return entry.score;
            }
        }
    }

    if depth == 0 {
        return calculate_score(board);
    }

    let legal_moves = MoveGen::new_legal(board);
    if legal_moves.len() == 0 {
        return calculate_score(board);
    }

    let mut captures = Vec::new();
    let mut non_captures = Vec::new();
    let opponent_pieces = board.color_combined(!board.side_to_move());

    for m in legal_moves {
        if (opponent_pieces & chess::BitBoard::from_square(m.get_dest())).popcnt() == 0 {
            non_captures.push(m);
        } else {
            captures.push(m);
        }
    }

    let mut best_score = f32::MIN;
    for m in captures.into_iter().chain(non_captures.into_iter()) {
        let next_board = board.make_move_new(m);
        let score = -lookahead(&next_board, depth - 1, -beta, -alpha, table);
        if score > best_score {
            best_score = score;
        }
        if best_score > alpha {
            alpha = best_score;
        }
        if alpha >= beta {
            break;
        }
    }

    let flag = if best_score <= original_alpha {
        Flag::Upper
    } else if best_score >= beta {
        Flag::Lower
    } else {
        Flag::Exact
    };

    table.insert(
        board.get_hash(),
        Entry {
            depth,
            score: best_score,
            flag,
        },
    );

    best_score
}

pub fn find_simple_move(board: &Board) -> Option<ChessMove> {
    let table = TranspositionTable::new();
    let moves: Vec<ChessMove> = MoveGen::new_legal(board).collect();
    moves
        .into_par_iter()
        .map(|m| {
            let score = -lookahead(
                &board.make_move_new(m),
                4,
                f32::MIN,
                f32::MAX,
                &table,
            );
            (m, score)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(m, _)| m)
}
