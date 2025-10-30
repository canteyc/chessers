use chess::{Board, ChessMove, MoveGen};

fn calculate_score(board: &Board) -> f32 {
    let mut legal_moves = MoveGen::new_legal(board);
    let mut total_score = legal_moves.len() as f32;
    
    let color = board.side_to_move();
    let opponent_pieces = board.color_combined(!color);
    legal_moves.set_iterator_mask(*opponent_pieces);
    total_score += legal_moves.len() as f32;

    total_score
}

pub fn find_simple_move(board: &Board) -> Option<ChessMove> {
    MoveGen::new_legal(board).max_by(|a, b| {
        let next_board = board.make_move_new(*a);
        let bot_score = if next_board.checkers().popcnt() > 0 {
            if next_board.status() == chess::BoardStatus::Checkmate { f32::INFINITY } else { 0.0 }
        } else {
            calculate_score(&next_board.null_move().unwrap())
        };
        let opponent_score = calculate_score(&next_board);
        let ratio_a = bot_score / opponent_score;
        
        let next_board = board.make_move_new(*b);
        let bot_score = if next_board.checkers().popcnt() > 0 {
            if next_board.status() == chess::BoardStatus::Checkmate { f32::INFINITY } else { 0.0 }
        } else {
            calculate_score(&next_board.null_move().unwrap())
        };
        let opponent_score = calculate_score(&next_board);
        let ratio_b = bot_score / opponent_score;

        println!("Move A: {:?}, Ratio: {}", a.to_string(), ratio_a);
        println!("Move B: {:?}, Ratio: {}", b.to_string(), ratio_b);
        ratio_a.total_cmp(&ratio_b)
    })
}
