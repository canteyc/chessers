use chess::{Board, Game};
use crate::player::Player;
use crate::players::RandomPlayer;

mod playground;
mod player;
mod players;
mod ui;

fn main() {
    let mut game = Game::new();
    let p1 = RandomPlayer {};
    let p2 = RandomPlayer {};
    for i in 1..10 {
        let p1_move = p1.make_move(&game.current_position());
        game.make_move(p1_move);
        let p2_move = p2.make_move(&game.current_position());
        game.make_move(p2_move);
        println!("{}\n", game.current_position().combined());
    }
}
