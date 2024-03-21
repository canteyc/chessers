use candle_nn::{VarMap};
use chess::Game;
use crate::player::{Player};
use crate::arena::{Arena, check_game};
use crate::nn::ChessNet;
use crate::ui::{ConsoleUI, UI};

mod playground;
mod player;
mod ui;
mod nn;
mod arena;

fn main() {
    // play();
    let start = chrono::Utc::now();
    let mut arena = Arena::new();
    arena.train();
    println!("Spent: {}s", (chrono::Utc::now() - start).num_seconds());
}

fn play() {
    let mut game = Game::new();
    let p1 = ChessNet::new(VarMap::new());
    let p2 = ChessNet::new(VarMap::new());
    let gui = ConsoleUI {};
    gui.update(&game.current_position());

    for _ in 1..1000 {
        let p1_move = p1.make_move(&game.current_position());
        game.make_move(p1_move);
        if check_game(&mut game) {
            gui.update(&game.current_position());
            break;
        };
        let p2_move = p2.make_move(&game.current_position());
        game.make_move(p2_move);
        if check_game(&mut game) {
            gui.update(&game.current_position());
            break;
        };
        gui.update(&game.current_position());
    }
}

