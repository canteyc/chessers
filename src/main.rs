use chess::Game;
use crate::player::{HumanPlayer, Player};
use player::RandomPlayer;
use crate::ui::{ConsoleUI, UI};

mod playground;
mod player;
mod ui;
mod nn;

fn main() {
    let mut game = Game::new();
    let p1 = HumanPlayer {};
    let p2 = RandomPlayer {};
    let gui = ConsoleUI {};
    gui.update(&game.current_position());

    for i in 1..1000 {
        let p1_move = p1.make_move(&game.current_position());
        game.make_move(p1_move);
        if check_game(&mut game, i) { break; };
        let p2_move = p2.make_move(&game.current_position());
        game.make_move(p2_move);
        if check_game(&mut game, i) { break; };
        gui.update(&game.current_position());
    }
}

fn check_game(game: &mut Game, i: i32) -> bool{
    if game.can_declare_draw() {
        game.declare_draw();
    }
    match game.result() {
        Some(_) => {
            println!("finished after {}", i);
            if game.result().is_some() {
                println!("{:?}", game.result().expect(""))
            }
            true
        },
        None => false,
    }
}
