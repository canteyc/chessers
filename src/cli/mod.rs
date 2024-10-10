use chess::Game;
use clap::{Args, Parser, Subcommand};
use crate::arena::{Arena, check_game};
use crate::nn::ChessNet;
use crate::player::{Player, HumanPlayer, RandomPlayer};
use crate::ui::{UI, ConsoleUI};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Play a single game of chess
    Play (PlayArgs),

    /// Run a genetic optimization to train a ChessBot
    Train (TrainArgs),
}

#[derive(Args)]
struct PlayArgs {
    /// Set white player. Pass a safetensors file to use a ChessBot
    #[arg(short, long, default_value = "human")]
    white: Option<String>,

    /// Set black player. Pass a safetensors file to use a ChessBot
    #[arg(short, long, default_value = "human")]
    black: Option<String>
}

#[derive(Args)]
struct TrainArgs {
    /// Number of bots in each generation (increases runtime by n^2)
    #[arg(short, long, default_value_t = 2)]
    population: usize,

    /// Number of generations to run (increases runtime by n)
    #[arg(short, long, default_value_t = 2)]
    generations: i32,
}

impl Cli {
    pub fn run(&self) {
        match &self.command {
            Commands::Play(args) => {
                let white = create_player(args.white.as_ref().expect("How did white get unset?").as_str());
                let black = create_player(args.black.as_ref().expect("How did black get unset?").as_str());
                play_game(white, black);
            },
            Commands::Train(args) => {
                let mut arena = Arena::new(args.population, args.generations);
                arena.train();
            },
        }
    }
}


fn create_player(source: &str) -> Box<dyn Player> {
    match source { 
            "human" => Box::new(HumanPlayer {}),
            file if file.contains("safetensors") => Box::new(ChessNet::from_file(file)),
            _ => Box::new(RandomPlayer {})
        }
}


fn play_game(p1: Box<dyn Player>, p2: Box<dyn Player>) {
    let mut game = Game::new();
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
