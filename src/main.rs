use clap::Parser;

mod playground;
mod player;
mod ui;
mod nn;
mod arena;
mod cli;

fn main() {
    let start = chrono::Utc::now();
    let args = cli::Cli::parse();
    args.run();
    println!("Spent: {}s", (chrono::Utc::now() - start).num_seconds());
}

