use std::time::Instant;
use std::fs;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: {} <tokenizer.json> <input.txt>", args[0]);
        std::process::exit(1);
    }
    
    let tokenizer_path = &args[1];
    let input_path = &args[2];
    
    // Load tokenizer
    let load_start = Instant::now();
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let load_time = load_start.elapsed();
    
    // Read input file
    let text = fs::read_to_string(input_path)?;
    let num_chars = text.chars().count();
    
    // Benchmark encoding
    let encode_start = Instant::now();
    let encoding = tokenizer.encode(text, false)?;
    let encode_time = encode_start.elapsed();
    
    let num_tokens = encoding.get_ids().len();
    
    // Print results in a parseable format
    println!("load_time_ms:{}", load_time.as_millis());
    println!("encode_time_ms:{}", encode_time.as_millis());
    println!("num_tokens:{}", num_tokens);
    println!("num_chars:{}", num_chars);
    println!("tokens_per_sec:{:.2}", num_tokens as f64 / encode_time.as_secs_f64());
    
    Ok(())
}
