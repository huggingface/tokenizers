use std::error::Error;
use std::time::{Duration, Instant};
use tk_encode::models::unigram::Unigram;
use tk_train::{Trainer, UnigramTrainer};

fn argument(index: usize, default: usize) -> Result<usize, Box<dyn Error + Send + Sync>> {
    std::env::args()
        .nth(index)
        .map(|value| value.parse::<usize>().map_err(Into::into))
        .unwrap_or(Ok(default))
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let sentence_count = argument(1, 5_000)?;
    let run_count = argument(2, 3)?;
    if sentence_count == 0 || run_count == 0 {
        return Err("sentence count and run count must be greater than zero".into());
    }

    let templates = [
        "1. The morning sun illuminated the quiet streets.",
        "2. She carefully examined the results of the experiment.",
        "3. Technology continues to transform modern education.",
        "4. The old wooden bridge creaked under his weight.",
        "5. Everyone deserves an opportunity to express their opinions.",
        "6. The researchers collected additional data to verify their hypothesis.",
        "7. A sudden change in temperature affected the experiment.",
        "8. He struggled to explain the underlying mathematical principle.",
        "9. The committee postponed its decision until next Monday.",
        "10. Reading regularly improves vocabulary and comprehension.",
        "11. The software automatically detects duplicate entries.",
        "12. Several unexpected errors appeared during compilation.",
        "13. The children enthusiastically explored the abandoned building.",
        "14. Her explanation was considerably more convincing than his.",
        "15. The machine operates independently of external supervision.",
        "16. We should distinguish between correlation and causation.",
        "17. The solution became increasingly complicated as the problem developed.",
        "18. His observations contradicted the previously established theory.",
        "19. The library contains thousands of historical documents.",
        "20. She eventually discovered the source of the malfunction.",
        "21. Effective communication requires clarity and precision.",
        "22. The company introduced several improvements to its existing products.",
        "23. Mathematics provides a framework for understanding complex phenomena.",
        "24. The application consumes significantly less memory than its predecessor.",
        "25. Nobody anticipated such an unusual outcome.",
        "26. The professor encouraged students to question conventional assumptions.",
        "27. A small modification produced a substantial improvement.",
        "28. The weather remained relatively stable throughout the afternoon.",
        "29. The algorithm efficiently processes large amounts of information.",
        "30. Their disagreement originated from a misunderstanding.",
        "31. The evidence supports a completely different interpretation.",
        "32. He accidentally deleted the original configuration file.",
        "33. The experiment demonstrated an interesting relationship between the variables.",
        "34. Persistence is often more valuable than natural talent.",
        "35. The manuscript contains several grammatical inconsistencies.",
        "36. She approached the situation with remarkable patience.",
        "37. The device automatically adjusts its operating frequency.",
        "38. Every assumption should be examined critically.",
        "39. The construction workers repaired the damaged pavement.",
        "40. His argument relies on an incorrect interpretation of the definition.",
        "41. The database stores information in a structured format.",
        "42. A reasonable approximation is sometimes sufficient.",
        "43. The investigation revealed several previously unknown details.",
        "44. The students collaborated on a particularly challenging assignment.",
        "45. The program encountered an unexpected exception.",
        "46. The distinction between these concepts is surprisingly subtle.",
        "47. The engineer proposed a more efficient implementation.",
        "48. Their conclusions were based on insufficient evidence.",
        "49. The river gradually widened as it approached the coastline.",
        "50. Scientific progress depends on continuous experimentation.",
        "51. The tokenizer recognizes common linguistic patterns.",
        "52. Some programming languages prioritize safety over flexibility.",
        "53. The discussion eventually shifted toward more practical concerns.",
        "54. She carefully compared the two competing explanations.",
        "55. The results were consistent with our initial predictions.",
        "56. A seemingly insignificant detail changed the entire outcome.",
        "57. The application supports multiple file formats.",
        "58. His explanation lacked sufficient mathematical justification.",
        "59. The researchers attempted to reproduce the original experiment.",
        "60. Understanding the context is essential for interpreting information correctly.",
        "61. The new architecture reduces unnecessary computational overhead.",
        "62. Several participants expressed concerns about the proposed changes.",
        "63. The phenomenon occurs under specific environmental conditions.",
        "64. The document was revised to improve readability.",
        "65. An incorrect assumption can invalidate an otherwise reasonable argument.",
        "66. The machine learning model generalized poorly to unfamiliar examples.",
        "67. The experiment requires precise measurements.",
        "68. Their approach was fundamentally different from ours.",
        "69. The system automatically restores its previous configuration.",
        "70. She remained skeptical despite the apparent success.",
        "71. The mathematical expression can be simplified through factorization.",
        "72. The developers identified a critical performance bottleneck.",
        "73. The explanation becomes clearer when we consider a simpler example.",
        "74. Historical records provide valuable insight into ancient civilizations.",
        "75. The proposed method introduces additional computational complexity.",
        "76. He carefully verified every intermediate calculation.",
        "77. The program successfully processed the entire dataset.",
        "78. A reliable benchmark requires consistent testing conditions.",
        "79. The unexpected result prompted further investigation.",
        "80. Their interpretation was technically correct but misleading.",
        "81. The software supports incremental updates.",
        "82. We should avoid introducing unnecessary assumptions.",
        "83. The researchers published their findings in an academic journal.",
        "84. The difference becomes negligible at sufficiently large scales.",
        "85. The compiler generated optimized machine code.",
        "86. She demonstrated exceptional attention to detail.",
        "87. The original implementation suffered from excessive memory consumption.",
        "88. The problem can be reduced to a simpler mathematical form.",
        "89. The experiment produced reproducible results.",
        "90. His conclusion followed logically from the stated premises.",
        "91. The system must remain stable under heavy workloads.",
        "92. The distinction is important from both theoretical and practical perspectives.",
        "93. The students gradually developed a deeper understanding of the subject.",
        "94. The implementation prioritizes correctness and maintainability.",
        "95. Several alternative explanations remain plausible.",
        "96. The data was carefully examined before publication.",
        "97. The proposed optimization significantly reduced execution time.",
        "98. Every successful proof must establish its conclusion rigorously.",
        "99. The relationship between these variables deserves further examination.",
        "100. A simple solution is not necessarily an easy one to discover.",
    ];
    let corpus: Vec<String> = (0..sentence_count)
        .map(|index| {
            format!(
                "{} sample{:08} {}",
                templates[index % templates.len()],
                index,
                templates[(index / templates.len()) % templates.len()]
            )
        })
        .collect();

    let mut trainer = UnigramTrainer::builder()
        .show_progress(false)
        .vocab_size(256)
        .n_sub_iterations(2)
        .seed_size(50_000)
        .build()?;
    trainer.feed(corpus.iter().map(String::as_str), |sentence| {
        Ok(vec![sentence.to_owned()])
    })?;

    let mut total = Duration::ZERO;
    for run in 1..=run_count {
        let mut model = Unigram::default();
        let start = Instant::now();
        trainer.train(&mut model)?;
        let elapsed = start.elapsed();
        total += elapsed;
        println!("run {run}: {:.3}s", elapsed.as_secs_f64());
    }

    let average = total / run_count as u32;
    println!(
        "{sentence_count} unique sentences; average training time: {:.3}s",
        average.as_secs_f64()
    );
    Ok(())
}
