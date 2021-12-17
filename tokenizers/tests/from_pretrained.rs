#![cfg(feature = "http")]
use tokenizers::{FromPretrainedParameters, Result, Tokenizer};

#[test]
fn test_from_pretrained() -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;
    let encoding = tokenizer.encode("Hey there dear friend!", false)?;
    assert_eq!(
        encoding.get_tokens(),
        &["Hey", "there", "dear", "friend", "!"]
    );
    Ok(())
}

#[test]
fn test_from_pretrained_revision() -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained("anthony/tokenizers-test", None)?;
    let encoding = tokenizer.encode("Hey there dear friend!", false)?;
    assert_eq!(
        encoding.get_tokens(),
        &["hey", "there", "dear", "friend", "!"]
    );

    let tokenizer = Tokenizer::from_pretrained(
        "anthony/tokenizers-test",
        Some(FromPretrainedParameters {
            revision: "gpt-2".to_string(),
            ..Default::default()
        }),
    )?;
    let encoding = tokenizer.encode("Hey there dear friend!", false)?;
    assert_eq!(
        encoding.get_tokens(),
        &["Hey", "Ġthere", "Ġdear", "Ġfriend", "!"]
    );

    Ok(())
}
