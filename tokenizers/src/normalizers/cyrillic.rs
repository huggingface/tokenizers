use crate::tokenizer::{NormalizedString, Normalizer, Result};
use serde::{Deserialize, Serialize};
use regex::Regex;


#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CyrillicNormalizer;

impl CyrillicNormalizer {
    pub fn new() -> Self {
        Self {}
    }
}

fn cyrillic_to_latin(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        let latin: String = match ch {
            'А' => "A".to_string(), 'а' => "a".to_string(),
            'Б' => "B".to_string(), 'б' => "b".to_string(),
            'В' => "V".to_string(), 'в' => "v".to_string(),
            'Г' => "G".to_string(), 'г' => "g".to_string(),
            'Д' => "D".to_string(), 'д' => "d".to_string(),
            'Ђ' => "Đ".to_string(), 'ђ' => "đ".to_string(),
            'Е' => "E".to_string(), 'е' => "e".to_string(),
            'Ж' => "Ž".to_string(), 'ж' => "ž".to_string(),
            'З' => "Z".to_string(), 'з' => "z".to_string(),
            'И' => "I".to_string(), 'и' => "i".to_string(),
            'Ј' => "J".to_string(), 'ј' => "j".to_string(),
            'К' => "K".to_string(), 'к' => "k".to_string(),
            'Л' => "L".to_string(), 'л' => "l".to_string(),
            'Љ' => "Lj".to_string(), 'љ' => "lj".to_string(),
            'М' => "M".to_string(), 'м' => "m".to_string(),
            'Н' => "N".to_string(), 'н' => "n".to_string(),
            'Њ' => "Nj".to_string(), 'њ' => "nj".to_string(),
            'О' => "O".to_string(), 'о' => "o".to_string(),
            'П' => "P".to_string(), 'п' => "p".to_string(),
            'Р' => "R".to_string(), 'р' => "r".to_string(),
            'С' => "S".to_string(), 'с' => "s".to_string(),
            'Т' => "T".to_string(), 'т' => "t".to_string(),
            'Ћ' => "Ć".to_string(), 'ћ' => "ć".to_string(),
            'У' => "U".to_string(), 'у' => "u".to_string(),
            'Ф' => "F".to_string(), 'ф' => "f".to_string(),
            'Х' => "H".to_string(), 'х' => "h".to_string(),
            'Ц' => "C".to_string(), 'ц' => "c".to_string(),
            'Ч' => "Č".to_string(), 'ч' => "č".to_string(),
            'Џ' => "Dž".to_string(), 'џ' => "dž".to_string(),
            'Ш' => "Š".to_string(), 'ш' => "š".to_string(),

            'Й' => "Y".to_string(), 'й' => "y".to_string(),
            'Ъ' => "Ǎ".to_string(), 'ъ' => "ǎ".to_string(),
            'Ь' => "".to_string(), 'ь' => "".to_string(),
            'Ю' => "Yu".to_string(), 'ю' => "yu".to_string(),
            'Я' => "Ya".to_string(), 'я' => "ya".to_string(),
            'Щ' => "Šy".to_string(), 'щ' => "šy".to_string(),
            _ => ch.to_string(),
        };
        out.push_str(&latin);
    }
    out
}

pub fn latin_to_cyrillic(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        let mapped = match (ch, chars.peek()) {
            // Serbian digraphs
            ('n', Some('j')) => { chars.next(); "њ".to_string() }
            ('N', Some('j')) => { chars.next(); "Њ".to_string() }
            ('l', Some('j')) => { chars.next(); "љ".to_string() }
            ('L', Some('j')) => { chars.next(); "Љ".to_string() }
            ('d', Some('ž')) => { chars.next(); "џ".to_string() }
            ('D', Some('ž')) => { chars.next(); "Џ".to_string() }

            ('Y', Some('u')) => { chars.next(); "Ю".to_string() }
            ('y', Some('u')) => { chars.next(); "ю".to_string() }
            ('Y', Some('a')) => { chars.next(); "Я".to_string() }
            ('y', Some('a')) => { chars.next(); "я".to_string() }
            ('Š', Some('y')) => { chars.next(); "Щ".to_string() }
            ('š', Some('y')) => { chars.next(); "щ".to_string() }

            // Serbian single letters
            ('a', _) => "а".to_string(), ('A', _) => "А".to_string(),
            ('b', _) => "б".to_string(), ('B', _) => "Б".to_string(),
            ('v', _) => "в".to_string(), ('V', _) => "В".to_string(),
            ('g', _) => "г".to_string(), ('G', _) => "Г".to_string(),
            ('d', _) => "д".to_string(), ('D', _) => "Д".to_string(),
            ('đ', _) => "ђ".to_string(), ('Đ', _) => "Ђ".to_string(),
            ('e', _) => "е".to_string(), ('E', _) => "Е".to_string(),
            ('ž', _) => "ж".to_string(), ('Ž', _) => "Ж".to_string(),
            ('z', _) => "з".to_string(), ('Z', _) => "З".to_string(),
            ('i', _) => "и".to_string(), ('I', _) => "И".to_string(),
            ('j', _) => "ј".to_string(), ('J', _) => "Ј".to_string(),
            ('k', _) => "к".to_string(), ('K', _) => "К".to_string(),
            ('l', _) => "л".to_string(), ('L', _) => "Л".to_string(),
            ('m', _) => "м".to_string(), ('M', _) => "М".to_string(),
            ('n', _) => "н".to_string(), ('N', _) => "Н".to_string(),
            ('o', _) => "о".to_string(), ('O', _) => "О".to_string(),
            ('p', _) => "п".to_string(), ('P', _) => "П".to_string(),
            ('r', _) => "р".to_string(), ('R', _) => "Р".to_string(),
            ('s', _) => "с".to_string(), ('S', _) => "С".to_string(),
            ('t', _) => "т".to_string(), ('T', _) => "Т".to_string(),
            ('ć', _) => "ћ".to_string(), ('Ć', _) => "Ћ".to_string(),
            ('u', _) => "у".to_string(), ('U', _) => "У".to_string(),
            ('f', _) => "ф".to_string(), ('F', _) => "Ф".to_string(),
            ('h', _) => "х".to_string(), ('H', _) => "Х".to_string(),
            ('c', _) => "ц".to_string(), ('C', _) => "Ц".to_string(),
            ('č', _) => "ч".to_string(), ('Č', _) => "Ч".to_string(),
            ('š', _) => "ш".to_string(), ('Š', _) => "Ш".to_string(),

            ('Y', _) => "Й".to_string(), ('y', _) => "й".to_string(),
            ('Ǎ', _) => "Ъ".to_string(), ('ǎ', _) => "ъ".to_string(),
            // fallback: keep original
            _ => ch.to_string(),
        };
        out.push_str(&mapped);
    }
    out
}



impl Normalizer for CyrillicNormalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        let re = Regex::new(r"\s?\p{Cyrillic}[^\p{Latin}]*").unwrap();
        let text = normalized.get().to_string();
        let new_text = re.replace_all(&text, |caps: &regex::Captures| {
            let latin = cyrillic_to_latin(&caps[0]);
            format!("<cyr>{}</cyr>", &latin)
        }).to_string();
        let cleanup = Regex::new(r"(\s+)</cyr>").unwrap();
        let clean_text = cleanup.replace_all(&new_text, |caps: &regex::Captures| {
            format!("</cyr>{}", &caps[1])
        }).to_string();
        *normalized = NormalizedString::from(clean_text);
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::NormalizedString;

    #[test]
    fn test_cyrillic_normalizer() {
        let original = "1. Hej жабо, Поздрављам те Я, dobri svete, iz године 2029-те.";
        let expected = "1. Hej<cyr> žabo, Pozdravljam te Ya,</cyr> dobri svete, iz<cyr> godine 2029-te.</cyr>";
        let mut ns = NormalizedString::from(original);
        let normalizer = CyrillicNormalizer::new();
        normalizer.normalize(&mut ns).unwrap();
        assert_eq!(ns.get(), expected);
    }
}
