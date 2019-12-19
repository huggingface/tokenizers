use super::Pair;

#[derive(Clone, Default)]
pub struct Word {
    chars: Vec<u32>,
    sizes: Vec<usize>,
}

impl Word {
    pub fn new() -> Self {
        Word {
            chars: vec![],
            sizes: vec![],
        }
    }

    pub fn add(&mut self, c: u32) {
        self.chars.push(c);
        self.sizes.push(1);
    }

    pub fn merge(&mut self, c1: u32, c2: u32, replacement: u32) -> Vec<(Pair, i32)> {
        let mut changes: Vec<(Pair, i32)> = vec![];
        let mut i = 0;
        loop {
            if i >= self.chars.len() {
                break;
            }

            // Found a pair
            if self.chars[i] == c1 && i + 1 < self.chars.len() && self.chars[i + 1] == c2 {
                let first = self.chars[i];
                let second = self.chars[i + 1];

                // If there are other characters before the pair
                if i > 0 {
                    changes.push(((self.chars[i - 1], first), -1));
                    changes.push(((self.chars[i - 1], replacement), 1));
                }

                // Remove in place
                self.chars.insert(i, replacement); // Insert replacement before first char of pair
                self.chars.remove(i + 1); // Remove first char of pair
                self.chars.remove(i + 1); // And then the second

                // Update sizes
                let new_size = self.sizes[i] + self.sizes[i + 1];
                self.sizes[i] = new_size;
                self.sizes.remove(i + 1);

                // If there are other characters after the pair
                if i < self.chars.len() - 1 {
                    changes.push(((second, self.chars[i + 1]), -1));
                    changes.push(((replacement, self.chars[i + 1]), 1));
                }
            }

            i += 1;
        }

        changes
    }

    pub fn get_chars(&self) -> &Vec<u32> {
        &self.chars
    }

    pub fn get_offsets(&self) -> Vec<(usize, usize)> {
        let mut offsets = vec![];
        let mut pos = 0;
        for size in &self.sizes {
            offsets.push((pos, pos + size));
            pos += size;
        }
        offsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge() {
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3}.
        let mut word = Word::new();
        word.add(0); // 'h'
        word.add(1); // 'e'
        word.add(2); // 'l'
        word.add(2); // 'l'
        word.add(3); // 'o'

        // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
        // say that 'll' has the ID of 4 in the updated word-to-id vocab.
        let changes = word.merge(2, 2, 4);

        // So the word should now look like this:
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                4u32, // 'll'
                3u32, // 'o'
            ]
        );

        // The return value `changes` will be used to update the pair counts during
        // training. This merge affects the counts for the pairs
        // ('e', 'l') ~= (1, 2),
        // ('e', 'll') ~= (1, 4),
        // ('l', 'o') ~= (2, 3), and
        // ('ll', 'o') ~= (4, 3).
        // So the changes should reflect that:
        assert_eq!(
            changes,
            &[
                ((1u32, 2u32), -1i32), // count for ('e', 'l') should be decreased by 1.
                ((1u32, 4u32), 1i32),  // count for ('e', 'll') should be increased by 1.
                ((2u32, 3u32), -1i32), // count for ('l', 'o') should be decreased by 1.
                ((4u32, 3u32), 1i32),  // count for ('ll', 'o') should be increased by 1.
            ]
        );
    }
}
