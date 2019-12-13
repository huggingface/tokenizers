use super::Pair;

// TODO: Add tests
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
