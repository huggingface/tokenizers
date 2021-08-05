class Libtokenizers < Formula
  desc "libtokenizers for haskell-bindings"
  homepage "https://github.com/hasktorch/tokenizers"
  url "https://github.com/hasktorch/tokenizers/releases/download/libtokenizers-v0.1/libtokenizers-macos.zip"
  sha256 "a87ec816bdfe8acebca077d1ac51823ed7efb1d309047dbd98345c26d47accea"
  license "Apache-2.0"

  bottle :unneeded

  def install
    system "bash", "-c", "cp -a lib/* #{lib}"
  end
end
