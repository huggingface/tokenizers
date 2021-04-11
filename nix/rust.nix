{ pkgs
}:

with pkgs;

let

  self = rec {
    tokenizers = naersk.buildPackage {
      src = ../tokenizers;
      buildInputs = [ libiconv pkgconfig ];
    };

    tokenizers-haskell = naersk.buildPackage {
      src = ../bindings/haskell;
      buildInputs = [ tokenizers ];
    };
  };

in

  self
