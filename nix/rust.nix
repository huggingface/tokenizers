{ pkgs
}:

with pkgs;

let

  self = rec {
    tokenizers = naersk.buildPackage {
      src = ../.;
      buildInputs = [ libiconv pkgconfig ];
    };
  };

in

  self
