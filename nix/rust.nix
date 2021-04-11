{ pkgs
}:

with pkgs;

let

  self = rec {
    tokenizers = naersk.buildPackage {
      src = ../.;
      buildInputs = [ libiconv pkgconfig ];
      release = true;
      cargoBuildOptions = x: x ++ [ "-p" "tokenizers" ];
      cargoTestOptions = x: x ++ [ "-p" "tokenzers" ];
      copyBins = true;
      copyLibs = true;
      copyTarget = false;
    };

    tokenizers-haskell = naersk.buildPackage {
      src = ../.;
      buildInputs = [ libiconv pkgconfig ];
      release = true;
      cargoBuildOptions = x: x ++ [ "-p" "tokenizers-haskell" ];
      cargoTestOptions = x: x ++ [ "-p" "tokenizers-haskell" ];
      copyBins = false;
      copyLibs = true;
      copyTarget = false;
    };
  };

in

  self
