{ pkgs
}:

with pkgs;

let

  pkg = naersk.buildPackage {
    src = ../tokenizers;
    buildInputs = [ libiconv pkgconfig ];
  };

in

  pkg
