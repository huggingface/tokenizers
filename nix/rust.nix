{ pkgs
}:

with pkgs;

let

  pkg = naersk.buildPackage ../tokenizers;

in

  pkg
