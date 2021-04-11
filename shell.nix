{ pkgs ? import ./nix/default.nix {}
}:

with pkgs;

let

  shell = mkShell {
    nativeBuildInputs = [ cargo libiconv pkgconfig ];
  };

in

  shell
