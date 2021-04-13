{ pkgs ? import ./nix/default.nix {}
}:

with pkgs;

let

  shell = mkShell {
    nativeBuildInputs = [ cargo rustc rls libiconv pkgconfig ];
  };

in

  shell
