{ pkgs ? import ./nix/default.nix {}
}:

with pkgs;

let

  self = {
    inherit (tokenizersPackages)
      tokenizers
      tokenizers-haskell
      ;
    
    shell = import ./shell.nix {
      inherit pkgs;
    };
  };

in

  self
