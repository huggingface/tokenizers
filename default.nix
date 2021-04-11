{ pkgs ? import ./nix/default.nix {}
}:

with pkgs;

let

  self = {
    inherit (tokenizersPackages)
      tokenizers
      ;
    
    shell = import ./shell.nix {
      inherit pkgs;
    };
  };

in

  self
