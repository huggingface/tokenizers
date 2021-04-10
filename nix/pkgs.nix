pkgs: _: with pkgs; {
  tokenizers = import ./rust.nix {
    inherit
      pkgs
      ;
  };
}
