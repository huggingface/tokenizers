pkgs: _: with pkgs; {
  tokenizersPackages = import ./rust.nix {
    inherit
      stdenv
      pkgs
      ;
  };
}
