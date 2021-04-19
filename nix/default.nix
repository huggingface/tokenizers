{}:

let

  sources = import ./sources.nix { inherit pkgs; };
  nixpkgsSrc = sources.nixpkgs-unstable;

  overlays =
    [ (pkgs: _: with pkgs; {
        naersk = callPackage sources.naersk {};
      })
      (import ./pkgs.nix)
    ];

  pkgs = import nixpkgsSrc {
    inherit overlays;
  };
  
in pkgs
