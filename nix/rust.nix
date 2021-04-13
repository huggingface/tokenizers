{ stdenv
, pkgs
}:

with pkgs;

let

  patchLibs =
    if stdenv.isDarwin
    then ''
      install_name_tool -id $out/lib/libtokenizers_haskell.dylib $out/lib/libtokenizers_haskell.dylib
    ''
    else ''
      patchelf --set-rpath "$out/lib" $out/lib/libtokenizers_haskell.so
    '';

  self = {
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
      overrideMain = x: x // { postInstall = patchLibs; };
    };
  };

in

  self
