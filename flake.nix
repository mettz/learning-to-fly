{
  description = "Crazyflie firmware development environment";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        formatter = pkgs.alejandra;

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            boost
            hdf5
            openblas
            abseil-cpp
            protobuf
          ];
          nativeBuildInputs = with pkgs; [
            gcc
            gnumake
            cmake
            ninja
            gcc-arm-embedded
            cfclient
            (python3.withPackages (
              p: with p; [
                cflib
              ]
            ))
          ];
        };
      }
    );
}
