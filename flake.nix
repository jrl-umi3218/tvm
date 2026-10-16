{
  description = "A framework for writing and solving optimization problem, with an emphasis on robotic control (Beta)";

  inputs.mc-rtc-nix.url = "github:mc-rtc/nixpkgs";
  inputs.ccache-trigger.follows = "mc-rtc-nix/ccache-trigger";

  outputs =
    inputs:
    inputs.mc-rtc-nix.lib.mkMcRtcModule inputs (
      { lib, ... }:
      {
        mc-rtc-nix.overlays.ccache = inputs.ccache-trigger.value;

        flakoboros = {
          packages = {
            mpark-variant =
              {
                stdenv,
                lib,
                fetchFromGitHub,
                cmake,
              }:

              stdenv.mkDerivation {
                pname = "mpark-variant";
                version = "v1.4.0";

                src = fetchFromGitHub {
                  owner = "mpark";
                  repo = "variant";
                  tag = "v1.4.0";
                  hash = "sha256-RAIJTLFtvQFFyTapIwCHWT3IlyXMe8cEWdjRfHFp6D8=";
                };

                nativeBuildInputs = [
                  cmake
                ];

                doCheck = true;

                meta = with lib; {
                  description = "C++17 `std::variant` for C++11/14/17";
                  homepage = "https://github.com/mpark/variant";
                  license = licenses.bsl11;
                  platforms = platforms.all;
                };
              };
          };
          overrideAttrs.tvm =
            { pkgs-final, drv-prev, ... }:
            {
              src = lib.cleanSource ./.;
              cmakeFlages = [ (lib.cmakeBool "BUILD_TESTING" true) ];
              buildInputs = (drv-prev.buildInputs or [ ]) ++ [
                pkgs-final.gbenchmark
                pkgs-final.mpark-variant
              ];
            };
        };
      }
    );
}
