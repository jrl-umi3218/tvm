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
          overrideAttrs.doxygen-awesome-css = { };
          overrideAttrs.tvm =
            { pkgs-final, drv-prev, ... }:
            {
              src = lib.cleanSource ./.;
              cmakeFlags = (drv-prev.cmakeFlags or [ ]) ++ [
                (lib.cmakeBool "BUILD_TESTING" true)
                (lib.cmakeBool "BUILD_DOCUMENTATION" true)
                "-DDOXYGEN_AWESOME_CSS_PATH=${pkgs-final.doxygen-awesome-css}/share/doxygen-awesome-css"
              ];
              buildInputs = (drv-prev.buildInputs or [ ]) ++ [
                pkgs-final.doxygen
                pkgs-final.gbenchmark
                pkgs-final.doxygen-awesome-css
              ];
            };
        };
      }
    );
}
