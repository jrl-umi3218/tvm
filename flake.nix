{
  description = "A framework for writing and solving optimization problem, with an emphasis on robotic control (Beta)";

  inputs.mc-rtc-nix.url = "github:mc-rtc/nixpkgs";

  outputs =
    inputs:
    inputs.mc-rtc-nix.lib.mkFlakoboros inputs (
      { lib, ... }:
      {
        overrideAttrs.tvm = {pkgs-final, ...}: {
          src = lib.cleanSourceWith ./.;
        };
      }
    );
}
