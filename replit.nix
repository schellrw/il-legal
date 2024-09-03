{pkgs}: {
  deps = [
    pkgs.openssl
    pkgs.libxcrypt
    pkgs.pkg-config
    pkgs.arrow-cpp
    pkgs.bash
  ];
}
