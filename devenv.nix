{ pkgs, ... }: {
  languages.rust = {
    enable = true;
    channel = "stable";
    components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" ];
  };

  packages = with pkgs; [ clang-tools_19 cargo-expand lldb flatbuffers ];

  dotenv.enable = true;

  scripts = {
    sanitize = {
      exec = ''
        RUSTFLAGS="-Z sanitizer=thread" cargo +nightly run --target x86_64-unknown-linux-gnu
      '';
    };
  };
}
