{ pkgs, ... }:
{

  languages.rust = {
    enable = true;
    channel = "stable";
    components = [
      "rustc"
      "cargo"
      "clippy"
      "rustfmt"
      "rust-analyzer"
      "llvm-tools-preview"
    ];
  };

  packages = with pkgs; [
    clang-tools_19
    cargo-expand
    lldb
    flatbuffers
    cargo-llvm-cov
    cargo-nextest
    git-cliff
  ];

  dotenv.enable = true;

  scripts = {
    sanitize = {
      exec = ''
        RUSTFLAGS="-Z sanitizer=thread" cargo +nightly run --target x86_64-unknown-linux-gnu
      '';
    };

    run_tests = {
      exec = ''
        cargo llvm-cov nextest --html --ignore-filename-regex 'generated'
      '';
    };
  };
}
