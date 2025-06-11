use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=schema/");

    // flatc -o . --rust --rust-module-root-file mvf.fbs core.fbs extensions.fbs index.fbs types.fb

    // 1. Run flatc to generate the individual `*_generated.rs` files
    flatc_rust::run(flatc_rust::Args {
        lang: "rust",
        inputs: &[
            Path::new("schema/core.fbs"),
            Path::new("schema/index.fbs"),
            Path::new("schema/types.fbs"),
            Path::new("schema/extensions.fbs"),
            Path::new("schema/mvf.fbs"),
        ],
        out_dir: Path::new("./generated"),
        extra: &["--rust-module-root-file"],
        ..Default::default()
    })
    .expect("flatc code generation failed");
}
