use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=schema/");

    let out_dir = std::env::var("OUT_DIR").unwrap();

    // flatc -o . --rust --rust-module-root-file mvf.fbs core.fbs extensions.fbs index.fbs types.fb

    let generated_path = Path::join(Path::new(&out_dir), Path::new("fbs/"));

    flatc_rust::run(flatc_rust::Args {
        lang: "rust",
        inputs: &[
            Path::new("schema/core.fbs"),
            Path::new("schema/index.fbs"),
            Path::new("schema/types.fbs"),
            Path::new("schema/extensions.fbs"),
            Path::new("schema/mvf.fbs"),
        ],
        out_dir: generated_path.as_path(),
        extra: &["--rust-module-root-file"],
        ..Default::default()
    })
    .expect("flatc code generation failed");
}
