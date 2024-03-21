use std::env;
use std::path::{Path, PathBuf};

fn main() {
    cc::Build::new()
        .include("vendor/nvtx/c/include")
        .opt_level(2)
        .file(Path::new("c/src/lib.c"))
        .compile("nvtx");

    let bindings = bindgen::Builder::default()
        .clang_arg("-I")
        .clang_arg("vendor/nvtx/c/include")
        .header("c/include/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
