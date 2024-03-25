use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    cc::Build::new()
        .include("vendor/nvtx/c/include")
        .opt_level(2)
        .file(Path::new("c/src/lib.c"))
        .compile("nvtx");

    let bindings = bindgen::Builder::default()
        .detect_include_paths(true)
        .clang_arg("-I")
        .clang_arg("vendor/nvtx/c/include")
        .header("c/include/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_recursively(false)
        .generate_cstr(true)
        .default_alias_style(bindgen::AliasVariation::TypeAlias)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .wrap_unsafe_ops(true)
        .must_use_type("nvtxRangeId_t")
        .must_use_type("nvtxStringHandle_t")
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
