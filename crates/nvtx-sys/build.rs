use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    let mut lib_builder = cc::Build::new();
    let mut builder = bindgen::Builder::default();

    lib_builder
        .include("vendor/nvtx/c/include")
        .include("c/include")
        .opt_level(2)
        .file(Path::new("c/src/lib.c"));

    builder = builder
        .detect_include_paths(true)
        .clang_arg("-I")
        .clang_arg("vendor/nvtx/c/include")
        .clang_arg("-I")
        .clang_arg("c/include")
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
        .allowlist_type("nvtx.*")
        .allowlist_var("nvtx.*")
        .allowlist_var("NVTX.*")
        .allowlist_function("nvtx.*")
        .allowlist_type("wchar_t")
        .allowlist_type("CU.*")
        .allowlist_type("cuda.*_t")
        .blocklist_type(".*fntype.*")
        .blocklist_type("__.*");

    if cfg!(feature = "cuda") {
        builder = builder.clang_arg("-DENABLE_CUDA");
        lib_builder.define("ENABLE_CUDA", None);
    }
    if cfg!(feature = "cuda_runtime") {
        builder = builder.clang_arg("-DENABLE_CUDART");
        lib_builder.define("ENABLE_CUDART", None);
    }

    lib_builder.compile("nvtx");
    let bindings = builder.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
