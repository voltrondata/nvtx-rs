use tracing::{info, info_span, instrument};
use tracing_subscriber::prelude::*;

#[instrument(target = "Domain1", fields(color = "goldenrod", category = "A", payload = i))]
fn foo(i: u64) {
    for j in 1..=i {
        bar(j);
        std::thread::sleep(std::time::Duration::from_millis(10 * i));
    }
}

#[instrument(target = "Domain1", fields(color = "plum", category = "B", payload = j))]
fn bar(j: u64) {
    for k in 1..=j {
        baz(k);
        std::thread::sleep(std::time::Duration::from_millis(10 * j));
    }
}

#[instrument(target = "Domain1", fields(color = "salmon", category = "A", payload = k))]
fn baz(k: u64) {
    std::thread::sleep(std::time::Duration::from_millis(10 * k));
}

fn main() {
    tracing_subscriber::registry()
        .with(nvtx::tracing::NvtxLayer::default())
        .init();

    info!(
        name: "At the beginning of the program",
        target: "Domain2",
        color = "blue",
        category = "B",
    );

    let _r = info_span!(
        target: "Domain2",
        "At the beginning of the program",
        color = "blue",
        category = "B",
    );
    foo(10);
}
