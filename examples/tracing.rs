use tracing::{info, instrument};
use tracing_subscriber::prelude::*;

#[instrument(fields(color = "goldenrod", category = "A", payload = i, domain = "Domain1"))]
fn foo(i: u64) {
    for j in 1..=i {
        bar(j);
        std::thread::sleep(std::time::Duration::from_millis(10 * i));
    }
}

#[instrument(fields(color = "plum", category = "B", payload = j, domain = "Domain1"))]
fn bar(j: u64) {
    for k in 1..=j {
        baz(k);
        std::thread::sleep(std::time::Duration::from_millis(10 * j));
    }
}

#[instrument(fields(color = "salmon", category = "A", payload = k, domain = "Domain1"))]
fn baz(k: u64) {
    std::thread::sleep(std::time::Duration::from_millis(10 * k));
}

fn main() {
    tracing_subscriber::registry()
        .with(nvtx::tracing::NvtxLayer::default())
        .init();

    info!(
        message = "At the beginning of the program",
        color = "blue",
        category = "B",
        domain = "Domain2"
    );
    foo(10);
}
