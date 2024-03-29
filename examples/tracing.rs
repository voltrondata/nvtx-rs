use tracing::{info, instrument};
use tracing_subscriber::prelude::*;

#[instrument(fields(color = "goldenrod", category = 1, payload = i))]
fn foo(i: u64) {
    for j in 1..=i {
        bar(j);
        std::thread::sleep(std::time::Duration::from_millis(10 * i));
    }
}

#[instrument(fields(color = "plum", category = 1, payload = j))]
fn bar(j: u64) {
    for k in 1..=j {
        baz(k);
        std::thread::sleep(std::time::Duration::from_millis(10 * j));
    }
}

#[instrument(fields(color = "salmon", category = 2, payload = k))]
fn baz(k: u64) {
    std::thread::sleep(std::time::Duration::from_millis(10 * k));
}

fn main() {
    let layer = nvtx::tracing::NvtxLayer::new("nvtx");
    tracing_subscriber::registry().with(layer).init();

    info!(
        message = "At the beginning of the program",
        color = "blue",
        category = 2
    );
    foo(10);
}
