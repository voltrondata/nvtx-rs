use std::{thread, time};

use nvtx_rs::nvtx;

fn main() {
    // we must hold ranges with a proper name
    // _ will not work since drop() is called immediately
    let app = nvtx::range_start(
        nvtx::AttributeBuilder::default()
            .color(nvtx::colors::salmon)
            .message("Start 🦀")
            .build(),
    );
    thread::sleep(time::Duration::from_millis(5));
    for i in 10..=20 {
        {
            let iter = nvtx::range_start(
                nvtx::AttributeBuilder::default()
                    .color(nvtx::colors::cornflowerblue)
                    .message(format!("Iteration Number {}", i))
                    .payload(i)
                    .build(),
            );
            for j in 1..=i {
                {
                    let inner = nvtx::range_start(
                        nvtx::AttributeBuilder::default()
                            .color(nvtx::colors::beige)
                            .payload(j)
                            .message("Inner")
                            .build(),
                    );
                    thread::sleep(time::Duration::from_millis(10));
                    inner.end();
                    if j == i / 2 {
                        // we clone here since we conditionally end within an inner-loop iteration
                        iter.clone().end();
                    }
                }
                thread::sleep(time::Duration::from_millis(5));
            }
        }
        thread::sleep(time::Duration::from_millis(10));
        if i == 15 {
            // we clone here since we conditionally end within an inner-loop iteration
            app.clone().end();
        }
    }
}
