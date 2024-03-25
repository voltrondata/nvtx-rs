use std::{thread, time};

use nvtx_rs::nvtx;

fn main() {
    // we must hold ranges with a proper name
    // _ will not work since drop() is called immediately
    let _x = nvtx::range(
        nvtx::event_attributes_builder()
            .color(nvtx::colors::salmon)
            .message("Start 🦀")
            .build(),
    );
    thread::sleep(time::Duration::from_millis(5));
    for i in 1..=10 {
        {
            let _rng = nvtx::range(
                nvtx::event_attributes_builder()
                    .color(nvtx::colors::cornflowerblue)
                    .message(format!("Iteration Number {}", i))
                    .payload(i)
                    .build(),
            );
            for j in 1..=i {
                {
                    let _r = nvtx::range(
                        nvtx::event_attributes_builder()
                            .color(nvtx::colors::beige)
                            .payload(j)
                            .message("Inner")
                            .build(),
                    );
                    thread::sleep(time::Duration::from_millis(j * 5));
                }
                thread::sleep(time::Duration::from_millis(5));
            }
        }
        thread::sleep(time::Duration::from_millis(10));
    }
}
