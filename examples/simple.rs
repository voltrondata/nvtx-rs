use std::{thread, time};

use nvtx_rs::nvtx;

fn main() {
    let _ = nvtx::range_push(
        nvtx::AttributeBuilder::default()
            .color(nvtx::colors::salmon)
            .message("Entire Application")
            .build(),
    );
    thread::sleep(time::Duration::from_millis(50));
    for i in 1..=50 {
        {
            let _ = nvtx::range_push(
                nvtx::AttributeBuilder::default()
                    .color(nvtx::colors::cornflowerblue)
                    .message(format!("Iteration Number {}", i))
                    .payload(i)
                    .build(),
            );
            if i < 40 {
                thread::sleep(time::Duration::from_millis(10 * i));
            } else {
                for j in 1..=i {
                    let start_id = nvtx::range_push(
                        nvtx::AttributeBuilder::default()
                            .color(nvtx::colors::beige)
                            .payload(j)
                            .message("Inner Interation")
                            .build(),
                    );
                    thread::sleep(time::Duration::from_millis(j));
                    let stop_id = nvtx::range_pop();
                    assert!(start_id == stop_id);
                    thread::sleep(time::Duration::from_millis(5));
                }
            }
            if i % 10 == 0 {
                nvtx::mark("At an iteration divisible by 10");
            }
            nvtx::range_pop();
        }
        thread::sleep(time::Duration::from_millis(25));
    }
    nvtx::range_pop();
}
