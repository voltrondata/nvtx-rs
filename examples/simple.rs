use std::{thread, time};

use nvtx_rs::nvtx;

fn main() {
    let _ = nvtx::Range::new("Entire Application");
    thread::sleep(time::Duration::from_millis(100));
    for i in 1..=50 {
        let rng = nvtx::range_start(
            nvtx::AttributeBuilder::default()
                .color(nvtx::colors::cornflowerblue)
                .message(format!("Iter {}", i))
                .build(),
        );
        println!("Iteration {}", i);
        if i < 40 {
            thread::sleep(time::Duration::from_millis(10 * i));
        } else {
            for j in 1..=i {
                let start_id = nvtx::range_push(nvtx::AttributeBuilder::default().color(nvtx::colors::beige).payload(j).message("Inner Iter").build());
                thread::sleep(time::Duration::from_millis(10));
                let stop_id = nvtx::range_pop();
                assert!(start_id == stop_id);
                thread::sleep(time::Duration::from_millis(5));
            }
        }
        if i % 10 == 0 {
            nvtx::mark("At an iteration divisible by 10");
            println!("At an iteration divisible by 10");
        }
        nvtx::range_end(rng);
        thread::sleep(time::Duration::from_millis(25));
    }
    println!("Done");
}
