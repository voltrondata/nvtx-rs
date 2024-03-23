use std::{thread, time};

use nvtx_rs::nvtx;

fn main() {
    let mut domain = nvtx::Domain::new("Domain");
    let cat_handles = domain.register_categories(["A", "B"]);
    let str_handles = domain.register_strings(["alpha", "beta", "gamma"]);
    let [a, b] = domain.get_registered_categories(cat_handles).map(|h|h.unwrap());
    let [alpha, beta, gamma] = domain.get_registered_strings(str_handles).map(|h|h.unwrap());

    let _r = domain.range("Duration");

    let r1 = domain.range_start(
        nvtx::AttributeBuilder::default()
            .category(a)
            .color(nvtx::colors::olive)
            .message(alpha)
            .build(),
    );

    thread::sleep(time::Duration::from_millis(10));
    let r2 = domain.range_start(
        nvtx::AttributeBuilder::default()
            .category(a)
            .color(nvtx::colors::olive)
            .message(beta)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    let r3 = domain.range_start(
        nvtx::AttributeBuilder::default()
            .category(a)
            .color(nvtx::colors::olive)
            .message(gamma)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    r1.end();
    thread::sleep(time::Duration::from_millis(10));
    r2.end();
    thread::sleep(time::Duration::from_millis(10));
    r3.end();
    let p1 = domain.range_push(
        nvtx::AttributeBuilder::default()
            .category(b)
            .color(nvtx::colors::orangered)
            .message(alpha)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    let p2 = domain.range_push(
        nvtx::AttributeBuilder::default()
            .category(b)
            .color(nvtx::colors::orangered)
            .message(beta)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    let p3 = domain.range_push(
        nvtx::AttributeBuilder::default()
            .category(b)
            .color(nvtx::colors::orangered)
            .message(gamma)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    p3.pop();
    thread::sleep(time::Duration::from_millis(10));
    p2.pop();
    thread::sleep(time::Duration::from_millis(10));
    p1.pop();
}
