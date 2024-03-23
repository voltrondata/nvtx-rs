use std::{thread, time};

use nvtx_rs::nvtx;

fn main() {
    let mut domain = nvtx::Domain::new("Domain");
    let ha = domain.register_category("A");
    let hb = domain.register_category("B");
    let ra = domain.register_string("alpha");
    let rb = domain.register_string("beta");
    let rc = domain.register_string("gamma");
    let cat_a = domain.get_registered_category(ha).unwrap();
    let cat_b = domain.get_registered_category(hb).unwrap();
    let string_alpha = domain.get_registered_string(ra).unwrap();
    let string_beta = domain.get_registered_string(rb).unwrap();
    let string_gamma = domain.get_registered_string(rc).unwrap();

    let r1 = domain.range_start(
        nvtx::AttributeBuilder::default()
            .category(cat_a)
            .color(nvtx::colors::olive)
            .message(string_alpha)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    let r2 = domain.range_start(
        nvtx::AttributeBuilder::default()
            .category(cat_a)
            .color(nvtx::colors::olive)
            .message(string_beta)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    let r3 = domain.range_start(
        nvtx::AttributeBuilder::default()
            .category(cat_a)
            .color(nvtx::colors::olive)
            .message(string_gamma)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    domain.range_end(r1);
    thread::sleep(time::Duration::from_millis(10));
    domain.range_end(r2);
    thread::sleep(time::Duration::from_millis(10));
    domain.range_end(r3);
    domain.range_push(
        nvtx::AttributeBuilder::default()
            .category(cat_b)
            .color(nvtx::colors::orangered)
            .message(string_alpha)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    domain.range_push(
        nvtx::AttributeBuilder::default()
            .category(cat_b)
            .color(nvtx::colors::orangered)
            .message(string_beta)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    domain.range_push(
        nvtx::AttributeBuilder::default()
            .category(cat_b)
            .color(nvtx::colors::orangered)
            .message(string_gamma)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    domain.range_pop();
    thread::sleep(time::Duration::from_millis(10));
    domain.range_pop();
    thread::sleep(time::Duration::from_millis(10));
    domain.range_pop();
}
