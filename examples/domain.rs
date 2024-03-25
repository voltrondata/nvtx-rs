use std::{thread, time};

use nvtx_rs::nvtx;

fn main() {
    let mut domain = nvtx::Domain::new("Domain");
    let [alpha, beta, gamma] = domain.register_strings(["alpha", "beta", "gamma"]);
    let [a, b] = domain.register_categories(["A", "B"]);

    let _r = domain.range("Duration");

    let r1 = domain.range(
        nvtx::AttributeBuilder::default()
            .category(&a)
            .color(nvtx::colors::olive)
            .message(&alpha)
            .build(),
    );

    let x = domain.user_sync("cool");
    thread::sleep(time::Duration::from_millis(10));
    let y = x.acquire();
    thread::sleep(time::Duration::from_millis(10));
    let z = y.failed();
    thread::sleep(time::Duration::from_millis(10));
    let xx = z.acquire();
    thread::sleep(time::Duration::from_millis(10));
    let yy = xx.success();
    thread::sleep(time::Duration::from_millis(10));
    yy.releasing();

    thread::sleep(time::Duration::from_millis(10));
    let r2 = domain.range(
        nvtx::AttributeBuilder::default()
            .category(&a)
            .color(nvtx::colors::olive)
            .message(&beta)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    let r3 = domain.range(
        nvtx::AttributeBuilder::default()
            .category(&a)
            .color(nvtx::colors::olive)
            .message(&gamma)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    drop(r1);
    thread::sleep(time::Duration::from_millis(10));
    drop(r2);
    thread::sleep(time::Duration::from_millis(10));
    drop(r3);
    let p1 = domain.range(
        nvtx::AttributeBuilder::default()
            .category(&b)
            .color(nvtx::colors::orangered)
            .message(&alpha)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    let p2 = domain.range(
        nvtx::AttributeBuilder::default()
            .category(&b)
            .color(nvtx::colors::orangered)
            .message(&beta)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    let p3 = domain.range(
        nvtx::AttributeBuilder::default()
            .category(&b)
            .color(nvtx::colors::orangered)
            .message(&gamma)
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    drop(p3);
    thread::sleep(time::Duration::from_millis(10));
    drop(p2);
    thread::sleep(time::Duration::from_millis(10));
    drop(p1);
}
