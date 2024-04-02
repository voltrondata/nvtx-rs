use std::{thread, time};

fn main() {
    let domain = nvtx::Domain::new("Domain");
    let [alpha, beta, gamma] = domain.register_strings(["alpha", "beta", "gamma"]);
    let [a, b] = domain.register_categories(["A", "B"]);

    let _r = domain.range("Duration");

    let r1 = domain.range(
        domain
            .event_attributes_builder()
            .category(a.clone())
            .color(nvtx::color::olive)
            .message(alpha.clone())
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
    let zz = yy.release();
    drop(zz);

    thread::sleep(time::Duration::from_millis(10));
    let r2 = domain.range(
        domain
            .event_attributes_builder()
            .category(a.clone())
            .color(nvtx::color::olive)
            .message(beta.clone())
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    let r3 = domain.range(
        domain
            .event_attributes_builder()
            .category(a.clone())
            .color(nvtx::color::olive)
            .message(gamma.clone())
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    drop(r1);
    thread::sleep(time::Duration::from_millis(10));
    drop(r2);
    thread::sleep(time::Duration::from_millis(10));
    drop(r3);
    let d2 = nvtx::Domain::new("cool");

    let p1 = d2.range(
        domain
            .event_attributes_builder()
            .category(b.clone())
            .color(nvtx::color::orangered)
            .message(alpha.clone())
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    let p2 = d2.range(
        domain
            .event_attributes_builder()
            .category(b.clone())
            .color(nvtx::color::orangered)
            .message(beta.clone())
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    let p3 = d2.range(
        domain
            .event_attributes_builder()
            .category(b.clone())
            .color(nvtx::color::orangered)
            .message(gamma.clone())
            .build(),
    );
    thread::sleep(time::Duration::from_millis(10));
    drop(p3);
    thread::sleep(time::Duration::from_millis(10));
    drop(p2);
    thread::sleep(time::Duration::from_millis(10));
    drop(p1);
}
