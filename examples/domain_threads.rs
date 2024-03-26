use std::{
    ffi::{c_long, CString},
    thread::{self, sleep},
    time::Duration,
};

pub fn gettid() -> u32 {
    const SYS_GETTID: c_long = 186;
    extern "C" {
        fn syscall(num: c_long, ...) -> c_long;
    }
    unsafe { syscall(SYS_GETTID) as u32 }
}

fn main() {
    let domain = nvtx::Domain::new("Domain");
    let d = &domain;
    nvtx::name_thread(gettid(), CString::new("Main Thread").unwrap());

    thread::scope(|s| {
        let r = d.range("Start on main thread");
        let t1 = s.spawn(move || {
            nvtx::name_thread(gettid(), CString::new("Fork 1").unwrap());
            sleep(Duration::from_millis(10));
            drop(r);
        });
        let t2 = s.spawn(move || {
            nvtx::name_thread(gettid(), CString::new("Fork 2").unwrap());
            let r = d.range("Start on Fork 2");
            sleep(Duration::from_millis(20));
            r
        });
        let t3 = s.spawn(move || {
            nvtx::name_thread(gettid(), CString::new("Fork 3").unwrap());
            let _r = d.range("Start and end on Fork 3");
            sleep(Duration::from_millis(30));
        });

        let r1 = t2.join().unwrap();
        drop(r1);
        t1.join().expect("Failed to join");
        t3.join().expect("Failed to join");
    });
}
