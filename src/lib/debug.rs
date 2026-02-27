/// Debug utils
#[derive(Clone, Copy)]
pub struct OverflowStats {
    /// i16 overflow in backward loops
    pub forward_wraps:  u64,
    /// i32 master weight update overflow
    pub backward_wraps: u64,
    /// stochastic_downcast saturation
    pub downcast_clamps: u64,
}

#[cfg(debug_assertions)]
thread_local! {
    pub static OVERFLOW_STATS: std::cell::RefCell<OverflowStats> = 
        std::cell::RefCell::new(OverflowStats { 
            forward_wraps: 0, 
            backward_wraps: 0, 
            downcast_clamps: 0 
        });
}
#[cfg(debug_assertions)]
pub fn reset_overflow_stats() {
    crate::debug::OVERFLOW_STATS.with(|s: &std::cell::RefCell<crate::debug::OverflowStats>| {
        let mut s = s.borrow_mut();
        s.forward_wraps   = 0;
        s.backward_wraps  = 0;
        s.downcast_clamps = 0;
    });
}
#[macro_export]
macro_rules! checked_add_counting {
    ($acc:expr, $val:expr, $counter:ident) => {{
        #[cfg(debug_assertions)]
        {
            if $acc.checked_add($val).is_none() {
                // Explicitly type 's' as &RefCell<OverflowStats>
                $crate::debug::OVERFLOW_STATS.with(|s: &std::cell::RefCell<$crate::debug::OverflowStats>| {
                    s.borrow_mut().$counter += 1
                });
            }
        }
        $acc.wrapping_add($val)
    }};
}

#[macro_export]
macro_rules! checked_sub_counting {
    ($acc:expr, $val:expr, $counter:ident) => {{
        #[cfg(debug_assertions)]
        {
            if $acc.checked_sub($val).is_none() {
                // Explicitly type 's' as &RefCell<OverflowStats>
                $crate::debug::OVERFLOW_STATS.with(|s: &std::cell::RefCell<$crate::debug::OverflowStats>| {
                    s.borrow_mut().$counter += 1
                });
            }
        }
        $acc.wrapping_sub($val)
    }};
}

pub fn increase_clamp_downcast(){
    #[cfg(debug_assertions)]
    crate::debug::OVERFLOW_STATS.with(|s: &std::cell::RefCell<crate::debug::OverflowStats>| { s.borrow_mut().downcast_clamps += 1; });
}

#[cfg(debug_assertions)]
pub fn get_overflow_stats() -> OverflowStats {
    crate::debug::OVERFLOW_STATS.with(|s: &std::cell::RefCell<crate::debug::OverflowStats>| {
        let s = s.borrow();

        println!("forward wraps: {}", s.forward_wraps);
        println!("backward wraps: {}", s.backward_wraps);
        println!("downcast clamps: {}", s.downcast_clamps);
        OverflowStats {
            forward_wraps: s.forward_wraps,
            backward_wraps: s.backward_wraps,
            downcast_clamps: s.downcast_clamps,
        }
    })
}

