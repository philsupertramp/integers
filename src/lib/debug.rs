/// Debug utils
#[derive(Clone, Copy)]
pub struct OverflowStats {
    /// i32 overflow in backward loops
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


#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vals(forward: u64, backward: u64, clamps: u64) {
        crate::debug::OVERFLOW_STATS.with(|s: &std::cell::RefCell<crate::debug::OverflowStats>| {
            let s = s.borrow();

            assert_eq!(s.forward_wraps, forward);
            assert_eq!(s.backward_wraps, backward);
            assert_eq!(s.downcast_clamps, clamps);
        });
    }

    #[test]
    fn test_reset_overflow_stats(){
        crate::debug::OVERFLOW_STATS.with(|s: &std::cell::RefCell<crate::debug::OverflowStats>| {
            let mut s = s.borrow_mut();

            s.forward_wraps = 10;
            s.backward_wraps = 20;
            s.downcast_clamps = 100;
        });

        assert_vals(10, 20, 100);
        reset_overflow_stats();
        assert_vals(0, 0, 0);
    }

    #[test]
    fn test_check_add_counting(){
        let mut val: i32 = 10;
        let val2: i32 = 20;

        let new_val = checked_add_counting!(val, val2, forward_wraps);

        assert_vals(0, 0, 0);
        assert_eq!(new_val, val + val2);

        val = i32::MAX;
        let new_val = checked_add_counting!(val, val2, forward_wraps);

        assert_vals(1, 0, 0);
        assert_eq!(new_val, (val as i64 + val2 as i64) as i32);
    }

    #[test]
    fn test_check_sub_counting(){
        let mut val: i32 = 10;
        let val2: i32 = 20;

        let new_val = checked_sub_counting!(val, val2, backward_wraps);

        assert_vals(0, 0, 0);
        assert_eq!(new_val, val - val2);

        val = i32::MIN;
        let new_val = checked_sub_counting!(val, val2, backward_wraps);

        assert_vals(0, 1, 0);
        assert_eq!(new_val, (val as i64 - val2 as i64) as i32);
    }

    #[test]
    fn test_increase_clamp_downcast(){
        assert_vals(0, 0, 0);

        increase_clamp_downcast();

        assert_vals(0, 0, 1);
    }

    #[test]
    fn test_get_overflow_stats(){
        let stats = get_overflow_stats();

        assert_eq!(stats.forward_wraps, 0);
        assert_eq!(stats.backward_wraps, 0);
        assert_eq!(stats.downcast_clamps, 0);
        

        increase_clamp_downcast();
        increase_clamp_downcast();
        increase_clamp_downcast();
        increase_clamp_downcast();
        
        let new_stats = get_overflow_stats();

        assert_eq!(new_stats.forward_wraps, 0);
        assert_eq!(new_stats.backward_wraps, 0);
        assert_eq!(new_stats.downcast_clamps, 4);
    }
}

