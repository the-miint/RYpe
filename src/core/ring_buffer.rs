//! Array-backed ring buffer for O(1) deque operations with cache locality.
//!
//! This module provides a fixed-size circular buffer optimized for the
//! monotonic deque pattern used in minimizer extraction. Unlike `VecDeque`,
//! the buffer is stack-allocated with a known size, providing better cache
//! locality for small, fixed-size windows.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

/// A fixed-size circular buffer with deque-like operations.
///
/// This is optimized for the minimizer extraction algorithm which uses
/// a sliding window of bounded size. The buffer provides O(1) push/pop
/// operations at both ends with better cache locality than `VecDeque`.
///
/// # Type Parameters
/// * `T` - Element type
/// * `N` - Maximum capacity (const generic)
///
/// # Panics
/// `push_back` panics if the buffer is full. Callers must ensure
/// elements are popped before exceeding capacity.
///
/// # Panic Safety
/// This type is NOT unwind-safe. If a panic occurs during operations,
/// the buffer may be left in an inconsistent state. The `push_back` method
/// panics before modifying state, so in practice this is safe for the
/// intended use case (monotonic deque in minimizer extraction). However,
/// this type should not be used with `std::panic::catch_unwind`.
pub struct RingBuffer<T, const N: usize> {
    data: [MaybeUninit<T>; N],
    head: usize,
    len: usize,
    // Marker to make this type !UnwindSafe and !RefUnwindSafe
    // UnsafeCell is !Sync which transitively makes this !RefUnwindSafe
    _marker: UnsafeCell<()>,
}

impl<T, const N: usize> RingBuffer<T, N> {
    /// Create a new empty ring buffer.
    #[inline]
    pub fn new() -> Self {
        Self {
            // SAFETY: MaybeUninit doesn't require initialization
            data: unsafe { MaybeUninit::uninit().assume_init() },
            head: 0,
            len: 0,
            _marker: UnsafeCell::new(()),
        }
    }

    /// Clears all elements from the buffer.
    #[inline]
    pub fn clear(&mut self) {
        // Drop all elements
        for i in 0..self.len {
            let idx = (self.head + i) % N;
            // SAFETY: Elements 0..len are initialized
            unsafe {
                self.data[idx].assume_init_drop();
            }
        }
        self.head = 0;
        self.len = 0;
    }

    /// Returns a reference to the front element, or `None` if empty.
    #[inline]
    pub fn front(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            // SAFETY: head is valid when len > 0
            Some(unsafe { self.data[self.head].assume_init_ref() })
        }
    }

    /// Returns a reference to the back element, or `None` if empty.
    #[inline]
    pub fn back(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            let idx = (self.head + self.len - 1) % N;
            // SAFETY: This index is valid when len > 0
            Some(unsafe { self.data[idx].assume_init_ref() })
        }
    }

    /// Adds an element to the back of the buffer.
    ///
    /// # Panics
    /// Panics if the buffer is full (len == N).
    #[inline]
    pub fn push_back(&mut self, value: T) {
        assert!(
            self.len < N,
            "RingBuffer overflow: len={}, N={}",
            self.len,
            N
        );
        let idx = (self.head + self.len) % N;
        self.data[idx] = MaybeUninit::new(value);
        self.len += 1;
    }

    /// Removes and returns the front element, or `None` if empty.
    #[inline]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            // SAFETY: head is valid when len > 0
            let value = unsafe { self.data[self.head].assume_init_read() };
            self.head = (self.head + 1) % N;
            self.len -= 1;
            Some(value)
        }
    }

    /// Removes and returns the back element, or `None` if empty.
    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            let idx = (self.head + self.len) % N;
            // SAFETY: This index was valid before decrementing len
            Some(unsafe { self.data[idx].assume_init_read() })
        }
    }
}

impl<T, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for RingBuffer<T, N> {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_buffer_is_empty() {
        let buf: RingBuffer<i32, 8> = RingBuffer::new();
        assert_eq!(buf.front(), None);
        assert_eq!(buf.back(), None);
    }

    #[test]
    fn test_push_back_and_front() {
        let mut buf: RingBuffer<i32, 8> = RingBuffer::new();
        buf.push_back(1);
        buf.push_back(2);
        buf.push_back(3);
        assert_eq!(buf.front(), Some(&1));
        assert_eq!(buf.back(), Some(&3));
    }

    #[test]
    fn test_pop_front() {
        let mut buf: RingBuffer<i32, 8> = RingBuffer::new();
        buf.push_back(1);
        buf.push_back(2);
        buf.push_back(3);
        assert_eq!(buf.pop_front(), Some(1));
        assert_eq!(buf.pop_front(), Some(2));
        assert_eq!(buf.front(), Some(&3));
    }

    #[test]
    fn test_pop_back() {
        let mut buf: RingBuffer<i32, 8> = RingBuffer::new();
        buf.push_back(1);
        buf.push_back(2);
        buf.push_back(3);
        assert_eq!(buf.pop_back(), Some(3));
        assert_eq!(buf.pop_back(), Some(2));
        assert_eq!(buf.back(), Some(&1));
    }

    #[test]
    fn test_clear() {
        let mut buf: RingBuffer<i32, 8> = RingBuffer::new();
        buf.push_back(1);
        buf.push_back(2);
        buf.clear();
        assert_eq!(buf.front(), None);
        assert_eq!(buf.back(), None);
    }

    #[test]
    fn test_wraparound() {
        let mut buf: RingBuffer<i32, 4> = RingBuffer::new();
        // Fill and partially empty to cause wraparound
        buf.push_back(1);
        buf.push_back(2);
        buf.push_back(3);
        buf.pop_front(); // removes 1, head moves
        buf.pop_front(); // removes 2, head moves
        buf.push_back(4);
        buf.push_back(5);
        // Now: [3, 4, 5] with head wrapped
        assert_eq!(buf.pop_front(), Some(3));
        assert_eq!(buf.pop_front(), Some(4));
        assert_eq!(buf.pop_front(), Some(5));
        assert_eq!(buf.front(), None);
    }

    #[test]
    fn test_tuple_type() {
        // Test with the actual type used in minimizer extraction
        let mut buf: RingBuffer<(usize, u64), 256> = RingBuffer::new();
        buf.push_back((0, 12345));
        buf.push_back((1, 67890));
        assert_eq!(buf.front(), Some(&(0, 12345)));
        assert_eq!(buf.back(), Some(&(1, 67890)));
        assert_eq!(buf.pop_front(), Some((0, 12345)));
        assert_eq!(buf.front(), Some(&(1, 67890)));
    }

    #[test]
    fn test_monotonic_deque_pattern() {
        // Simulate the monotonic deque pattern used in minimizer extraction
        let mut buf: RingBuffer<(usize, u64), 8> = RingBuffer::new();
        let window_size = 4;

        // Simulate processing positions 0..10 with hashes
        let hashes = [50u64, 30, 40, 20, 60, 10, 70, 80, 15, 25];

        for (pos, &hash) in hashes.iter().enumerate() {
            // Remove elements outside window
            while let Some(&(p, _)) = buf.front() {
                if p + window_size <= pos {
                    buf.pop_front();
                } else {
                    break;
                }
            }
            // Remove elements with larger hash
            while let Some(&(_, v)) = buf.back() {
                if v >= hash {
                    buf.pop_back();
                } else {
                    break;
                }
            }
            buf.push_back((pos, hash));

            // After window is full, front should be minimum
            if pos >= window_size - 1 {
                let min = buf.front().unwrap();
                // Verify it's actually the minimum in window
                let window_start = pos.saturating_sub(window_size - 1);
                let actual_min = hashes[window_start..=pos].iter().min().unwrap();
                assert_eq!(min.1, *actual_min);
            }
        }
    }
}
