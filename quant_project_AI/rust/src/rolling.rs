/// High-performance ring buffer for OHLCV data.
/// Provides contiguous array views without copying on every append.
pub struct RollingOhlcv {
    capacity: usize,
    count: usize,
    write_pos: usize,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
}

impl RollingOhlcv {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            count: 0,
            write_pos: 0,
            open: vec![0.0; capacity],
            high: vec![0.0; capacity],
            low: vec![0.0; capacity],
            close: vec![0.0; capacity],
            volume: vec![0.0; capacity],
        }
    }

    /// Append a single bar. O(1) operation.
    pub fn append(&mut self, o: f64, h: f64, l: f64, c: f64, v: f64) {
        self.open[self.write_pos] = o;
        self.high[self.write_pos] = h;
        self.low[self.write_pos] = l;
        self.close[self.write_pos] = c;
        self.volume[self.write_pos] = v;
        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get contiguous arrays. If buffer hasn't wrapped, returns slices.
    /// If wrapped, allocates and concatenates (unavoidable for contiguous output).
    pub fn to_arrays(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        if self.count == 0 {
            return (vec![], vec![], vec![], vec![], vec![]);
        }
        if self.count < self.capacity {
            return (
                self.open[..self.count].to_vec(),
                self.high[..self.count].to_vec(),
                self.low[..self.count].to_vec(),
                self.close[..self.count].to_vec(),
                self.volume[..self.count].to_vec(),
            );
        }
        let start = self.write_pos;
        let mut o = Vec::with_capacity(self.capacity);
        let mut h = Vec::with_capacity(self.capacity);
        let mut l = Vec::with_capacity(self.capacity);
        let mut c = Vec::with_capacity(self.capacity);
        let mut v = Vec::with_capacity(self.capacity);
        o.extend_from_slice(&self.open[start..]);
        o.extend_from_slice(&self.open[..start]);
        h.extend_from_slice(&self.high[start..]);
        h.extend_from_slice(&self.high[..start]);
        l.extend_from_slice(&self.low[start..]);
        l.extend_from_slice(&self.low[..start]);
        c.extend_from_slice(&self.close[start..]);
        c.extend_from_slice(&self.close[..start]);
        v.extend_from_slice(&self.volume[start..]);
        v.extend_from_slice(&self.volume[..start]);
        (o, h, l, c, v)
    }

    /// Get the last N close values as a contiguous slice.
    pub fn last_n_close(&self, n: usize) -> Vec<f64> {
        let actual = n.min(self.count);
        let mut out = Vec::with_capacity(actual);
        for i in 0..actual {
            let idx = if self.write_pos >= actual {
                self.write_pos - actual + i
            } else {
                (self.capacity + self.write_pos - actual + i) % self.capacity
            };
            out.push(self.close[idx]);
        }
        out
    }

    /// Get the most recent close price.
    pub fn latest_close(&self) -> Option<f64> {
        if self.count == 0 { return None; }
        let idx = if self.write_pos == 0 {
            self.capacity - 1
        } else {
            self.write_pos - 1
        };
        Some(self.close[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let mut rb = RollingOhlcv::new(3);
        rb.append(1.0, 2.0, 0.5, 1.5, 100.0);
        rb.append(1.5, 2.5, 1.0, 2.0, 200.0);
        assert_eq!(rb.len(), 2);
        let (_, _, _, c, _) = rb.to_arrays();
        assert_eq!(c, vec![1.5, 2.0]);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let mut rb = RollingOhlcv::new(3);
        for i in 1..=5 {
            rb.append(i as f64, i as f64, i as f64, i as f64, i as f64);
        }
        assert_eq!(rb.len(), 3);
        let (_, _, _, c, _) = rb.to_arrays();
        assert_eq!(c, vec![3.0, 4.0, 5.0]);
        assert_eq!(rb.latest_close(), Some(5.0));
    }
}
