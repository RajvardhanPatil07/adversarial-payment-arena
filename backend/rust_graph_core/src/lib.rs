use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};

#[derive(Default, Debug)]
struct RecentCustomers {
    events: VecDeque<(f64, String)>,
    counts: HashMap<String, usize>,
}

impl RecentCustomers {
    fn purge(&mut self, now: f64, window_seconds: f64) {
        while self
            .events
            .front()
            .map(|(ts, _)| now - *ts > window_seconds)
            .unwrap_or(false)
        {
            if let Some((_, customer)) = self.events.pop_front() {
                let remove = if let Some(count) = self.counts.get_mut(&customer) {
                    *count -= 1;
                    *count == 0
                } else {
                    false
                };
                if remove {
                    self.counts.remove(&customer);
                }
            }
        }
    }

    fn prospective_distinct(&mut self, now: f64, window_seconds: f64, customer: &str) -> usize {
        self.purge(now, window_seconds);
        self.counts.len() + usize::from(!self.counts.contains_key(customer))
    }

    fn observe(&mut self, now: f64, window_seconds: f64, customer: &str) {
        self.purge(now, window_seconds);
        self.events.push_back((now, customer.to_owned()));
        *self.counts.entry(customer.to_owned()).or_insert(0) += 1;
    }
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn device_risk(degree: usize) -> f64 {
    match degree {
        0 | 1 => 0.0,
        2 => 0.18,
        _ => (0.55 + 0.08 * (degree.saturating_sub(3) as f64)).min(0.95),
    }
}

fn ip_risk(degree: usize) -> f64 {
    match degree {
        0 | 1 => 0.0,
        2..=4 => 0.03 * (degree.saturating_sub(1) as f64),
        _ => (0.10 + 0.03 * (degree.saturating_sub(5) as f64)).min(0.35),
    }
}

fn merchant_risk(degree: usize) -> f64 {
    match degree {
        0..=3 => 0.0,
        4..=7 => 0.025 * (degree.saturating_sub(3) as f64),
        _ => (0.20 + 0.04 * (degree.saturating_sub(8) as f64)).min(0.65),
    }
}

fn fuse_risks(parts: &[f64]) -> f64 {
    clamp01(1.0 - parts.iter().fold(1.0, |acc, value| acc * (1.0 - clamp01(*value))))
}

/// Bounded, temporal graph-risk state used on the authorization hot path.
///
/// Shared devices are strong evidence. Shared IPs are weak by themselves
/// because office, campus and carrier NATs are legitimate; an IP can produce a
/// hard ring only when it coincides with beneficiary/merchant convergence.
#[pyclass(module = "arena_graph_core")]
pub struct GraphRiskState {
    window_seconds: f64,
    devices: HashMap<String, RecentCustomers>,
    ips: HashMap<String, RecentCustomers>,
    merchants: HashMap<String, RecentCustomers>,
}

#[pymethods]
impl GraphRiskState {
    #[new]
    #[pyo3(signature = (window_seconds=600.0))]
    fn new(window_seconds: f64) -> PyResult<Self> {
        if !window_seconds.is_finite() || window_seconds <= 0.0 {
            return Err(PyValueError::new_err(
                "window_seconds must be a finite positive number",
            ));
        }
        Ok(Self {
            window_seconds,
            devices: HashMap::with_capacity(8192),
            ips: HashMap::with_capacity(8192),
            merchants: HashMap::with_capacity(2048),
        })
    }

    /// Return prospective risk BEFORE the current transaction is observed.
    /// Tuple: risk, ring, device_degree, ip_degree, merchant_fanin.
    fn check(
        &mut self,
        ts: f64,
        customer_id: &str,
        device_id: &str,
        ip_address: &str,
        merchant_id: &str,
    ) -> (f64, bool, usize, usize, usize) {
        let device_degree = self
            .devices
            .entry(device_id.to_owned())
            .or_default()
            .prospective_distinct(ts, self.window_seconds, customer_id);
        let ip_degree = self
            .ips
            .entry(ip_address.to_owned())
            .or_default()
            .prospective_distinct(ts, self.window_seconds, customer_id);
        let merchant_degree = self
            .merchants
            .entry(merchant_id.to_owned())
            .or_default()
            .prospective_distinct(ts, self.window_seconds, customer_id);

        let d = device_risk(device_degree);
        let i = ip_risk(ip_degree);
        let m = merchant_risk(merchant_degree);
        let ring = device_degree >= 3 || (ip_degree >= 5 && merchant_degree >= 5);
        let mut risk = fuse_risks(&[d, i, m]);
        if ring {
            risk = risk.max(0.72);
        }
        (risk, ring, device_degree, ip_degree, merchant_degree)
    }

    fn observe(
        &mut self,
        ts: f64,
        customer_id: &str,
        device_id: &str,
        ip_address: &str,
        merchant_id: &str,
    ) {
        self.devices
            .entry(device_id.to_owned())
            .or_default()
            .observe(ts, self.window_seconds, customer_id);
        self.ips
            .entry(ip_address.to_owned())
            .or_default()
            .observe(ts, self.window_seconds, customer_id);
        self.merchants
            .entry(merchant_id.to_owned())
            .or_default()
            .observe(ts, self.window_seconds, customer_id);
    }

    fn state_sizes(&self) -> (usize, usize, usize) {
        (self.devices.len(), self.ips.len(), self.merchants.len())
    }

    fn backend_name(&self) -> &'static str {
        "rust"
    }
}

#[pymodule]
fn arena_graph_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GraphRiskState>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn third_customer_on_shared_device_is_strong_ring() {
        let mut state = GraphRiskState::new(600.0).unwrap();
        state.observe(1.0, "C1", "D", "1.1.1.1", "M1");
        state.observe(2.0, "C2", "D", "2.2.2.2", "M2");
        let (risk, ring, device_degree, _, _) =
            state.check(3.0, "C3", "D", "3.3.3.3", "M3");
        assert!(ring);
        assert_eq!(device_degree, 3);
        assert!(risk >= 0.72);
    }

    #[test]
    fn shared_ip_alone_is_not_a_hard_ring() {
        let mut state = GraphRiskState::new(600.0).unwrap();
        for i in 0..8 {
            state.observe(
                i as f64,
                &format!("C{i}"),
                &format!("D{i}"),
                "10.0.0.1",
                &format!("M{i}"),
            );
        }
        let (risk, ring, _, ip_degree, merchant_degree) =
            state.check(9.0, "C9", "D9", "10.0.0.1", "M9");
        assert!(!ring);
        assert_eq!(ip_degree, 9);
        assert_eq!(merchant_degree, 1);
        assert!(risk < 0.5);
    }

    #[test]
    fn shared_ip_plus_beneficiary_convergence_can_be_ring_evidence() {
        let mut state = GraphRiskState::new(600.0).unwrap();
        for i in 0..4 {
            state.observe(
                i as f64,
                &format!("C{i}"),
                &format!("D{i}"),
                "10.0.0.1",
                "MULE",
            );
        }
        let (risk, ring, _, ip_degree, merchant_degree) =
            state.check(5.0, "C4", "D4", "10.0.0.1", "MULE");
        assert!(ring);
        assert_eq!(ip_degree, 5);
        assert_eq!(merchant_degree, 5);
        assert!(risk >= 0.72);
    }

    #[test]
    fn temporal_window_expires_old_customers() {
        let mut state = GraphRiskState::new(10.0).unwrap();
        state.observe(0.0, "C1", "D", "I", "M");
        state.observe(1.0, "C2", "D", "I", "M");
        let (_, ring, device_degree, _, _) = state.check(20.0, "C3", "D", "I", "M");
        assert!(!ring);
        assert_eq!(device_degree, 1);
    }
}
