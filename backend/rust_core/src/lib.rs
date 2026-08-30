use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Copy, Debug)]
struct CustomerEvent {
    ts: f64,
    amount: f64,
    mcc: i64,
}

#[derive(Clone, Copy, Debug)]
struct DeviceEvent {
    ts: f64,
}

#[derive(Clone, Debug)]
struct MerchantEvent {
    ts: f64,
    customer_id: String,
}

/// Rolling transaction state used by the Python velocity scorer.
///
/// The Python layer still owns schema validation, issuer lookups and feature
/// names. Rust owns the allocation-heavy per-entity deques and window scans so
/// high-volume scoring spends less time in Python object/list machinery.
#[pyclass(module = "arena_core")]
pub struct RollingFeatureState {
    maxlen: usize,
    customer_events: HashMap<String, VecDeque<CustomerEvent>>,
    device_events: HashMap<String, VecDeque<DeviceEvent>>,
    merchant_events: HashMap<String, VecDeque<MerchantEvent>>,
    device_first_seen: HashMap<String, f64>,
}

fn push_bounded<T>(queue: &mut VecDeque<T>, value: T, maxlen: usize) {
    if queue.len() >= maxlen {
        queue.pop_front();
    }
    queue.push_back(value);
}

#[pymethods]
impl RollingFeatureState {
    #[new]
    #[pyo3(signature = (maxlen=500))]
    fn new(maxlen: usize) -> PyResult<Self> {
        if maxlen == 0 {
            return Err(PyValueError::new_err("maxlen must be greater than zero"));
        }
        Ok(Self {
            maxlen,
            customer_events: HashMap::new(),
            device_events: HashMap::new(),
            merchant_events: HashMap::new(),
            device_first_seen: HashMap::new(),
        })
    }

    /// Compute the dynamic rolling features for one transaction without
    /// mutating state. The returned vector is intentionally primitive to keep
    /// Python/Rust conversion overhead low:
    ///
    /// [cust_count_10m, cust_amount_sum_10m, amount_over_mean,
    ///  cust_mcc_distinct_1h, device_age_hours, dev_count_10m,
    ///  merch_count_10m, merch_distinct_customers_10m, cust_history_count]
    fn features(
        &self,
        ts: f64,
        customer_id: &str,
        device_id: &str,
        merchant_id: &str,
        amount: f64,
    ) -> Vec<f64> {
        let customer = self.customer_events.get(customer_id);

        let mut customer_count_10m = 0usize;
        let mut customer_amount_10m = 0.0f64;
        let mut customer_mcc_1h: HashSet<i64> = HashSet::new();
        let mut customer_amount_total = 0.0f64;
        let mut customer_history_count = 0usize;

        if let Some(events) = customer {
            for event in events {
                let age = ts - event.ts;
                // Preserve the existing Python semantics exactly: out-of-order
                // future events are not silently discarded here because doing
                // so would change the trained model's feature distribution.
                if age <= 600.0 {
                    customer_count_10m += 1;
                    customer_amount_10m += event.amount;
                }
                if age <= 3600.0 {
                    customer_mcc_1h.insert(event.mcc);
                }
                customer_amount_total += event.amount;
                customer_history_count += 1;
            }
        }

        let historical_mean = if customer_history_count == 0 {
            amount
        } else {
            customer_amount_total / customer_history_count as f64
        };
        let amount_over_mean = amount / (historical_mean + 1e-6);

        let device_count_10m = self
            .device_events
            .get(device_id)
            .map(|events| events.iter().filter(|event| ts - event.ts <= 600.0).count())
            .unwrap_or(0);

        let mut merchant_count_10m = 0usize;
        let mut merchant_customers_10m: HashSet<&str> = HashSet::new();
        if let Some(events) = self.merchant_events.get(merchant_id) {
            for event in events {
                if ts - event.ts <= 600.0 {
                    merchant_count_10m += 1;
                    merchant_customers_10m.insert(event.customer_id.as_str());
                }
            }
        }

        let device_age_hours = self
            .device_first_seen
            .get(device_id)
            .map(|first| (ts - first) / 3600.0)
            .unwrap_or(0.0);

        vec![
            customer_count_10m as f64,
            customer_amount_10m,
            amount_over_mean,
            customer_mcc_1h.len() as f64,
            device_age_hours,
            device_count_10m as f64,
            merchant_count_10m as f64,
            merchant_customers_10m.len() as f64,
            customer_history_count as f64,
        ]
    }

    /// Fold an accepted transaction into the rolling state.
    fn observe(
        &mut self,
        ts: f64,
        customer_id: &str,
        device_id: &str,
        merchant_id: &str,
        amount: f64,
        mcc: i64,
    ) {
        let maxlen = self.maxlen;
        push_bounded(
            self.customer_events
                .entry(customer_id.to_owned())
                .or_default(),
            CustomerEvent { ts, amount, mcc },
            maxlen,
        );
        push_bounded(
            self.device_events.entry(device_id.to_owned()).or_default(),
            DeviceEvent { ts },
            maxlen,
        );
        push_bounded(
            self.merchant_events
                .entry(merchant_id.to_owned())
                .or_default(),
            MerchantEvent {
                ts,
                customer_id: customer_id.to_owned(),
            },
            maxlen,
        );
        self.device_first_seen
            .entry(device_id.to_owned())
            .or_insert(ts);
    }

    fn backend_name(&self) -> &'static str {
        "rust"
    }
}

#[pymodule]
fn arena_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RollingFeatureState>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_has_expected_cold_start_features() {
        let state = RollingFeatureState::new(500).unwrap();
        let f = state.features(1_000.0, "C1", "D1", "M1", 100.0);
        assert_eq!(f[0], 0.0);
        assert!((f[2] - 1.0).abs() < 1e-5);
        assert_eq!(f[8], 0.0);
    }

    #[test]
    fn rolling_windows_and_distinct_customers_match_contract() {
        let mut state = RollingFeatureState::new(500).unwrap();
        state.observe(900.0, "C1", "D1", "M1", 50.0, 5411);
        state.observe(950.0, "C2", "D1", "M1", 75.0, 5732);
        state.observe(980.0, "C1", "D2", "M1", 25.0, 5732);

        let f = state.features(1_000.0, "C1", "D1", "M1", 100.0);
        assert_eq!(f[0], 2.0);
        assert_eq!(f[1], 75.0);
        assert_eq!(f[3], 2.0);
        assert_eq!(f[5], 2.0);
        assert_eq!(f[6], 3.0);
        assert_eq!(f[7], 2.0);
        assert_eq!(f[8], 2.0);
    }

    #[test]
    fn maxlen_is_bounded() {
        let mut state = RollingFeatureState::new(2).unwrap();
        state.observe(1.0, "C1", "D1", "M1", 10.0, 1);
        state.observe(2.0, "C1", "D1", "M1", 20.0, 2);
        state.observe(3.0, "C1", "D1", "M1", 30.0, 3);
        let f = state.features(4.0, "C1", "D1", "M1", 40.0);
        assert_eq!(f[0], 2.0);
        assert_eq!(f[1], 50.0);
        assert_eq!(f[8], 2.0);
    }
}
