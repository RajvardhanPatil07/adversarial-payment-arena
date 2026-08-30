use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Copy, Debug)]
struct CustomerEvent {
    ts: f64,
    amount: f64,
    mcc: i64,
}

#[derive(Default, Debug)]
struct CustomerState {
    events: VecDeque<CustomerEvent>,
    amount_sum: f64,
    ordered: bool,
}

#[derive(Clone, Copy, Debug)]
struct DeviceEvent {
    ts: f64,
}

#[derive(Default, Debug)]
struct DeviceState {
    events: VecDeque<DeviceEvent>,
    ordered: bool,
}

#[derive(Clone, Debug)]
struct MerchantEvent {
    ts: f64,
    customer_id: String,
}

#[derive(Default, Debug)]
struct MerchantState {
    events: VecDeque<MerchantEvent>,
    ordered: bool,
}

/// Rolling transaction state used by the Python velocity scorer.
///
/// The normal arena stream is chronological. For that common case, window
/// scans walk newest-to-oldest and stop as soon as the requested horizon is
/// crossed. If an entity ever receives an out-of-order observation, its state
/// permanently falls back to a full scan so the original Python semantics are
/// preserved exactly for replay/experiment workloads.
#[pyclass(module = "arena_core")]
pub struct RollingFeatureState {
    maxlen: usize,
    customer_states: HashMap<String, CustomerState>,
    device_states: HashMap<String, DeviceState>,
    merchant_states: HashMap<String, MerchantState>,
    device_first_seen: HashMap<String, f64>,
}

fn was_monotonic<T, F>(queue: &VecDeque<T>, ts: f64, get_ts: F) -> bool
where
    F: Fn(&T) -> f64,
{
    queue.back().map(|last| ts >= get_ts(last)).unwrap_or(true)
}

impl RollingFeatureState {
    fn compute_features(
        &self,
        ts: f64,
        customer_id: &str,
        device_id: &str,
        merchant_id: &str,
        amount: f64,
    ) -> [f64; 9] {
        let mut customer_count_10m = 0usize;
        let mut customer_amount_10m = 0.0f64;
        let mut customer_mcc_1h: HashSet<i64> = HashSet::new();
        let mut customer_history_count = 0usize;
        let mut customer_amount_total = 0.0f64;

        if let Some(state) = self.customer_states.get(customer_id) {
            customer_history_count = state.events.len();
            customer_amount_total = state.amount_sum;

            if state.ordered {
                for event in state.events.iter().rev() {
                    let age = ts - event.ts;
                    if age > 3600.0 {
                        break;
                    }
                    customer_mcc_1h.insert(event.mcc);
                    if age <= 600.0 {
                        customer_count_10m += 1;
                        customer_amount_10m += event.amount;
                    }
                }
            } else {
                for event in &state.events {
                    let age = ts - event.ts;
                    if age <= 600.0 {
                        customer_count_10m += 1;
                        customer_amount_10m += event.amount;
                    }
                    if age <= 3600.0 {
                        customer_mcc_1h.insert(event.mcc);
                    }
                }
            }
        }

        let historical_mean = if customer_history_count == 0 {
            amount
        } else {
            customer_amount_total / customer_history_count as f64
        };
        let amount_over_mean = amount / (historical_mean + 1e-6);

        let device_count_10m = self
            .device_states
            .get(device_id)
            .map(|state| {
                if state.ordered {
                    state
                        .events
                        .iter()
                        .rev()
                        .take_while(|event| ts - event.ts <= 600.0)
                        .count()
                } else {
                    state
                        .events
                        .iter()
                        .filter(|event| ts - event.ts <= 600.0)
                        .count()
                }
            })
            .unwrap_or(0);

        let mut merchant_count_10m = 0usize;
        let mut merchant_customers_10m: HashSet<&str> = HashSet::new();
        if let Some(state) = self.merchant_states.get(merchant_id) {
            if state.ordered {
                for event in state.events.iter().rev() {
                    if ts - event.ts > 600.0 {
                        break;
                    }
                    merchant_count_10m += 1;
                    merchant_customers_10m.insert(event.customer_id.as_str());
                }
            } else {
                for event in &state.events {
                    if ts - event.ts <= 600.0 {
                        merchant_count_10m += 1;
                        merchant_customers_10m.insert(event.customer_id.as_str());
                    }
                }
            }
        }

        let device_age_hours = self
            .device_first_seen
            .get(device_id)
            .map(|first| (ts - first) / 3600.0)
            .unwrap_or(0.0);

        [
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
            customer_states: HashMap::with_capacity(1024),
            device_states: HashMap::with_capacity(2048),
            merchant_states: HashMap::with_capacity(64),
            device_first_seen: HashMap::with_capacity(2048),
        })
    }

    /// Compute dynamic rolling features without mutating state.
    ///
    /// A fixed tuple avoids allocating a Rust Vec for every score. Python keeps
    /// the public feature names/order and maps these primitive values into the
    /// existing XGBoost feature contract.
    fn features(
        &self,
        ts: f64,
        customer_id: &str,
        device_id: &str,
        merchant_id: &str,
        amount: f64,
    ) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let f = self.compute_features(ts, customer_id, device_id, merchant_id, amount);
        (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8])
    }

    /// Fold an accepted transaction into bounded per-entity state.
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

        let customer = self
            .customer_states
            .entry(customer_id.to_owned())
            .or_insert_with(|| CustomerState {
                ordered: true,
                ..CustomerState::default()
            });
        customer.ordered &= was_monotonic(&customer.events, ts, |event| event.ts);
        if customer.events.len() >= maxlen {
            if let Some(evicted) = customer.events.pop_front() {
                customer.amount_sum -= evicted.amount;
            }
        }
        customer.events.push_back(CustomerEvent { ts, amount, mcc });
        customer.amount_sum += amount;

        let device = self
            .device_states
            .entry(device_id.to_owned())
            .or_insert_with(|| DeviceState {
                ordered: true,
                ..DeviceState::default()
            });
        device.ordered &= was_monotonic(&device.events, ts, |event| event.ts);
        if device.events.len() >= maxlen {
            device.events.pop_front();
        }
        device.events.push_back(DeviceEvent { ts });

        let merchant = self
            .merchant_states
            .entry(merchant_id.to_owned())
            .or_insert_with(|| MerchantState {
                ordered: true,
                ..MerchantState::default()
            });
        merchant.ordered &= was_monotonic(&merchant.events, ts, |event| event.ts);
        if merchant.events.len() >= maxlen {
            merchant.events.pop_front();
        }
        merchant.events.push_back(MerchantEvent {
            ts,
            customer_id: customer_id.to_owned(),
        });

        self.device_first_seen
            .entry(device_id.to_owned())
            .or_insert(ts);
    }

    fn backend_name(&self) -> &'static str {
        "rust"
    }

    /// Lightweight observability for load tests and /health diagnostics.
    fn state_sizes(&self) -> (usize, usize, usize, usize) {
        (
            self.customer_states.len(),
            self.device_states.len(),
            self.merchant_states.len(),
            self.device_first_seen.len(),
        )
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
        let f = state.compute_features(1_000.0, "C1", "D1", "M1", 100.0);
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

        let f = state.compute_features(1_000.0, "C1", "D1", "M1", 100.0);
        assert_eq!(f[0], 2.0);
        assert_eq!(f[1], 75.0);
        assert_eq!(f[3], 2.0);
        assert_eq!(f[5], 2.0);
        assert_eq!(f[6], 3.0);
        assert_eq!(f[7], 2.0);
        assert_eq!(f[8], 2.0);
    }

    #[test]
    fn maxlen_keeps_running_mean_exact() {
        let mut state = RollingFeatureState::new(2).unwrap();
        state.observe(1.0, "C1", "D1", "M1", 10.0, 1);
        state.observe(2.0, "C1", "D1", "M1", 20.0, 2);
        state.observe(3.0, "C1", "D1", "M1", 30.0, 3);
        let f = state.compute_features(4.0, "C1", "D1", "M1", 40.0);
        assert_eq!(f[0], 2.0);
        assert_eq!(f[1], 50.0);
        assert_eq!(f[8], 2.0);
        assert!((f[2] - (40.0 / 25.000001)).abs() < 1e-10);
    }

    #[test]
    fn out_of_order_events_preserve_full_scan_semantics() {
        let mut state = RollingFeatureState::new(10).unwrap();
        state.observe(1_000.0, "C1", "D1", "M1", 10.0, 1);
        state.observe(100.0, "C1", "D1", "M1", 20.0, 2); // marks state unordered
        state.observe(950.0, "C1", "D1", "M1", 30.0, 3);

        let f = state.compute_features(1_000.0, "C1", "D1", "M1", 40.0);
        assert_eq!(f[0], 2.0); // ts=100 is outside 10m; 1000 and 950 are inside
        assert_eq!(f[3], 3.0); // all three are within 1h under original <= semantics
        assert_eq!(f[5], 2.0);
        assert_eq!(f[6], 2.0);
    }
}
