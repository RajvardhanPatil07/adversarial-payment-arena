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
    ten_min: VecDeque<(f64, f64)>,
    ten_min_amount_sum: f64,
    one_hour_mcc: VecDeque<(f64, i64)>,
    one_hour_mcc_counts: HashMap<i64, usize>,
}

#[derive(Clone, Copy, Debug)]
struct DeviceEvent {
    ts: f64,
}

#[derive(Default, Debug)]
struct DeviceState {
    events: VecDeque<DeviceEvent>,
    ordered: bool,
    ten_min: VecDeque<f64>,
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
    ten_min: VecDeque<(f64, String)>,
    ten_min_customer_counts: HashMap<String, usize>,
}

/// Rolling transaction state used by the Python velocity scorer.
///
/// The normal arena stream is chronological. The fused high-throughput path
/// maintains incremental 10-minute/1-hour queues and counters, making feature
/// reads amortized O(1) with respect to retained history. If an entity receives
/// an out-of-order observation, it permanently falls back to the original full
/// scan semantics so replay/experiment behavior remains unchanged.
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

fn decrement_count<K>(counts: &mut HashMap<K, usize>, key: &K)
where
    K: Eq + std::hash::Hash,
{
    let remove = if let Some(count) = counts.get_mut(key) {
        *count -= 1;
        *count == 0
    } else {
        false
    };
    if remove {
        counts.remove(key);
    }
}

fn purge_customer_windows(state: &mut CustomerState, ts: f64) {
    while state
        .ten_min
        .front()
        .map(|(event_ts, _)| ts - *event_ts > 600.0)
        .unwrap_or(false)
    {
        if let Some((_, amount)) = state.ten_min.pop_front() {
            state.ten_min_amount_sum -= amount;
        }
    }

    while state
        .one_hour_mcc
        .front()
        .map(|(event_ts, _)| ts - *event_ts > 3600.0)
        .unwrap_or(false)
    {
        if let Some((_, mcc)) = state.one_hour_mcc.pop_front() {
            decrement_count(&mut state.one_hour_mcc_counts, &mcc);
        }
    }
}

fn remove_customer_eviction_from_windows(state: &mut CustomerState, evicted: CustomerEvent) {
    if state
        .ten_min
        .front()
        .map(|(ts, amount)| *ts == evicted.ts && *amount == evicted.amount)
        .unwrap_or(false)
    {
        state.ten_min.pop_front();
        state.ten_min_amount_sum -= evicted.amount;
    }

    if state
        .one_hour_mcc
        .front()
        .map(|(ts, mcc)| *ts == evicted.ts && *mcc == evicted.mcc)
        .unwrap_or(false)
    {
        state.one_hour_mcc.pop_front();
        decrement_count(&mut state.one_hour_mcc_counts, &evicted.mcc);
    }
}

fn purge_device_window(state: &mut DeviceState, ts: f64) {
    while state
        .ten_min
        .front()
        .map(|event_ts| ts - *event_ts > 600.0)
        .unwrap_or(false)
    {
        state.ten_min.pop_front();
    }
}

fn remove_device_eviction_from_window(state: &mut DeviceState, evicted: DeviceEvent) {
    if state
        .ten_min
        .front()
        .map(|event_ts| *event_ts == evicted.ts)
        .unwrap_or(false)
    {
        state.ten_min.pop_front();
    }
}

fn purge_merchant_window(state: &mut MerchantState, ts: f64) {
    while state
        .ten_min
        .front()
        .map(|(event_ts, _)| ts - *event_ts > 600.0)
        .unwrap_or(false)
    {
        if let Some((_, customer_id)) = state.ten_min.pop_front() {
            decrement_count(&mut state.ten_min_customer_counts, &customer_id);
        }
    }
}

fn remove_merchant_eviction_from_window(state: &mut MerchantState, evicted: &MerchantEvent) {
    if state
        .ten_min
        .front()
        .map(|(ts, customer_id)| *ts == evicted.ts && customer_id == &evicted.customer_id)
        .unwrap_or(false)
    {
        if let Some((_, customer_id)) = state.ten_min.pop_front() {
            decrement_count(&mut state.ten_min_customer_counts, &customer_id);
        }
    }
}

impl RollingFeatureState {
    /// Reference implementation: scan retained history exactly like the Python
    /// fallback. Used by scalar ``features()`` and by unordered entities.
    fn compute_features_scan(
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

    /// Fast pre-observation feature read for the fused chronological path.
    /// Falls back per entity whenever the current timestamp would be unordered.
    fn compute_features_fast(
        &mut self,
        ts: f64,
        customer_id: &str,
        device_id: &str,
        merchant_id: &str,
        amount: f64,
    ) -> [f64; 9] {
        let (
            customer_count_10m,
            customer_amount_10m,
            customer_mcc_distinct_1h,
            customer_history_count,
            customer_amount_total,
        ) = if let Some(state) = self.customer_states.get_mut(customer_id) {
            let query_ordered = state.ordered
                && was_monotonic(&state.events, ts, |event| event.ts);
            if query_ordered {
                purge_customer_windows(state, ts);
                (
                    state.ten_min.len(),
                    state.ten_min_amount_sum,
                    state.one_hour_mcc_counts.len(),
                    state.events.len(),
                    state.amount_sum,
                )
            } else {
                let mut count_10m = 0usize;
                let mut amount_10m = 0.0f64;
                let mut mccs: HashSet<i64> = HashSet::new();
                for event in &state.events {
                    let age = ts - event.ts;
                    if age <= 600.0 {
                        count_10m += 1;
                        amount_10m += event.amount;
                    }
                    if age <= 3600.0 {
                        mccs.insert(event.mcc);
                    }
                }
                (
                    count_10m,
                    amount_10m,
                    mccs.len(),
                    state.events.len(),
                    state.amount_sum,
                )
            }
        } else {
            (0, 0.0, 0, 0, 0.0)
        };

        let historical_mean = if customer_history_count == 0 {
            amount
        } else {
            customer_amount_total / customer_history_count as f64
        };
        let amount_over_mean = amount / (historical_mean + 1e-6);

        let device_count_10m = if let Some(state) = self.device_states.get_mut(device_id) {
            let query_ordered = state.ordered
                && was_monotonic(&state.events, ts, |event| event.ts);
            if query_ordered {
                purge_device_window(state, ts);
                state.ten_min.len()
            } else {
                state
                    .events
                    .iter()
                    .filter(|event| ts - event.ts <= 600.0)
                    .count()
            }
        } else {
            0
        };

        let (merchant_count_10m, merchant_distinct_10m) =
            if let Some(state) = self.merchant_states.get_mut(merchant_id) {
                let query_ordered = state.ordered
                    && was_monotonic(&state.events, ts, |event| event.ts);
                if query_ordered {
                    purge_merchant_window(state, ts);
                    (state.ten_min.len(), state.ten_min_customer_counts.len())
                } else {
                    let mut count = 0usize;
                    let mut customers: HashSet<&str> = HashSet::new();
                    for event in &state.events {
                        if ts - event.ts <= 600.0 {
                            count += 1;
                            customers.insert(event.customer_id.as_str());
                        }
                    }
                    (count, customers.len())
                }
            } else {
                (0, 0)
            };

        let device_age_hours = self
            .device_first_seen
            .get(device_id)
            .map(|first| (ts - first) / 3600.0)
            .unwrap_or(0.0);

        [
            customer_count_10m as f64,
            customer_amount_10m,
            amount_over_mean,
            customer_mcc_distinct_1h as f64,
            device_age_hours,
            device_count_10m as f64,
            merchant_count_10m as f64,
            merchant_distinct_10m as f64,
            customer_history_count as f64,
        ]
    }

    fn observe_inner(
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
        let monotonic = customer.ordered
            && was_monotonic(&customer.events, ts, |event| event.ts);
        if monotonic {
            purge_customer_windows(customer, ts);
        } else {
            customer.ordered = false;
        }
        if customer.events.len() >= maxlen {
            if let Some(evicted) = customer.events.pop_front() {
                customer.amount_sum -= evicted.amount;
                if customer.ordered {
                    remove_customer_eviction_from_windows(customer, evicted);
                }
            }
        }
        customer.events.push_back(CustomerEvent { ts, amount, mcc });
        customer.amount_sum += amount;
        if customer.ordered {
            customer.ten_min.push_back((ts, amount));
            customer.ten_min_amount_sum += amount;
            customer.one_hour_mcc.push_back((ts, mcc));
            *customer.one_hour_mcc_counts.entry(mcc).or_insert(0) += 1;
        }

        let device = self
            .device_states
            .entry(device_id.to_owned())
            .or_insert_with(|| DeviceState {
                ordered: true,
                ..DeviceState::default()
            });
        let monotonic = device.ordered
            && was_monotonic(&device.events, ts, |event| event.ts);
        if monotonic {
            purge_device_window(device, ts);
        } else {
            device.ordered = false;
        }
        if device.events.len() >= maxlen {
            if let Some(evicted) = device.events.pop_front() {
                if device.ordered {
                    remove_device_eviction_from_window(device, evicted);
                }
            }
        }
        device.events.push_back(DeviceEvent { ts });
        if device.ordered {
            device.ten_min.push_back(ts);
        }

        let merchant = self
            .merchant_states
            .entry(merchant_id.to_owned())
            .or_insert_with(|| MerchantState {
                ordered: true,
                ..MerchantState::default()
            });
        let monotonic = merchant.ordered
            && was_monotonic(&merchant.events, ts, |event| event.ts);
        if monotonic {
            purge_merchant_window(merchant, ts);
        } else {
            merchant.ordered = false;
        }
        if merchant.events.len() >= maxlen {
            if let Some(evicted) = merchant.events.pop_front() {
                if merchant.ordered {
                    remove_merchant_eviction_from_window(merchant, &evicted);
                }
            }
        }
        merchant.events.push_back(MerchantEvent {
            ts,
            customer_id: customer_id.to_owned(),
        });
        if merchant.ordered {
            merchant.ten_min.push_back((ts, customer_id.to_owned()));
            *merchant
                .ten_min_customer_counts
                .entry(customer_id.to_owned())
                .or_insert(0) += 1;
        }

        self.device_first_seen
            .entry(device_id.to_owned())
            .or_insert(ts);
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
            customer_states: HashMap::with_capacity(4096),
            device_states: HashMap::with_capacity(8192),
            merchant_states: HashMap::with_capacity(1024),
            device_first_seen: HashMap::with_capacity(8192),
        })
    }

    /// Scalar/reference feature read. This intentionally retains the original
    /// scan semantics and does not mutate cached chronological windows.
    fn features(
        &self,
        ts: f64,
        customer_id: &str,
        device_id: &str,
        merchant_id: &str,
        amount: f64,
    ) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let f = self.compute_features_scan(ts, customer_id, device_id, merchant_id, amount);
        (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8])
    }

    /// Compute pre-observation features using incremental chronological windows,
    /// then fold the current event into state in the same native call.
    fn features_and_observe(
        &mut self,
        ts: f64,
        customer_id: &str,
        device_id: &str,
        merchant_id: &str,
        amount: f64,
        mcc: i64,
    ) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let f = self.compute_features_fast(ts, customer_id, device_id, merchant_id, amount);
        self.observe_inner(ts, customer_id, device_id, merchant_id, amount, mcc);
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
        self.observe_inner(ts, customer_id, device_id, merchant_id, amount, mcc);
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
        let f = state.compute_features_scan(1_000.0, "C1", "D1", "M1", 100.0);
        assert_eq!(f[0], 0.0);
        assert!((f[2] - 1.0).abs() < 1e-5);
        assert_eq!(f[8], 0.0);
    }

    #[test]
    fn rolling_windows_and_distinct_customers_match_contract() {
        let mut state = RollingFeatureState::new(500).unwrap();
        state.observe_inner(900.0, "C1", "D1", "M1", 50.0, 5411);
        state.observe_inner(950.0, "C2", "D1", "M1", 75.0, 5732);
        state.observe_inner(980.0, "C1", "D2", "M1", 25.0, 5732);

        let scan = state.compute_features_scan(1_000.0, "C1", "D1", "M1", 100.0);
        let fast = state.compute_features_fast(1_000.0, "C1", "D1", "M1", 100.0);
        assert_eq!(scan, fast);
        assert_eq!(scan[0], 2.0);
        assert_eq!(scan[1], 75.0);
        assert_eq!(scan[3], 2.0);
        assert_eq!(scan[5], 2.0);
        assert_eq!(scan[6], 3.0);
        assert_eq!(scan[7], 2.0);
        assert_eq!(scan[8], 2.0);
    }

    #[test]
    fn maxlen_keeps_running_mean_and_windows_exact() {
        let mut state = RollingFeatureState::new(2).unwrap();
        state.observe_inner(1.0, "C1", "D1", "M1", 10.0, 1);
        state.observe_inner(2.0, "C1", "D1", "M1", 20.0, 2);
        state.observe_inner(3.0, "C1", "D1", "M1", 30.0, 3);
        let scan = state.compute_features_scan(4.0, "C1", "D1", "M1", 40.0);
        let fast = state.compute_features_fast(4.0, "C1", "D1", "M1", 40.0);
        assert_eq!(scan, fast);
        assert_eq!(scan[0], 2.0);
        assert_eq!(scan[1], 50.0);
        assert_eq!(scan[8], 2.0);
        assert!((scan[2] - (40.0 / 25.000001)).abs() < 1e-10);
    }

    #[test]
    fn chronological_expiry_fast_path_matches_scan() {
        let mut state = RollingFeatureState::new(100);
        let mut state = state.unwrap();
        for index in 0..50 {
            let ts = index as f64 * 120.0;
            state.observe_inner(ts, "C1", "D1", "M1", 10.0 + index as f64, index % 5);
        }
        let scan = state.compute_features_scan(6_100.0, "C1", "D1", "M1", 100.0);
        let fast = state.compute_features_fast(6_100.0, "C1", "D1", "M1", 100.0);
        assert_eq!(scan, fast);
    }

    #[test]
    fn out_of_order_events_preserve_full_scan_semantics() {
        let mut state = RollingFeatureState::new(10).unwrap();
        state.observe_inner(1_000.0, "C1", "D1", "M1", 10.0, 1);
        state.observe_inner(100.0, "C1", "D1", "M1", 20.0, 2); // marks state unordered
        state.observe_inner(950.0, "C1", "D1", "M1", 30.0, 3);

        let scan = state.compute_features_scan(1_000.0, "C1", "D1", "M1", 40.0);
        let fast = state.compute_features_fast(1_000.0, "C1", "D1", "M1", 40.0);
        assert_eq!(scan, fast);
        assert_eq!(scan[0], 2.0); // ts=100 is outside 10m; 1000 and 950 are inside
        assert_eq!(scan[3], 3.0); // original <= semantics include all three in 1h
        assert_eq!(scan[5], 2.0);
        assert_eq!(scan[6], 2.0);
    }
}
