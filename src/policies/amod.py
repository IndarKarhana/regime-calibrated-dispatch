"""AMoD-oriented repositioning extensions.

The simulator does not model battery state natively, so this module keeps a
policy-local charge ledger for a small centralized-fleet experiment. It is meant
as an emerging-technology implication check, not a full EV-fleet simulator.
"""

from __future__ import annotations

import numpy as np

from src.policies.anticipatory import _build_zone_fractions, _HexGrid
from src.policies.external_baselines import _idle_state_by_zone, _transport_lp_reposition
from src.simulator.entities import DriverStatus, RepositionInstruction, SimState
from src.simulator.routing import RoutingClient, _haversine_m


class ChargingAwareShareLPReposition:
    """Share-target LP with simple charge constraints for centralized AMoD.

    Drivers carry a policy-local charge fraction. Idle low-charge vehicles are
    forced to the nearest charging zone before the remaining idle supply is
    balanced toward forecast demand shares.
    """

    def __init__(
        self,
        prior,
        *,
        max_move_fraction: float = 0.30,
        charge_threshold: float = 0.18,
        low_charge_start_fraction: float = 0.12,
        drive_consumption_per_hour: float = 0.16,
        idle_consumption_per_hour: float = 0.015,
        charge_rate_per_hour: float = 0.55,
        h3_res: int | None = None,
        bbox: dict | None = None,
    ):
        self._grid = _HexGrid(h3_res, bbox=bbox)
        self._zone_fractions = _build_zone_fractions(prior, self._grid)
        self._max_move_fraction = max_move_fraction
        self._charge_threshold = charge_threshold
        self._low_charge_start_fraction = low_charge_start_fraction
        self._drive_consumption_per_hour = drive_consumption_per_hour
        self._idle_consumption_per_hour = idle_consumption_per_hour
        self._charge_rate_per_hour = charge_rate_per_hour
        self._charge: dict[int, float] = {}
        self._last_time: float | None = None
        self._cost_cache: dict[tuple[int, int], float] = {}
        self._charger_zones = self._default_charger_zones()
        self.low_charge_observations = 0
        self.forced_charging_moves = 0
        self.reposition_calls = 0

    def _default_charger_zones(self) -> list[int]:
        anchors = [
            (-74.010, 40.715),
            (-73.985, 40.755),
            (-73.955, 40.805),
        ]
        zones = []
        for lon, lat in anchors:
            distances = [
                _haversine_m(self._grid.hex_centers[hx], (lon, lat))
                for hx in self._grid.hex_ids
            ]
            zones.append(int(np.argmin(distances)))
        return sorted(set(zones))

    def _nearest_charger(self, zone_idx: int) -> int:
        src_center = self._grid.hex_centers[self._grid.hex_ids[zone_idx]]
        distances = [
            _haversine_m(src_center, self._grid.hex_centers[self._grid.hex_ids[cz]])
            for cz in self._charger_zones
        ]
        return self._charger_zones[int(np.argmin(distances))]

    def _ensure_charge_state(self, state: SimState) -> None:
        for driver in state.drivers:
            if driver.id not in self._charge:
                base = 1.0
                if driver.id % max(int(1.0 / self._low_charge_start_fraction), 1) == 0:
                    base = self._charge_threshold * 0.8
                self._charge[driver.id] = base

    def _update_charge(self, state: SimState) -> None:
        self._ensure_charge_state(state)
        if self._last_time is None:
            self._last_time = state.time
            return
        dt_h = max(state.time - self._last_time, 0.0) / 3600.0
        self._last_time = state.time
        for driver in state.drivers:
            charge = self._charge.get(driver.id, 1.0)
            if driver.status == DriverStatus.IDLE:
                zone = self._grid.locate(driver.lat, driver.lon)
                if zone in self._charger_zones:
                    charge += self._charge_rate_per_hour * dt_h
                else:
                    charge -= self._idle_consumption_per_hour * dt_h
            else:
                charge -= self._drive_consumption_per_hour * dt_h
            self._charge[driver.id] = float(np.clip(charge, 0.0, 1.0))

    def charging_summary(self) -> dict[str, float]:
        calls = max(self.reposition_calls, 1)
        return {
            "charge_violation_rate": self.low_charge_observations / calls,
            "forced_charging_moves": float(self.forced_charging_moves),
            "charger_zone_count": float(len(self._charger_zones)),
        }

    def reposition(
        self,
        state: SimState,
        router: RoutingClient,
    ) -> list[RepositionInstruction]:
        self.reposition_calls += 1
        self._update_charge(state)
        idle_per_zone, idle_drivers = _idle_state_by_zone(state, self._grid)
        total_idle = int(idle_per_zone.sum())
        if total_idle <= 0:
            return []

        instructions: list[RepositionInstruction] = []
        low_charge_ids: set[int] = set()
        for zone, driver_ids in list(idle_drivers.items()):
            for driver_id in list(driver_ids):
                if self._charge.get(driver_id, 1.0) >= self._charge_threshold:
                    continue
                low_charge_ids.add(driver_id)
                self.low_charge_observations += 1
                charger = self._nearest_charger(zone)
                lon, lat = self._grid.hex_centers[self._grid.hex_ids[charger]]
                instructions.append(RepositionInstruction(
                    driver_id=driver_id,
                    target_lon=lon,
                    target_lat=lat,
                ))
                self.forced_charging_moves += 1
                idle_per_zone[zone] = max(idle_per_zone[zone] - 1.0, 0.0)

        for zone in list(idle_drivers):
            idle_drivers[zone] = [
                driver_id for driver_id in idle_drivers[zone]
                if driver_id not in low_charge_ids
            ]

        remaining_idle = int(idle_per_zone.sum())
        if remaining_idle <= 0:
            return instructions

        target_idle = remaining_idle * self._zone_fractions
        surplus = np.maximum(idle_per_zone - target_idle, 0.0)
        deficit = np.maximum(target_idle - idle_per_zone, 0.0)
        instructions.extend(_transport_lp_reposition(
            self._grid,
            idle_drivers,
            surplus,
            deficit,
            max_moves=int(remaining_idle * self._max_move_fraction),
            router=router,
            cost_cache=self._cost_cache,
            rng=np.random.default_rng(int(state.time) + 211),
        ))
        return instructions
