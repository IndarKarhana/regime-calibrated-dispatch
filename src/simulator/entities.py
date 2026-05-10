"""Core data structures for the ride-hailing simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class DriverStatus(Enum):
    IDLE = auto()
    EN_ROUTE_PICKUP = auto()
    IN_TRIP = auto()


@dataclass
class RideRequest:
    id: int
    time: float  # seconds from sim start
    pickup_lon: float
    pickup_lat: float
    dropoff_lon: float
    dropoff_lat: float
    assigned: bool = False
    picked_up: bool = False
    completed: bool = False
    expired: bool = False
    wait_time: float = 0.0  # seconds until pickup


@dataclass
class Driver:
    id: int
    lon: float
    lat: float
    status: DriverStatus = DriverStatus.IDLE
    current_request: RideRequest | None = None
    remaining_seconds: float = 0.0  # time to finish current leg
    idle_seconds: float = 0.0
    dest_lon: float = 0.0
    dest_lat: float = 0.0


@dataclass
class Assignment:
    driver_id: int
    request_id: int
    pickup_time_est: float  # estimated seconds to reach pickup


@dataclass
class RepositionInstruction:
    driver_id: int
    target_lon: float
    target_lat: float


@dataclass
class MetricsAccumulator:
    total_requests: int = 0
    completed_trips: int = 0
    expired_requests: int = 0
    total_wait_seconds: float = 0.0
    total_pickup_distance_m: float = 0.0
    total_reposition_distance_m: float = 0.0
    total_reposition_seconds: float = 0.0
    reposition_legs: int = 0
    total_idle_seconds: float = 0.0
    wait_times: list[float] = field(default_factory=list)
    wait_zone_ids: list[str] = field(default_factory=list)


@dataclass
class SimState:
    time: float = 0.0  # current sim time in seconds
    drivers: list[Driver] = field(default_factory=list)
    pending_requests: list[RideRequest] = field(default_factory=list)
    active_trips: list[RideRequest] = field(default_factory=list)
    metrics: MetricsAccumulator = field(default_factory=MetricsAccumulator)
