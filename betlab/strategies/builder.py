from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field


@dataclass
class StrategyFilter:
    field: str
    operator: str
    value: any


@dataclass
class StrategySignal:
    field: str
    operator: str
    value: any


@dataclass
class StrategySelection:
    side: str
    runner: str | None = None


@dataclass
class StrategyPrice:
    source: str
    minimum: float | None = None
    maximum: float | None = None


@dataclass
class StrategyStaking:
    method: str = "flat"
    kwargs: dict = field(default_factory=dict)


@dataclass
class StrategyRisk:
    max_open_exposure: float | None = None
    max_daily_loss: float | None = None


@dataclass
class StrategyValidation:
    method: str = "walk_forward"
    training_window: int | None = None
    test_window: int | None = None
    step: int | None = None


@dataclass
class StrategyDefinition:
    id: str
    name: str
    sport: str
    market: str
    filters: list[StrategyFilter] = field(default_factory=list)
    signal: StrategySignal | None = None
    selection: StrategySelection | None = None
    price: StrategyPrice | None = None
    staking: StrategyStaking = field(default_factory=StrategyStaking)
    risk: StrategyRisk = field(default_factory=StrategyRisk)
    validation: StrategyValidation = field(default_factory=StrategyValidation)
    created_at: str = ""
    version: int = 1


class StrategyBuilder:
    def __init__(self):
        self._id = uuid.uuid4().hex[:12]
        self._name = ""
        self._sport = ""
        self._market = ""
        self._filters: list[StrategyFilter] = []
        self._signal: StrategySignal | None = None
        self._selection: StrategySelection | None = None
        self._price: StrategyPrice | None = None
        self._staking = StrategyStaking()
        self._risk = StrategyRisk()
        self._validation = StrategyValidation()

    def set_name(self, name: str) -> StrategyBuilder:
        self._name = name
        return self

    def set_sport(self, sport: str) -> StrategyBuilder:
        self._sport = sport
        return self

    def set_market(self, market: str) -> StrategyBuilder:
        self._market = market
        return self

    def add_filter(self, field: str, operator: str, value: any) -> StrategyBuilder:
        self._filters.append(StrategyFilter(field=field, operator=operator, value=value))
        return self

    def set_signal(self, field: str, operator: str, value: any) -> StrategyBuilder:
        self._signal = StrategySignal(field=field, operator=operator, value=value)
        return self

    def set_selection(self, side: str, runner: str | None = None) -> StrategyBuilder:
        self._selection = StrategySelection(side=side, runner=runner)
        return self

    def set_price(self, source: str, minimum: float | None = None, maximum: float | None = None) -> StrategyBuilder:
        self._price = StrategyPrice(source=source, minimum=minimum, maximum=maximum)
        return self

    def set_staking(self, method: str = "flat", **kwargs) -> StrategyBuilder:
        self._staking = StrategyStaking(method=method, kwargs=kwargs)
        return self

    def set_risk(self, max_open_exposure: float | None = None, max_daily_loss: float | None = None) -> StrategyBuilder:
        self._risk = StrategyRisk(max_open_exposure=max_open_exposure, max_daily_loss=max_daily_loss)
        return self

    def set_validation(
        self,
        method: str = "walk_forward",
        training_window: int | None = None,
        test_window: int | None = None,
        step: int | None = None,
    ) -> StrategyBuilder:
        self._validation = StrategyValidation(
            method=method,
            training_window=training_window,
            test_window=test_window,
            step=step,
        )
        return self

    def build(self) -> StrategyDefinition:
        from datetime import datetime

        return StrategyDefinition(
            id=self._id,
            name=self._name,
            sport=self._sport,
            market=self._market,
            filters=list(self._filters),
            signal=self._signal,
            selection=self._selection,
            price=self._price,
            staking=self._staking,
            risk=self._risk,
            validation=self._validation,
            created_at=datetime.utcnow().isoformat(),
        )

    def to_json(self) -> str:
        return json.dumps(self._to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> StrategyBuilder:
        data = json.loads(json_str)
        builder = cls()
        builder._id = data.get("id", builder._id)
        builder._name = data.get("name", "")
        builder._sport = data.get("sport", "")
        builder._market = data.get("market", "")
        builder._filters = [StrategyFilter(**f) for f in data.get("filters", [])]
        sig = data.get("signal")
        builder._signal = StrategySignal(**sig) if sig else None
        sel = data.get("selection")
        builder._selection = StrategySelection(**sel) if sel else None
        pr = data.get("price")
        builder._price = StrategyPrice(**pr) if pr else None
        stk = data.get("staking", {})
        builder._staking = StrategyStaking(
            method=stk.get("method", "flat"),
            kwargs=stk.get("kwargs", {}),
        )
        risk = data.get("risk", {})
        builder._risk = StrategyRisk(
            max_open_exposure=risk.get("max_open_exposure"),
            max_daily_loss=risk.get("max_daily_loss"),
        )
        val = data.get("validation", {})
        builder._validation = StrategyValidation(
            method=val.get("method", "walk_forward"),
            training_window=val.get("training_window"),
            test_window=val.get("test_window"),
            step=val.get("step"),
        )
        return builder

    def _to_dict(self) -> dict:
        return {
            "id": self._id,
            "name": self._name,
            "sport": self._sport,
            "market": self._market,
            "filters": [
                {"field": f.field, "operator": f.operator, "value": f.value}
                for f in self._filters
            ],
            "signal": (
                {"field": self._signal.field, "operator": self._signal.operator, "value": self._signal.value}
                if self._signal
                else None
            ),
            "selection": (
                {"side": self._selection.side, "runner": self._selection.runner}
                if self._selection
                else None
            ),
            "price": (
                {
                    "source": self._price.source,
                    "minimum": self._price.minimum,
                    "maximum": self._price.maximum,
                }
                if self._price
                else None
            ),
            "staking": {
                "method": self._staking.method,
                "kwargs": self._staking.kwargs,
            },
            "risk": {
                "max_open_exposure": self._risk.max_open_exposure,
                "max_daily_loss": self._risk.max_daily_loss,
            },
            "validation": {
                "method": self._validation.method,
                "training_window": self._validation.training_window,
                "test_window": self._validation.test_window,
                "step": self._validation.step,
            },
        }
