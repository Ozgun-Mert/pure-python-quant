import enum
from typing import List
from dataclasses import dataclass


"""
Data model classes for the Pure Python Quant trading rule discovery system.

This module defines the core data structures used throughout the project:

1. Type (Enum): Defines trading signal direction (HIGH/bullish or LOW/bearish)
2. Rule (Class): Represents a single conditional trading rule (e.g., RSI > 70)
3. Value (Class): Represents a complete discovered trading signal with multiple rules and metrics

These classes work together to structure discovered trading signals. A Value object contains
multiple Rule objects, and each Rule defines one condition. Together, multiple Rule objects
form a complete multi-condition trading signal with associated performance metrics.

Example: A Value for a bullish signal on 3-day outlook might contain:
- Rules: [RSI > 70, SMA_20_Ratio > 0.05, Cross_20_50 == 1]
- Metrics: win_rate=0.72, support=145, percentage_profit=0.023
"""


class Type(enum.Enum):
    """
    Enumeration representing the direction of a trading signal.
    
    This enum defines the two fundamental types of predictions:
    - HIGH: Bullish signal indicating expected upward price movement
    - LOW: Bearish signal indicating expected downward price movement
    
    These types are used to classify trading rules discovered by the Gini Engine.
    Each rule is associated with either a HIGH (bullish) or LOW (bearish) prediction.
    The type determines how to interpret the rule conditions - a HIGH type rule
    suggests buying/long positions, while a LOW type rule suggests selling/short positions.
    """
    HIGH = "high"
    LOW = "low"


class Rule:
    """
    Represents a single conditional trading rule based on a technical indicator.
    
    A Rule encapsulates one condition in a trading signal, such as "RSI > 70" or "SMA_20_Ratio <= 0.05".
    Multiple Rule objects are combined together in a Value object to form complete multi-condition
    trading signals. Each rule defines:
    - Which feature (technical indicator) to test
    - What threshold value to compare against
    - Which comparison operator to use (==, <=, >, <, >=)
    
    The rule is evaluated by comparing the feature value from market data against the threshold
    using the specified operation. Rules are atomic building blocks that combine via AND logic
    (all rules in a combination must be true for the signal to trigger).
    """
    def __init__(self, feature: str, threshold: float, operation: str):
        """
        Initialize a trading rule with a feature, threshold, and operation.
        
        Parameters:
        - feature: Name of the technical indicator field (e.g., "RSI", "SMA_20_Ratio")
        - threshold: The comparison value for this rule (float)
        - operation: The comparison operator as a string ("==", "<=", ">", "<", ">=")
        """
        self._feature = feature
        self._threshold = threshold
        self._operation = operation

    def __str__(self) -> str:
        """
        Return a human-readable string representation of this rule.

        The output includes the feature name and threshold value, suitable for
        logging or display in user-facing prediction summaries.

        Returns:
            str: Formatted string like "Rule(Feature: RSI, Threshold: 70.0)".
        """
        return f"Rule(Feature: {self._feature}, Threshold: {self._threshold})"

    def __repr__(self) -> str:
        """
        Return a developer-oriented string representation of this rule.

        Used by debuggers and the interactive interpreter to show the rule's
        core identifying attributes in a compact, unambiguous format.

        Returns:
            str: Angle-bracket representation with feature and threshold.
        """
        return f"<Feature: {self._feature}, Threshold: {self._threshold}>"


    @property
    def feature(self) -> str:
        """Get the feature/indicator name for this rule."""
        return self._feature

    @feature.setter
    def feature(self, value: str):
        """Set the feature/indicator name for this rule."""
        self._feature = value

    @property
    def threshold(self) -> float:
        """Get the threshold value to compare against."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        """Set the threshold value to compare against."""
        self._threshold = value

    @property
    def operation(self) -> str:
        """Get the comparison operator (==, <=, >, <, >=)."""
        return self._operation

    @operation.setter
    def operation(self, value: str):
        """Set the comparison operator (==, <=, >, <, >=)."""
        self._operation = value
        
    def to_dict(self) -> dict:
        """
        Convert the Rule object to a dictionary for JSON serialization.
        
        Returns a dictionary containing the feature, threshold, and operation.
        This format matches the structure used by the Gini Engine for storing
        discovered rules in JSON files.
        
        Returns:
        - Dictionary with keys: "feature", "threshold", "operation"
        """
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "operation": self.operation
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Rule':
        """
        Reconstruct a Rule instance from a JSON-compatible dictionary.

        Expects a dictionary with exactly the keys produced by `to_dict()`:
        "feature", "threshold", and "operation". This is the inverse of
        serialization and is used when loading discovered rules from disk.

        Parameters:
            data: Dictionary containing rule fields as stored in JSON output files.

        Returns:
            Rule: A fully initialized Rule object with the deserialized values.
        """
        return cls(
            feature=data["feature"],
            threshold=data["threshold"],
            operation=data["operation"]
        )

class Value:
    """
    Represents a complete discovered trading signal/rule with its performance metrics.
    
    A Value object encapsulates a multi-condition trading rule that was discovered by the Gini Engine.
    It combines:
    - Multiple Rule objects (the actual conditions to check)
    - Performance metrics (win_rate, support, profit)
    - Metadata (day horizon, signal type, feature combination)
    
    Each Value represents one trading signal strategy discovered during the rule discovery process.
    For example, a Value might represent: "When Direction=1 AND RSI>70 AND SMA_20_Ratio>0.05,
    then the price has a 72% chance of going up in the next 3 days with an average profit of 2.3%".
    
    The metrics stored help evaluate the quality and reliability of the discovered signal:
    - win_rate: Historical success rate of this signal
    - support: How many historical occurrences support this signal
    - percentage_profit: Average profit when the signal triggers
    """
    def __init__(self, day: int, type: Type, combination: List[str], win_rate: float, support: int, percentage_profit: float, which_day_is_higher: float, rules: List[Rule]):
        """
        Initialize a trading signal value with rules and metrics.
        
        Parameters:
        - day: Time horizon in days for this signal (3, 90, 180, or 365)
        - type: Type enum indicating HIGH (bullish) or LOW (bearish) signal
        - combination: List of feature names used in this signal (e.g., ["RSI", "SMA_20_Ratio"])
        - win_rate: Historical win rate of this signal (0.0 to 1.0)
        - support: Number of historical instances when this signal triggered
        - percentage_profit: Average percentage profit when signal triggers
        - which_day_is_higher: Average day on which price exceeded signal threshold (if applicable)
        - rules: List of Rule objects that make up this complete signal
        """
        self._day = day
        self._type = type
        self._combination = combination
        self._win_rate = win_rate
        self._support = support
        self._percentage_profit = percentage_profit
        self._which_day_is_higher = which_day_is_higher
        self._rules = rules

    def __str__(self) -> str:
        """
        Return a concise, human-readable summary of this trading signal.

        Formats the signal direction, win rate, average profit, and historical
        support count as percentages for easy reading in console output.

        Returns:
            str: One-line summary of the signal's key performance metrics.
        """
        return f"Signal(Type: {self.type.name}, Win Ratio: %{self.win_rate*100:.1f}, Profit: %{self.percentage_profit*100:.1f}, Support: {self.support})"

    def __repr__(self) -> str:
        """
        Return a detailed developer-oriented representation of this signal.

        Includes type, win rate, profit, support count, and the full list of
        constituent Rule objects for debugging and interactive inspection.

        Returns:
            str: Multi-field angle-bracket representation of the Value object.
        """
        return f"<Value {self.type.name} | Win: {self.win_rate:.2f} | Profit: {self.percentage_profit:.2f}, Support: {self.support}, Rules: {self.rules} >\n"

    @property
    def day(self) -> int:
        """Get the time horizon in days for this signal's predictions."""
        return self._day

    @day.setter
    def day(self, value: int):
        """Set the time horizon in days for this signal's predictions."""
        self._day = value

    @property
    def type(self) -> Type:
        """Get the signal direction type (HIGH for bullish, LOW for bearish)."""
        return self._type

    @type.setter
    def type(self, value: Type):
        """Set the signal direction type (HIGH for bullish, LOW for bearish)."""
        self._type = value

    @property
    def combination(self) -> List[str]:
        """Get the list of feature names used in this signal."""
        return self._combination

    @combination.setter
    def combination(self, value: List[str]):
        """Set the list of feature names used in this signal."""
        self._combination = value

    @property
    def win_rate(self) -> float:
        """Get the historical win rate of this signal (0.0 to 1.0)."""
        return self._win_rate

    @win_rate.setter
    def win_rate(self, value: float):
        """Set the historical win rate of this signal (0.0 to 1.0)."""
        self._win_rate = value

    @property
    def support(self) -> int:
        """Get the number of historical instances when this signal triggered."""
        return self._support

    @support.setter
    def support(self, value: int):
        """Set the number of historical instances when this signal triggered."""
        self._support = value

    @property
    def percentage_profit(self) -> float:
        """Get the average percentage profit when this signal triggers."""
        return self._percentage_profit

    @percentage_profit.setter
    def percentage_profit(self, value: float):
        """Set the average percentage profit when this signal triggers."""
        self._percentage_profit = value

    @property
    def which_day_is_higher(self) -> float:
        """Get the average day on which the signal condition was first met."""
        return self._which_day_is_higher

    @which_day_is_higher.setter
    def which_day_is_higher(self, value: float):
        """Set the average day on which the signal condition was first met."""
        self._which_day_is_higher = value

    @property
    def rules(self) -> List[Rule]:
        """Get the list of Rule objects that make up this complete signal."""
        return self._rules

    @rules.setter
    def rules(self, value: List[Rule]):
        """Set the list of Rule objects that make up this complete signal."""
        self._rules = value

    def to_dict(self) -> dict:
        """
        Convert the Value object to a dictionary for JSON serialization.
        
        This method creates a complete representation of the trading signal suitable
        for saving to JSON files. It converts all attributes to JSON-serializable types,
        including converting the Type enum to its string value and converting all Rule
        objects to dictionaries.
        
        Returns a dictionary containing:
        - day: Time horizon in days
        - type: Signal type as string ("high" or "low")
        - combination: List of feature names
        - win_rate: Historical win rate as float
        - support: Number of historical instances as integer
        - percentage_profit: Average profit percentage as float
        - which_day_is_higher: Average signal trigger day as float
        - rules: List of rule dictionaries (each with feature, threshold, operation)
        
        Returns:
        - Dictionary with all signal information in JSON-compatible format
        """
        return {
            "day": self.day,
            "type": self.type.value,
            "combination": self.combination,
            "win_rate": self.win_rate,
            "support": self.support,
            "percentage_profit": self.percentage_profit,
            "which_day_is_higher": self.which_day_is_higher,
            "rules": [rule.to_dict() for rule in self.rules]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Value':
        """
        Reconstruct a Value instance from a JSON-compatible dictionary.

        Deserializes all scalar fields directly, converts the "type" string
        back into a Type enum member, and recursively builds Rule objects from
        the nested "rules" list. This is the inverse of `to_dict()` and is
        used when loading Gini Engine output files.

        Parameters:
            data: Dictionary containing all Value fields as stored in JSON files.
                  Must include keys: day, type, combination, win_rate, support,
                  percentage_profit, which_day_is_higher, and rules.

        Returns:
            Value: A fully initialized Value object with deserialized rules.
        """
        signal_type = Type(data["type"])
        
        rules_list = [Rule.from_dict(rule_data) for rule_data in data["rules"]]
        
        return cls(
            day=data["day"],
            type=signal_type,
            combination=data["combination"],
            win_rate=data["win_rate"],
            support=data["support"],
            percentage_profit=data["percentage_profit"],
            which_day_is_higher=data["which_day_is_higher"],
            rules=rules_list
        )
@dataclass
class Prediction:
    """
    Represents a live trading prediction generated for a specific ticker and date.

    A Prediction is produced when all conditions of a champion trading signal
    (discovered by the Gini Engine and selected by best_signal_finder) are met
    on a given day's market data. It bundles the signal direction, estimated
    probability of success, expected price change, time horizon, and the list
    of rules that fired.

    Attributes:
        ticker: Stock symbol (e.g., "aapl") for which the prediction applies.
        date: ISO date string (YYYY-MM-DD) of the market day being evaluated.
        signal_type: Direction label — "HIGH" for bullish or "LOW" for bearish.
        probability: Estimated success probability, capped at 0.8, derived from
                     the champion signal's historical win rate.
        expected_change_pct: Expected percentage price change over the signal's
                             horizon, taken from the champion's average profit.
        expected_days: Prediction time horizon in days (3, 90, 180, or 365).
        rules_matched: Human-readable string representations of each Rule that
                       evaluated to True on the target date.
    """
    ticker: str
    date: str
    signal_type: str
    probability: float
    expected_change_pct: float
    expected_days: int
    rules_matched: List[str]

    def __str__(self) -> str:
        """
        Return a formatted, user-friendly summary of this prediction.

        Displays the date, ticker, time horizon, signal direction, probability,
        and expected price move as a single readable line suitable for console
        output in predictor.py.

        Returns:
            str: Formatted prediction string with percentage-formatted metrics.
        """
        direction = "Bullish (+)" if self.signal_type == "HIGH" else "Bearish (-)"
        return (f"[{self.date}] {self.ticker.upper()} | {self.expected_days}-Day Outlook | "
                f"Signal: {direction} | Prob: {self.probability * 100:.1f}% | "
                f"Expected Move: {self.expected_change_pct * 100:.2f}%")