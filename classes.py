import enum
from typing import List


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