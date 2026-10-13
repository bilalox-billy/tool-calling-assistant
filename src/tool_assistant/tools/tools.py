# src/level2/tools.py
"""
Tool definitions with Pydantic input schemas.
Each tool: schema, validate, execute.
"""
import re
import math
from datetime import datetime, date
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


# ── Tool input schemas ────────────────────────────────────────────────────────

class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        min_length=1,
        max_length=256,
        description="A safe mathematical expression using digits and operators: + - * / ( ) ."
    )
    precision: int = Field(default=2, ge=0, le=10, description="Decimal places")

    @classmethod
    def is_safe(cls, expression: str) -> bool:
        """Only allow digits, spaces, and basic operators."""
        return bool(re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression))


class WeatherInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str = Field(min_length=1, max_length=100, description="City name")
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit"
    )


class DateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["today", "days_until", "days_since", "day_of_week"] = Field(
        description="Date operation to perform"
    )
    target_date: str | None = Field(
        default=None,
        description="Target date in YYYY-MM-DD format (required for days_until, days_since, day_of_week)"
    )


# ── Tool implementations ──────────────────────────────────────────────────────

class CalculatorTool:
    name = "calculate"
    description = (
        "Evaluate a mathematical expression and return the numeric result. "
        "Use this when the user asks to calculate, compute, or do arithmetic. "
        "Input: a safe expression like '12 * 8' or '(3 + 4) / 2'. Returns a number."
    )
    input_schema = CalculatorInput

    def get_ollama_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": CalculatorInput.model_json_schema(),
            }
        }

    def execute(self, validated: CalculatorInput) -> str:
        if not CalculatorInput.is_safe(validated.expression):
            return f"Error: Expression contains unsafe characters: {validated.expression!r}"
        try:
            result = eval(validated.expression)  # safe after regex check
            return str(round(float(result), validated.precision))
        except ZeroDivisionError:
            return "Error: Cannot divide by zero"
        except Exception as e:
            return f"Error: Could not evaluate expression: {e}"


class WeatherTool:
    """Simulated weather tool — returns fake but consistent data."""
    name = "get_weather"
    description = (
        "Get the current weather conditions for a specified city. "
        "Use when the user asks about weather, temperature, or climate. "
        "Returns temperature and weather condition."
    )
    input_schema = WeatherInput

    # Simulated weather data
    WEATHER_DB = {
        "paris": (22.0, "partly cloudy"),
        "london": (15.0, "overcast"),
        "tokyo": (28.0, "sunny"),
        "new york": (18.0, "clear"),
        "berlin": (12.0, "rainy"),
    }

    def get_ollama_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": WeatherInput.model_json_schema(),
            }
        }

    def execute(self, validated: WeatherInput) -> str:
        city_key = validated.city.lower()
        temp_c, condition = self.WEATHER_DB.get(city_key, (20.0, "unknown"))

        if validated.unit == "fahrenheit":
            temp = round(temp_c * 9 / 5 + 32, 1)
            unit_str = "°F"
        else:
            temp = temp_c
            unit_str = "°C"

        import json
        return json.dumps({
            "city": validated.city,
            "temperature": temp,
            "unit": unit_str,
            "condition": condition,
        })


class DateTool:
    name = "get_date"
    description = (
        "Perform date operations: get today's date, calculate days until/since a date, "
        "or get the day of week. Use when users ask about dates or time differences."
    )
    input_schema = DateInput

    def get_ollama_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": DateInput.model_json_schema(),
            }
        }

    def execute(self, validated: DateInput) -> str:
        today = date.today()

        if validated.operation == "today":
            return today.strftime("%A, %B %d, %Y")

        if validated.target_date is None:
            return "Error: target_date is required for this operation"

        try:
            target = date.fromisoformat(validated.target_date)
        except ValueError:
            return f"Error: Invalid date format. Use YYYY-MM-DD, got: {validated.target_date!r}"

        if validated.operation == "days_until":
            delta = (target - today).days
            if delta < 0:
                return f"{abs(delta)} days ago (date has passed)"
            return f"{delta} days until {target.strftime('%B %d, %Y')}"

        if validated.operation == "days_since":
            delta = (today - target).days
            return f"{delta} days since {target.strftime('%B %d, %Y')}"

        if validated.operation == "day_of_week":
            return target.strftime("%A")

        return "Error: Unknown operation"
