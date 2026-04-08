"""Processors module for PharmaScan."""

from pharmascan.processors.counter_verification import (
    generate_counter_verification_xlsx,
)
from pharmascan.processors.data_loader import (
    calculate_summary_stats,
    detect_rapid_revisits,
    detect_repeat_visits,
    load_and_process,
    load_data,
    normalize_column_names,
    parse_dates,
)
from pharmascan.processors.data_prep import (
    apply_column_mapping,
    auto_map_columns,
    get_group_colors,
    get_groups_order,
    get_system_fields,
    profile_column,
    score_column_vs_field,
)
from pharmascan.processors.rules_engine import (
    get_dx_drug_blacklist,
    get_prescriber_allowed,
    run_rules_engine,
)

__all__ = [
    "load_data",
    "normalize_column_names",
    "parse_dates",
    "calculate_summary_stats",
    "detect_repeat_visits",
    "detect_rapid_revisits",
    "load_and_process",
    "generate_counter_verification_xlsx",
    "profile_column",
    "score_column_vs_field",
    "auto_map_columns",
    "apply_column_mapping",
    "get_system_fields",
    "get_groups_order",
    "get_group_colors",
    "run_rules_engine",
    "get_dx_drug_blacklist",
    "get_prescriber_allowed",
]
