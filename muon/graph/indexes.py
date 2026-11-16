"""Names of the full-text index on entity names.

Hazard: ingestion creates the index and retrieval queries it, and both read these
constants. Change a value here and recreate the index on every database.
"""

NAME_INDEX = "entity_names"
NAME_INDEX_LABELS = ("Entity", "Work", "Series")
NAME_FIELD = "name"
ALIASES_FIELD = "aliases"
