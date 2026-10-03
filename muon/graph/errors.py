class GraphError(Exception):
    """Base for every error the graph layer raises."""


class GraphConnectionError(GraphError):
    """The database could not be reached or refused the credentials."""


class GraphQueryError(GraphError):
    """The database rejected a query."""
