import pytest

from muon.api.errors import derive_error_code


@pytest.mark.parametrize(
    ("status_code", "code"),
    [
        (404, "not_found"),
        (405, "method_not_allowed"),
        (422, "validation_error"),
        (500, "internal_server_error"),
    ],
)
def test_derive_error_code_uses_the_status_phrase(status_code: int, code: str) -> None:
    assert derive_error_code(status_code) == code
