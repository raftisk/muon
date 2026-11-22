"""Auth routes. Auth is off: every caller is the anonymous user and no cookie is set."""

from fastapi import APIRouter, Response
from pydantic import BaseModel

NO_CONTENT = 204


class CurrentUser(BaseModel):
    id: str
    name: str


ANONYMOUS_USER = CurrentUser(id="anonymous", name="Anonymous")


def create_auth_router() -> APIRouter:
    router = APIRouter(prefix="/auth")

    @router.post("/login", status_code=NO_CONTENT)
    async def login() -> Response:
        return Response(status_code=NO_CONTENT)

    @router.post("/logout", status_code=NO_CONTENT)
    async def logout() -> Response:
        return Response(status_code=NO_CONTENT)

    @router.get("/me")
    async def get_current_user() -> CurrentUser:
        return ANONYMOUS_USER

    return router
