from __future__ import annotations

import uuid

from fastapi import Header, HTTPException, status


def err(code: str, message: str, status_code: int = status.HTTP_400_BAD_REQUEST) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def parse_bearer_grab_id(authorization: str | None) -> uuid.UUID:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise err("unauthorized", "Missing or invalid Authorization header (expected Bearer <grab_id>).", 401)
    token = authorization.split(" ", 1)[1].strip()
    try:
        return uuid.UUID(token)
    except ValueError as e:
        raise err("invalid_token", "grab_id must be a UUID.", 401) from e


def require_self_grab_id(
    authorization: str | None = Header(default=None),
) -> uuid.UUID:
    return parse_bearer_grab_id(authorization)


def require_matching_grab(
    grab_id: uuid.UUID,
    authorization: str | None = Header(default=None),
) -> uuid.UUID:
    token_id = parse_bearer_grab_id(authorization)
    if token_id != grab_id:
        raise err("forbidden", "Bearer grab_id does not match the requested resource.", 403)
    return token_id
