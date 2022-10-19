import os
from functools import wraps
from fastapi import HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED

X_API_KEY = os.getenv("X_API_KEY", "MYAPIKEY")


def authorize(f):
    @wraps(f)
    def decorated_function(*args, **kws):
        if kws["header"] != X_API_KEY:
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED,
                                detail='wrong api key',
                                headers={"WWW-Authenticate": "Bearer"})

        return f(*args, **kws)

    return decorated_function


def handle_error(f):
    @wraps(f)
    def decorated_function(*args, **kws):
        try:
            resp = f(*args, **kws)
            return resp
        except Exception as ex:
            import traceback
            traceback = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
            print(traceback)
            return HTTPException(status_code=420,
                                 detail=str(ex))

    return decorated_function
