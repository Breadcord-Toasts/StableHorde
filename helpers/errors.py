class GenericHordeError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class ValidationError(GenericHordeError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code)

class InvalidAPIKey(GenericHordeError):
    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message, status_code)

class RequestNotfound(GenericHordeError):
    def __init__(self, message: str, status_code: int = 404):
        super().__init__(message, status_code)

class TooManyPrompts(GenericHordeError):
    def __init__(self, message: str, status_code: int = 429):
        super().__init__(message, status_code)

class MaintenanceMode(GenericHordeError):
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message, status_code)

class RequestError(GenericHordeError):
    def __new__(cls, message: str, status_code: int) -> GenericHordeError:
        match status_code:
            case 400:
                return ValidationError(message, status_code)
            case 401:
                return InvalidAPIKey(message, status_code)
            case 404:
                return RequestNotfound(message, status_code)
            case 429:
                return TooManyPrompts(message, status_code)
            case 503:
                return MaintenanceMode(message, status_code)
            case _:
                return GenericHordeError(message, status_code)


