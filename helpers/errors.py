from discord.app_commands import AppCommandError


class HordeAPIError(AppCommandError):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code

class ValidationError(HordeAPIError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code)

class InvalidAPIKeyError(HordeAPIError):
    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message, status_code)

class RequestNotfoundError(HordeAPIError):
    def __init__(self, message: str, status_code: int = 404):
        super().__init__(message, status_code)

class TooManyPromptsError(HordeAPIError):
    def __init__(self, message: str, status_code: int = 429):
        super().__init__(message, status_code)

class MaintenanceModeError(HordeAPIError):
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message, status_code)

class RequestError(HordeAPIError):
    def __new__(cls, message: str, status_code: int) -> HordeAPIError:
        match status_code:
            case 400:
                return ValidationError(message, status_code)
            case 401:
                return InvalidAPIKeyError(message, status_code)
            case 404:
                return RequestNotfoundError(message, status_code)
            case 429:
                return TooManyPromptsError(message, status_code)
            case 503:
                return MaintenanceModeError(message, status_code)
            case _:
                return HordeAPIError(message, status_code)

class InsufficientKudosError(AppCommandError):
    ...

class FaultedGenerationError(AppCommandError):
    ...

class MissingGenerationsError(AppCommandError):
    ...

class GenerationTimeoutError(AppCommandError):
    ...