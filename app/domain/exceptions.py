class DomainException(Exception):
    pass


class ConnectionException(DomainException):
    pass


class PublishException(DomainException):
    pass


class ConsumeException(DomainException):
    pass
