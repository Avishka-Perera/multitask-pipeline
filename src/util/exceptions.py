class DatasetCorruptionError(Exception):
    """Exception raised for dataset corruption issues."""

    def __init__(self, message="Dataset corruption detected"):
        self.message = message
        super().__init__(self.message)
