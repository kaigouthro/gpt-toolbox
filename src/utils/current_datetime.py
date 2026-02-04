from datetime import datetime
import pytz

def current_datetime():
    """
    Returns the current datetime in the local timezone. 
    (Generally useful for Agent predictions that require the current datetime.
    """
    # Localize the current datetime
    local_timezone = pytz.timezone('America/New_York') # TODO get from env

    localized_datetime = local_timezone.localize(datetime.now())

    return localized_datetime.strftime('%B-%d-%Y %H:%M:%S %Z (UTC%z)')
