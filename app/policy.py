# policy.py

class Decision:
    def __init__(self, origin, context=None):
        self.origin = origin
        self.context = context or []


def decide(route, personal_hits, academic_hits):
    # 1️perosnal has the highest priority
    if personal_hits:
        return Decision("personal", personal_hits)

    #then Academic
    if academic_hits:
        return Decision("academic", academic_hits)

    #else Fallback
    return Decision("general")


##								       ##
#this module is not in use but can be used for extended behavior control#
##								       ##
