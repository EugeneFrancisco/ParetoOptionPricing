from dataclasses import dataclass

@dataclass
class state:
    '''
    Represents a state in the MDP
    '''
    time: int # the time till expiration
    price: float