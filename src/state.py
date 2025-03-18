from dataclasses import dataclass

@dataclass
class state:
    '''
    Represents a state in the MDP
    '''
    time: int = 0 # the time till expiration
    price: float = 0
    terminal: bool = False