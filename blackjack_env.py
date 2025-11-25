import random
from enum import Enum
from typing import Tuple, Dict


class Action(Enum):
    HIT = 0
    STAND = 1


class BlackjackEnv:
    
    def __init__(self):
        self.num_states = 18
        self.num_actions = 2
        
        self.state_mapping = {i: i - 4 for i in range(4, 22)}
        self.reverse_state_mapping = {i - 4: i for i in range(4, 22)}
        
        self.reset()
        
    def reset(self) -> int:
        self.player_hand = self._draw_card() + self._draw_card()
        self.dealer_hand = self._draw_card()
        self.dealer_hidden = self._draw_card()
        self.done = False
        
        return self._get_state()
    
    def _draw_card(self) -> int:
        card = random.randint(1, 13)
        if card > 10:
            return 10
        return card
    
    def _get_state(self) -> int:
        player_value = self._calculate_hand_value(self.player_hand)
        
        if player_value < 4:
            player_value = 4
        elif player_value > 21:
            player_value = 21
        
        return self.state_mapping.get(player_value, 0)
    
    def _calculate_hand_value(self, hand: int) -> int:
        return hand
    
    def step(self, action: Action) -> Tuple[int, float, bool, Dict]:
        if self.done:
            raise ValueError("Episódio já terminou. Chame reset() primeiro.")
        
        reward = 0.0
        info = {}
        
        if action == Action.HIT:
            new_card = self._draw_card()
            self.player_hand += new_card
            
            if self.player_hand > 21:
                self.done = True
                reward = -1.0
                info['result'] = 'bust'
            else:
                reward = 0.0
                
        elif action == Action.STAND:
            self.done = True
            
            dealer_total = self.dealer_hand + self.dealer_hidden
            while dealer_total < 17:
                dealer_total += self._draw_card()
                if dealer_total > 21:
                    dealer_total = -1
                    break
            
            if dealer_total == -1:
                reward = 1.0
                info['result'] = 'dealer_bust'
            elif dealer_total > self.player_hand:
                reward = -1.0
                info['result'] = 'dealer_wins'
            elif dealer_total < self.player_hand:
                reward = 1.0
                info['result'] = 'player_wins'
            else:
                reward = 0.0
                info['result'] = 'tie'
        
        new_state = self._get_state()
        
        info['player_hand'] = self.player_hand
        info['dealer_showing'] = self.dealer_hand
        if self.done:
            info['dealer_total'] = dealer_total if action == Action.STAND else None
        
        return new_state, reward, self.done, info
    
    def get_state_value(self, state: int) -> int:
        return self.reverse_state_mapping.get(state, 4)
    
    def render(self) -> str:
        dealer_total = self.dealer_hand + self.dealer_hidden if self.done else None
        result = f"Jogador: {self.player_hand}"
        if self.done and dealer_total is not None:
            result += f" | Dealer: {dealer_total}"
        else:
            result += f" | Dealer mostra: {self.dealer_hand}"
        return result

