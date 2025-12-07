import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:56:50.392162
# Source Brief: brief_00761.md
# Brief Index: 761
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "A turn-based, deck-building card game. Battle through waves of digital enemies by strategically playing attack and defense cards."
    )
    user_guide = (
        "Controls: ←→ arrows to select a card. Press space to play the selected card or shift to discard it for momentum. Doing nothing ends your turn."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 12)
        self.font_medium = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Visual & Theming ---
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_GRID = (20, 40, 70)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_TEXT_DIM = (150, 150, 180)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_ENEMY = (255, 0, 128)
        self.COLOR_MOMENTUM = (255, 255, 0)
        self.COLOR_HEALTH = (0, 255, 128)
        self.COLOR_BLOCK = (100, 150, 255)
        self.COLOR_ATTACK = (255, 100, 100)
        self.COLOR_CARD_BG = (25, 50, 90)
        self.COLOR_CARD_BORDER = (50, 90, 160)
        self.COLOR_CARD_SELECTED = (255, 255, 0)

        # --- Game Constants ---
        self.MAX_STEPS = 1000
        self.MAX_WAVES = 10
        self.PLAYER_MAX_HP = 50
        self.PLAYER_STARTING_MOMENTUM = 3
        self.PLAYER_MAX_MOMENTUM = 10
        self.PLAYER_MOMENTUM_PER_TURN = 2
        self.PLAYER_STARTING_HAND_SIZE = 5
        self.PLAYER_CARDS_PER_TURN = 1
        self.ENEMY_BASE_HP = 20
        self.ENEMY_BASE_ATTACK = 2
        self.ENEMY_BASE_HAND_SIZE = 3

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.game_over_status = "" # "WIN", "LOSE", "TIMEOUT"
        self.game_phase = "PLAYER_TURN" # PLAYER_TURN, ACTION_RESOLUTION, ENEMY_TURN, WAVE_CLEARED, GAME_OVER
        
        self.player_hp = 0
        self.player_block = 0
        self.player_momentum = 0
        self.player_deck = []
        self.player_hand = []
        self.player_discard = []
        self.selected_card_idx = 0
        
        self.enemy_hp = 0
        self.enemy_max_hp = 0
        self.enemy_block = 0
        self.enemy_momentum = 0
        self.enemy_hand = []
        self.enemy_hand_size = 0
        self.enemy_avg_attack = 0

        self.particles = []
        self.animations = deque()
        self.current_reward = 0.0

        # --- Card Definitions ---
        self._define_cards()

        self.reset()

    def _define_cards(self):
        self.CARD_TEMPLATES = {
            "player": [
                {"name": "Quick Slash", "cost": 1, "attack": 6, "block": 0, "desc": "Deal 6 damage."},
                {"name": "Aimed Shot", "cost": 2, "attack": 12, "block": 0, "desc": "Deal 12 damage."},
                {"name": "Barrier", "cost": 1, "attack": 0, "block": 7, "desc": "Gain 7 Block."},
                {"name": "Fortify", "cost": 2, "attack": 0, "block": 14, "desc": "Gain 14 Block."},
                {"name": "Adrenaline", "cost": 0, "attack": 0, "block": 0, "momentum": 2, "desc": "Gain 2 Momentum."},
                {"name": "Combat Rush", "cost": 1, "attack": 4, "block": 4, "desc": "Deal 4 damage. Gain 4 Block."},
            ],
            "enemy": [
                # These are generated procedurally
            ]
        }

    def _create_player_deck(self):
        deck = []
        for _ in range(4): deck.append(self.CARD_TEMPLATES["player"][0].copy()) # Quick Slash
        for _ in range(4): deck.append(self.CARD_TEMPLATES["player"][2].copy()) # Barrier
        deck.append(self.CARD_TEMPLATES["player"][1].copy()) # Aimed Shot
        deck.append(self.CARD_TEMPLATES["player"][3].copy()) # Fortify
        deck.append(self.CARD_TEMPLATES["player"][4].copy()) # Adrenaline
        deck.append(self.CARD_TEMPLATES["player"][5].copy()) # Combat Rush
        return deck

    def _generate_enemy_card(self):
        card_type = random.choice(["attack", "block", "mixed"])
        cost = random.randint(1, 2)
        if card_type == "attack":
            attack = self.enemy_avg_attack + random.randint(-1, 2) * cost
            return {"name": "Virus Spike", "cost": cost, "attack": max(1, attack), "block": 0}
        elif card_type == "block":
            block = self.enemy_avg_attack + random.randint(0, 3) * cost
            return {"name": "Firewall", "cost": cost, "attack": 0, "block": max(1, block)}
        else: # mixed
            attack = self.enemy_avg_attack // 2 + random.randint(0, 1) * cost
            block = self.enemy_avg_attack // 2 + random.randint(0, 1) * cost
            return {"name": "Counter-Protocol", "cost": cost, "attack": max(1, attack), "block": max(1, block)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.game_over_status = ""
        self.game_phase = "PLAYER_TURN"
        
        self.player_hp = self.PLAYER_MAX_HP
        self.player_block = 0
        self.player_momentum = self.PLAYER_STARTING_MOMENTUM
        
        self.player_deck = self._create_player_deck()
        random.shuffle(self.player_deck)
        self.player_discard = []
        self.player_hand = []
        self._draw_player_cards(self.PLAYER_STARTING_HAND_SIZE)
        
        self._setup_new_wave()
        
        self.selected_card_idx = 0
        self.particles = []
        self.animations.clear()

        return self._get_observation(), self._get_info()

    def _setup_new_wave(self):
        self.enemy_hand_size = self.ENEMY_BASE_HAND_SIZE + (self.wave - 1) // 3
        self.enemy_avg_attack = self.ENEMY_BASE_ATTACK + (self.wave - 1) // 5
        self.enemy_max_hp = self.ENEMY_BASE_HP + (self.wave - 1) * 10
        self.enemy_hp = self.enemy_max_hp
        self.enemy_block = 0
        self.enemy_momentum = 2
        self.enemy_hand = [self._generate_enemy_card() for _ in range(self.enemy_hand_size)]
        self._add_animation("text", text=f"WAVE {self.wave}", pos=(self.screen_width/2, self.screen_height/2-50),
                            font=self.font_title, color=self.COLOR_PLAYER, life=60, y_vel=-1)

    def _draw_player_cards(self, num_to_draw):
        for _ in range(num_to_draw):
            if not self.player_deck:
                if not self.player_discard:
                    return # No cards left anywhere
                self.player_deck.extend(self.player_discard)
                self.player_discard.clear()
                random.shuffle(self.player_deck)
                # Sound: deck_shuffle.wav
            if self.player_deck and len(self.player_hand) < 10:
                self.player_hand.append(self.player_deck.pop(0))

    def step(self, action):
        self.steps += 1
        self.current_reward = 0.0
        
        if self.game_phase == "PLAYER_TURN":
            self._handle_player_action(action)
        
        self._update_animations()
        self._update_particles()

        if not self.animations:
            if self.game_phase == "ACTION_RESOLUTION":
                if self.enemy_hp <= 0:
                    self.current_reward += 1.0 # Wave clear reward
                    self.score += 1
                    self.wave += 1
                    if self.wave > self.MAX_WAVES:
                        self.game_phase = "GAME_OVER"
                        self.game_over_status = "WIN"
                    else:
                        self.game_phase = "WAVE_CLEARED"
                else:
                    self.game_phase = "ENEMY_TURN"
            
            elif self.game_phase == "ENEMY_TURN":
                self._handle_enemy_turn()
                self.game_phase = "ENEMY_RESOLUTION"

            elif self.game_phase == "ENEMY_RESOLUTION":
                if self.player_hp <= 0:
                    self.game_phase = "GAME_OVER"
                    self.game_over_status = "LOSE"
                else:
                    self.game_phase = "PLAYER_TURN"
                    self._start_player_turn()

            elif self.game_phase == "WAVE_CLEARED":
                self._setup_new_wave()
                self.game_phase = "PLAYER_TURN"
                self._start_player_turn()

        terminated = self.game_phase == "GAME_OVER" or self.steps >= self.MAX_STEPS
        truncated = False
        if self.steps >= self.MAX_STEPS and not self.game_phase == "GAME_OVER":
            self.game_over = True
            self.game_over_status = "TIMEOUT"
            terminated = True
        
        if terminated and self.game_over_status == "WIN":
            self.current_reward += 100.0
        elif terminated and self.game_over_status == "LOSE":
            self.current_reward -= 100.0
        
        return self._get_observation(), self.current_reward, terminated, truncated, self._get_info()

    def _handle_player_action(self, action):
        movement, space_press, shift_press = action
        turn_ended = False

        # Action priority: Play > Discard > End Turn (no-op)
        if space_press == 1: # Play card
            if 0 <= self.selected_card_idx < len(self.player_hand):
                card = self.player_hand[self.selected_card_idx]
                if self.player_momentum >= card.get('cost', 0):
                    # Sound: card_play.wav
                    self.player_momentum -= card.get('cost', 0)
                    self._execute_card_effect(card, 'player', self.selected_card_idx)
                    self.player_discard.append(self.player_hand.pop(self.selected_card_idx))
                    self.selected_card_idx = min(self.selected_card_idx, max(0, len(self.player_hand) - 1))
                    turn_ended = True
        elif shift_press == 1: # Discard card
            if 0 <= self.selected_card_idx < len(self.player_hand):
                # Sound: card_discard.wav
                self.player_momentum = min(self.PLAYER_MAX_MOMENTUM, self.player_momentum + 1)
                self._add_animation("text", text="+1", pos=(150, 270), color=self.COLOR_MOMENTUM, life=45, y_vel=-0.8)
                self.player_discard.append(self.player_hand.pop(self.selected_card_idx))
                self.selected_card_idx = min(self.selected_card_idx, max(0, len(self.player_hand) - 1))
                turn_ended = True
        
        if movement == 3: # Left
            self.selected_card_idx = max(0, self.selected_card_idx - 1)
        elif movement == 4: # Right
            if self.player_hand:
                self.selected_card_idx = min(len(self.player_hand) - 1, self.selected_card_idx + 1)
        
        # No-op action (movement 0, space 0, shift 0) ends the turn
        if movement == 0 and space_press == 0 and shift_press == 0:
            turn_ended = True

        if turn_ended:
            self.game_phase = "ACTION_RESOLUTION"
    
    def _handle_enemy_turn(self):
        # Sound: enemy_turn.wav
        self.enemy_block = 0
        self.enemy_momentum += self.PLAYER_MOMENTUM_PER_TURN
        
        # AI: find best card to play
        playable_cards = [c for c in self.enemy_hand if c.get('cost', 0) <= self.enemy_momentum]
        if not playable_cards:
            return # End turn if no cards can be played

        # Simple logic: prioritize attack cards
        card_to_play = sorted(playable_cards, key=lambda c: c.get('attack', 0), reverse=True)[0]
        
        self.enemy_momentum -= card_to_play.get('cost', 0)
        self._execute_card_effect(card_to_play, 'enemy')
        self.enemy_hand.remove(card_to_play)
        
        # Replenish hand
        if len(self.enemy_hand) < self.enemy_hand_size:
            self.enemy_hand.append(self._generate_enemy_card())

    def _execute_card_effect(self, card, caster, card_idx=None):
        caster_pos = (self.screen_width/2, self.screen_height - 130) if caster == 'player' else (self.screen_width/2, 130)
        target_pos = (self.screen_width/2, 100) if caster == 'player' else (self.screen_width/2, self.screen_height - 100)

        if card.get('attack', 0) > 0:
            damage = card.get('attack', 0)
            if caster == 'player':
                blocked_damage = min(self.enemy_block, damage)
                self.enemy_block -= blocked_damage
                remaining_damage = damage - blocked_damage
                actual_damage = min(self.enemy_hp, remaining_damage)
                self.enemy_hp -= actual_damage
                self.current_reward += actual_damage * 0.1
                if blocked_damage > 0: self._add_animation("text", text=f"-{blocked_damage}", pos=(target_pos[0]+20, target_pos[1]), color=self.COLOR_BLOCK, life=60, y_vel=-1)
                if actual_damage > 0: self._add_animation("text", text=f"-{actual_damage}", pos=target_pos, color=self.COLOR_ATTACK, life=60, y_vel=-1)
                self._create_particle_burst(target_pos, self.COLOR_ATTACK, 20)
            else: # caster is enemy
                blocked_damage = min(self.player_block, damage)
                self.player_block -= blocked_damage
                remaining_damage = damage - blocked_damage
                actual_damage = min(self.player_hp, remaining_damage)
                self.player_hp -= actual_damage
                self.current_reward -= actual_damage * 0.1
                if blocked_damage > 0: self._add_animation("text", text=f"-{blocked_damage}", pos=(target_pos[0]+20, target_pos[1]), color=self.COLOR_BLOCK, life=60, y_vel=-1)
                if actual_damage > 0: self._add_animation("text", text=f"-{actual_damage}", pos=target_pos, color=self.COLOR_ATTACK, life=60, y_vel=-1)
                self._create_particle_burst(target_pos, self.COLOR_ATTACK, 20)
        
        if card.get('block', 0) > 0:
            block = card.get('block', 0)
            if caster == 'player':
                self.player_block += block
                self._add_animation("text", text=f"+{block}", pos=(caster_pos[0]+20, caster_pos[1]), color=self.COLOR_BLOCK, life=60, y_vel=-1)
            else:
                self.enemy_block += block
                self._add_animation("text", text=f"+{block}", pos=(caster_pos[0]+20, caster_pos[1]), color=self.COLOR_BLOCK, life=60, y_vel=-1)
            self._create_particle_burst(caster_pos, self.COLOR_BLOCK, 10, 0.5)

        if card.get('momentum', 0) > 0:
            if caster == 'player':
                self.player_momentum = min(self.PLAYER_MAX_MOMENTUM, self.player_momentum + card.get('momentum',0))
                self._add_animation("text", text=f"+{card.get('momentum',0)}", pos=(150, 270), color=self.COLOR_MOMENTUM, life=45, y_vel=-0.8)

    def _start_player_turn(self):
        self.player_block = 0
        self.player_momentum = min(self.PLAYER_MAX_MOMENTUM, self.player_momentum + self.PLAYER_MOMENTUM_PER_TURN)
        self._draw_player_cards(self.PLAYER_CARDS_PER_TURN)
        self.selected_card_idx = min(self.selected_card_idx, max(0, len(self.player_hand) - 1))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "player_hp": self.player_hp}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_grid()
        
        self._render_piles()
        self._render_enemy_hand()
        self._render_player_hand()

        self._render_ui()
        
        self._render_particles()
        self._render_animations()

        if self.game_phase == "GAME_OVER":
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_grid(self):
        for i in range(0, self.screen_width, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.screen_height))
        for i in range(0, self.screen_height, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.screen_width, i))
    
    def _render_ui(self):
        # Player Stats
        self._render_stat_bar(self.player_hp, self.PLAYER_MAX_HP, self.player_block, (50, 360), self.COLOR_HEALTH, self.COLOR_PLAYER, "PLAYER")
        self._render_momentum_bar(self.player_momentum, self.PLAYER_MAX_MOMENTUM, (50, 330), self.COLOR_MOMENTUM)

        # Enemy Stats
        self._render_stat_bar(self.enemy_hp, self.enemy_max_hp, self.enemy_block, (self.screen_width - 250, 40), self.COLOR_HEALTH, self.COLOR_ENEMY, f"ENEMY WAVE {self.wave}")

        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.screen_width - 150, self.screen_height - 30))

    def _render_stat_bar(self, hp, max_hp, block, pos, color, entity_color, label):
        bar_width = 200
        bar_height = 20
        
        # Label
        label_text = self.font_medium.render(label, True, entity_color)
        self.screen.blit(label_text, (pos[0], pos[1] - 22))

        # BG
        bg_rect = pygame.Rect(pos[0], pos[1], bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_GRID, bg_rect, border_radius=3)
        
        # HP
        hp_ratio = max(0, hp / max_hp)
        hp_rect = pygame.Rect(pos[0], pos[1], int(bar_width * hp_ratio), bar_height)
        pygame.draw.rect(self.screen, color, hp_rect, border_radius=3)
        
        # Block
        if block > 0:
            block_width = min(bar_width, int(block * 5)) # Arbitrary scale for visibility
            block_rect = pygame.Rect(pos[0] + int(bar_width * hp_ratio), pos[1], block_width, bar_height)
            pygame.draw.rect(self.screen, self.COLOR_BLOCK, block_rect, border_radius=3)
            block_text = self.font_medium.render(str(block), True, self.COLOR_TEXT)
            self.screen.blit(block_text, (pos[0] - 30, pos[1]))

        # Text
        hp_text = self.font_medium.render(f"{hp}/{max_hp}", True, self.COLOR_TEXT)
        text_rect = hp_text.get_rect(center=bg_rect.center)
        self.screen.blit(hp_text, text_rect)
        pygame.draw.rect(self.screen, self.COLOR_CARD_BORDER, bg_rect, 1, border_radius=3)

    def _render_momentum_bar(self, momentum, max_momentum, pos, color):
        bar_width = 200
        bar_height = 10
        bg_rect = pygame.Rect(pos[0], pos[1], bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_GRID, bg_rect, border_radius=3)
        
        mom_ratio = max(0, momentum / max_momentum)
        mom_rect = pygame.Rect(pos[0], pos[1], int(bar_width * mom_ratio), bar_height)
        pygame.draw.rect(self.screen, color, mom_rect, border_radius=3)
        
        label_text = self.font_medium.render(f"Momentum: {momentum}/{max_momentum}", True, self.COLOR_TEXT)
        self.screen.blit(label_text, (pos[0], pos[1] - 22))
        pygame.draw.rect(self.screen, self.COLOR_CARD_BORDER, bg_rect, 1, border_radius=3)

    def _render_piles(self):
        # Player Draw Pile
        if self.player_deck:
            self._render_card_pile((50, self.screen_height / 2 + 50), len(self.player_deck), "Draw")
        # Player Discard Pile
        if self.player_discard:
            self._render_card_pile((self.screen_width - 100, self.screen_height / 2 + 50), len(self.player_discard), "Discard")
    
    def _render_card_pile(self, pos, count, label):
        card_w, card_h = 80, 110
        for i in range(min(count, 5)):
            rect = pygame.Rect(pos[0] + i * 2, pos[1] - i * 2, card_w, card_h)
            pygame.draw.rect(self.screen, self.COLOR_CARD_BG, rect, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_CARD_BORDER, rect, 2, border_radius=5)
        
        text = self.font_medium.render(f"{label} [{count}]", True, self.COLOR_TEXT_DIM)
        self.screen.blit(text, (pos[0], pos[1] + card_h + 5))

    def _render_player_hand(self):
        hand_size = len(self.player_hand)
        if hand_size == 0: return
        
        card_w, card_h = 90, 130
        total_width = (hand_size * (card_w + 10)) - 10
        start_x = (self.screen_width - total_width) / 2
        
        for i, card in enumerate(self.player_hand):
            is_selected = (i == self.selected_card_idx and self.game_phase == "PLAYER_TURN")
            can_afford = self.player_momentum >= card.get('cost', 0)
            
            card_x = start_x + i * (card_w + 10)
            card_y = self.screen_height - card_h - 20
            if is_selected:
                card_y -= 20
            
            self._draw_card(self.screen, (card_x, card_y), (card_w, card_h), card, is_selected, not can_afford)

    def _render_enemy_hand(self):
        hand_size = len(self.enemy_hand)
        if hand_size == 0: return

        card_w, card_h = 60, 90
        total_width = (hand_size * (card_w - 30))
        start_x = (self.screen_width - total_width) / 2

        for i, card in enumerate(self.enemy_hand):
            card_x = start_x + i * (card_w - 30)
            card_y = 80
            self._draw_card(self.screen, (card_x, card_y), (card_w, card_h), card, is_enemy=True)

    def _draw_card(self, surface, pos, size, card_data, is_selected=False, is_unaffordable=False, is_enemy=False):
        rect = pygame.Rect(pos, size)
        
        # Glow for selected card
        if is_selected:
            glow_rect = rect.inflate(10, 10)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_CARD_SELECTED, 100), s.get_rect(), border_radius=12)
            pygame.draw.rect(s, (*self.COLOR_CARD_SELECTED, 150), s.get_rect().inflate(-4, -4), border_radius=10)
            surface.blit(s, glow_rect.topleft)

        # Card body
        pygame.draw.rect(surface, self.COLOR_CARD_BG, rect, border_radius=8)
        
        # Unaffordable tint
        if is_unaffordable:
            s = pygame.Surface(size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 150))
            surface.blit(s, pos)

        # Border
        border_color = self.COLOR_CARD_SELECTED if is_selected else self.COLOR_CARD_BORDER
        pygame.draw.rect(surface, border_color, rect, 2, border_radius=8)
        
        if is_enemy: # Simplified enemy card
            name_text = self.font_small.render("???", True, self.COLOR_TEXT_DIM)
            surface.blit(name_text, (rect.x + 5, rect.y + 5))
            return

        # Card Content
        name_text = self.font_medium.render(card_data['name'], True, self.COLOR_TEXT)
        surface.blit(name_text, (rect.x + 10, rect.y + 10))

        cost = card_data.get('cost', 0)
        cost_text = self.font_large.render(str(cost), True, self.COLOR_MOMENTUM)
        pygame.gfxdraw.filled_circle(surface, rect.x + 20, rect.y + 40, 12, self.COLOR_MOMENTUM)
        pygame.gfxdraw.aacircle(surface, rect.x + 20, rect.y + 40, 12, self.COLOR_TEXT)
        cost_rect = cost_text.get_rect(center=(rect.x + 20, rect.y + 40))
        surface.blit(cost_text, cost_rect)

        y_offset = rect.y + 60
        if card_data.get('attack', 0) > 0:
            text = self.font_small.render(f"Attack: {card_data['attack']}", True, self.COLOR_ATTACK)
            surface.blit(text, (rect.x + 10, y_offset))
            y_offset += 15
        if card_data.get('block', 0) > 0:
            text = self.font_small.render(f"Block: {card_data['block']}", True, self.COLOR_BLOCK)
            surface.blit(text, (rect.x + 10, y_offset))
            y_offset += 15
        if card_data.get('momentum', 0) > 0:
            text = self.font_small.render(f"Momentum: +{card_data['momentum']}", True, self.COLOR_MOMENTUM)
            surface.blit(text, (rect.x + 10, y_offset))
            y_offset += 15

    def _render_game_over(self):
        s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 220))
        self.screen.blit(s, (0, 0))
        
        status_text = "VICTORY" if self.game_over_status == "WIN" else "DEFEAT"
        status_color = self.COLOR_PLAYER if self.game_over_status == "WIN" else self.COLOR_ENEMY
        
        title = self.font_title.render(status_text, True, status_color)
        title_rect = title.get_rect(center=(self.screen_width / 2, self.screen_height / 2 - 20))
        self.screen.blit(title, title_rect)
        
        score_text = self.font_large.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 30))
        self.screen.blit(score_text, score_rect)

    def _add_animation(self, anim_type, **kwargs):
        kwargs['type'] = anim_type
        kwargs['life'] = kwargs.get('life', 60)
        self.animations.append(kwargs)

    def _update_animations(self):
        if not self.animations: return
        
        anim = self.animations[0]
        anim['life'] -= 1

        if anim['type'] == 'text':
            anim['pos'] = (anim['pos'][0], anim['pos'][1] + anim.get('y_vel', 0))

        if anim['life'] <= 0:
            self.animations.popleft()

    def _render_animations(self):
        for anim in self.animations:
            if anim['type'] == 'text':
                alpha = min(255, int(255 * (anim['life'] / 30))) if anim['life'] < 30 else 255
                font = anim.get('font', self.font_large)
                text_surf = font.render(anim['text'], True, (*anim['color'], alpha))
                text_rect = text_surf.get_rect(center=anim['pos'])
                self.screen.blit(text_surf, text_rect)

    def _create_particle_burst(self, pos, color, count, speed=1):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = (math.cos(angle) * random.uniform(1, 3) * speed, math.sin(angle) * random.uniform(1, 3) * speed)
            life = random.randint(20, 40)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(5 * (p['life'] / p['max_life']))
            if size > 0:
                rect = pygame.Rect(p['pos'][0] - size/2, p['pos'][1] - size/2, size, size)
                s = pygame.Surface((size, size), pygame.SRCALPHA)
                pygame.draw.rect(s, color, s.get_rect())
                self.screen.blit(s, rect.topleft)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual play and debugging
    # It will not be run by the evaluation system, which only imports the GameEnv class.
    # We can use a real display here for testing.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Manual Play Controls ---
    print("--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")

    # Create a display surface if one does not exist
    try:
        display_screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Card Battler")
    except pygame.error:
        print("No display available, cannot run interactive mode.")
        env.close()
        exit()

    running = True
    while running:
        action = [0, 0, 0] # Default: End Turn / No-op
        action_taken = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                    action_taken = True
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                    action_taken = True
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                    action_taken = True
                elif event.key == pygame.K_LSHIFT:
                    action[2] = 1
                    action_taken = True
                elif event.key == pygame.K_RETURN:
                    # Explicitly end turn
                    action = [0, 0, 0]
                    action_taken = True
                elif event.key == pygame.K_r:
                    # Reset game
                    obs, info = env.reset()
                    print("--- Game Reset ---")
                    action_taken = False # Don't step after reset
        
        if action_taken and not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Phase: {env.game_phase}")
            if terminated:
                print(f"Game Over! Status: {env.game_over_status}, Final Score: {info['score']}")

        # Get the current observation to render
        # If the game is turn-based, we might just call _get_observation without stepping
        if not action_taken:
            obs = env._get_observation()

        # Render for human viewing
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Run at 30 FPS

    env.close()