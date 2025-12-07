import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:38:14.985933
# Source Brief: brief_00523.md
# Brief Index: 523
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a futuristic tower defense game.
    The player defends four portals from waves of enemies by placing time-manipulating cards.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Defend four portals from enemy waves by placing time-manipulating cards with past and future effects."
    user_guide = "Use arrow keys to select a portal, space to place a card, and shift to cycle cards or change a card's time state."
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000
    TOTAL_WAVES = 20

    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_GRID = (30, 20, 70)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TEXT_DIM = (150, 150, 180)
    COLOR_PLAYER_CURSOR = (255, 255, 0)
    
    PORTAL_COLORS = {
        'BLUE': (50, 150, 255),
        'GREEN': (50, 255, 150),
        'RED': (255, 100, 100),
        'PURPLE': (200, 100, 255),
    }

    ENEMY_COLORS = {
        'GRUNT': (255, 50, 50),
        'SPRINTER': (255, 150, 50),
        'BRUTE': (200, 30, 100),
    }

    CARD_TIME_STATE_COLORS = {
        'PAST': (100, 255, 100),
        'FUTURE': (100, 100, 255),
    }

    # Card Definitions
    CARD_DEFINITIONS = {
        "BLAST": {
            "past_desc": "Periodic AoE damage.",
            "future_desc": "High-damage single-target beam.",
            "cooldown": 60, # 2 seconds
        },
        "SLOW": {
            "past_desc": "AoE slowing field.",
            "future_desc": "Freezes one enemy.",
            "cooldown": 90, # 3 seconds
        },
        "HEAL": {
            "past_desc": "Heals portal over time.",
            "future_desc": "Creates a temporary shield.",
            "cooldown": 150, # 5 seconds
        },
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_card = pygame.font.SysFont("Consolas", 14)
        self.font_title = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.portals = None
        self.enemies = None
        self.placed_cards = None
        self.particles = None
        self.player_hand = None
        self.wave_num = None
        self.wave_timer = None
        self.wave_enemy_count = None
        self.wave_enemies_spawned = None
        self.wave_complete = None
        self.pending_rewards = None
        self.selected_portal_idx = None
        self.selected_hand_idx = None
        self.last_space_held = None
        self.last_shift_held = None
        
        self.reset()
        # self.validate_implementation() # Commented out as it's for dev purposes

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.pending_rewards = 0.0

        # Player state
        self.selected_portal_idx = 0
        self.selected_hand_idx = 0
        self.last_space_held = False
        self.last_shift_held = False

        # Game entities
        self._initialize_portals()
        self.enemies = []
        self.placed_cards = []
        self.particles = deque()
        
        # Progression
        self.wave_num = 0
        self.wave_timer = 150 # 5 seconds until first wave
        self.wave_complete = True
        self.player_hand = ["BLAST", "SLOW"]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.pending_rewards = 0.0

        self._handle_player_input(action)
        self._update_game_logic()
        
        reward = self.pending_rewards
        terminated = self._check_termination()

        if terminated:
            if self.wave_num > self.TOTAL_WAVES:
                reward += 100 # Victory bonus
                self.score += 100
            else:
                reward -= 100 # Defeat penalty
                self.score -= 100

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Private Helper Methods: Core Logic ---

    def _handle_player_input(self, action):
        movement, space_held, shift_held = action
        
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        # Action 0: Movement (selects portal)
        if movement > 0:
            self.selected_portal_idx = movement - 1

        # Action 1: Space (place card)
        if space_pressed:
            self._place_card()

        # Action 2: Shift (cycle time state of placed cards at portal OR cycle hand)
        if shift_pressed:
            # If there are cards at the selected portal, cycle their state
            portal_cards = [card for card in self.placed_cards if card['portal_idx'] == self.selected_portal_idx]
            if portal_cards:
                # Cycle state of the most recently placed card at this portal
                target_card = portal_cards[-1]
                if target_card['time_state'] == 'PAST':
                    target_card['time_state'] = 'FUTURE'
                else:
                    target_card['time_state'] = 'PAST'
                # sfx: time_shift.wav
            else:
                # If no cards, cycle selected card in hand
                self.selected_hand_idx = (self.selected_hand_idx + 1) % len(self.player_hand)
                # sfx: ui_cycle.wav

        self.last_space_held = bool(space_held)
        self.last_shift_held = bool(shift_held)

    def _place_card(self):
        portal = self.portals[self.selected_portal_idx]
        
        # Check if there's a free slot
        placed_count = sum(1 for card in self.placed_cards if card['portal_idx'] == self.selected_portal_idx)
        if placed_count >= portal['max_slots']:
            # sfx: action_fail.wav
            return

        card_name = self.player_hand[self.selected_hand_idx]
        new_card = {
            "name": card_name,
            "portal_idx": self.selected_portal_idx,
            "time_state": 'PAST', # Default to PAST, player can shift it
            "cooldown": 0,
            "max_cooldown": self.CARD_DEFINITIONS[card_name]["cooldown"],
        }
        self.placed_cards.append(new_card)
        # sfx: card_place.wav
        
        # Create placement particles
        for _ in range(20):
            self._create_particle(
                pos=portal['slots'][placed_count], 
                color=(255, 255, 200), 
                life=20, 
                speed_range=(1, 3)
            )

    def _update_game_logic(self):
        self._update_waves()
        self._update_enemies()
        self._update_cards()
        self._update_particles()
        
    def _initialize_portals(self):
        self.portals = []
        # My action mapping uses movement-1, so:
        # 0->Up, 1->Down, 2->Left, 3->Right
        # I'll map them as: 0:Up, 1:Down, 2:Left, 3:Right
        positions = [
            (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.2), # 0: Top
            (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.8), # 1: Bottom
            (self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT / 2), # 2: Left
            (self.SCREEN_WIDTH * 0.8, self.SCREEN_HEIGHT / 2), # 3: Right
        ]
        colors = [self.PORTAL_COLORS['BLUE'], self.PORTAL_COLORS['GREEN'], self.PORTAL_COLORS['RED'], self.PORTAL_COLORS['PURPLE']]
        
        for i in range(4):
            portal_pos = positions[i]
            slot_positions = []
            for j in range(3):
                angle = (math.pi * 2 / 3) * j
                slot_pos = (portal_pos[0] + math.cos(angle) * 40, portal_pos[1] + math.sin(angle) * 40)
                slot_positions.append(slot_pos)

            self.portals.append({
                "pos": portal_pos,
                "color": colors[i],
                "health": 100,
                "max_health": 100,
                "shield": 0,
                "max_shield": 50,
                "max_slots": 3,
                "slots": slot_positions,
            })

    def _update_waves(self):
        if self.wave_complete:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.wave_num += 1
                if self.wave_num > self.TOTAL_WAVES:
                    return # Game won
                
                self.wave_complete = False
                self.wave_enemy_count = 2 + self.wave_num
                self.wave_enemies_spawned = 0
                self.wave_timer = 20 # Time between spawns
                
                # Unlock new cards
                if self.wave_num == 2 and "HEAL" not in self.player_hand:
                    self.player_hand.append("HEAL")
                
                self.pending_rewards += 1.0 # Wave survival bonus
        else:
            # Spawn enemies during a wave
            self.wave_timer -= 1
            if self.wave_timer <= 0 and self.wave_enemies_spawned < self.wave_enemy_count:
                self._spawn_enemy()
                self.wave_enemies_spawned += 1
                self.wave_timer = max(15, 45 - self.wave_num * 2) # Faster spawns in later waves

            # Check for wave completion
            if self.wave_enemies_spawned >= self.wave_enemy_count and not self.enemies:
                self.wave_complete = True
                self.wave_timer = 240 # 8 seconds grace period

    def _spawn_enemy(self):
        # Spawn at random screen edge
        side = self.np_random.integers(4)
        if side == 0: pos = (self.np_random.uniform(0, self.SCREEN_WIDTH), -10)
        elif side == 1: pos = (self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 10)
        elif side == 2: pos = (-10, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        else: pos = (self.SCREEN_WIDTH + 10, self.np_random.uniform(0, self.SCREEN_HEIGHT))

        # Choose enemy type based on wave
        enemy_type = 'GRUNT'
        if self.wave_num >= 5:
            if self.np_random.random() < 0.3: enemy_type = 'SPRINTER'
        if self.wave_num >= 10:
            if self.np_random.random() < 0.2: enemy_type = 'BRUTE'

        # Scale health with wave number
        health_multiplier = 1 + (self.wave_num - 1) * 0.05
        
        enemy_stats = {
            'GRUNT': {'health': 100 * health_multiplier, 'speed': 1.0},
            'SPRINTER': {'health': 60 * health_multiplier, 'speed': 1.8},
            'BRUTE': {'health': 250 * health_multiplier, 'speed': 0.6},
        }

        # Target the portal with the highest health
        target_portal_idx = max(range(len(self.portals)), key=lambda i: self.portals[i]['health'])

        self.enemies.append({
            "pos": np.array(pos, dtype=float),
            "type": enemy_type,
            "health": enemy_stats[enemy_type]['health'],
            "max_health": enemy_stats[enemy_type]['health'],
            "base_speed": enemy_stats[enemy_type]['speed'],
            "current_speed": enemy_stats[enemy_type]['speed'],
            "target_portal_idx": target_portal_idx,
            "frozen_timer": 0,
        })

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy['frozen_timer'] > 0:
                enemy['frozen_timer'] -= 1
                continue

            portal = self.portals[enemy['target_portal_idx']]
            direction = np.array(portal['pos']) - enemy['pos']
            dist = np.linalg.norm(direction)

            if dist < 20: # Reached portal
                damage = enemy['health'] # Enemy's remaining health is damage
                
                # Apply to shield first
                shield_damage = min(portal['shield'], damage)
                portal['shield'] -= shield_damage
                damage -= shield_damage
                
                # Then to health
                portal['health'] -= damage
                portal['health'] = max(0, portal['health'])
                
                self.enemies.remove(enemy)
                # sfx: portal_damage.wav
                self._create_particle(pos=portal['pos'], color=(255, 0, 0), count=50, life=40)
                continue
            
            # Reset speed every frame unless slowed
            enemy['current_speed'] = enemy['base_speed']
            
            # Movement
            norm_direction = direction / dist
            enemy['pos'] += norm_direction * enemy['current_speed']

    def _update_cards(self):
        for card in self.placed_cards:
            card['cooldown'] = max(0, card['cooldown'] - 1)
            if card['cooldown'] > 0:
                continue

            portal = self.portals[card['portal_idx']]
            card_pos = portal['slots'][self.placed_cards.index(card) % portal['max_slots']]
            
            # --- CARD EFFECTS ---
            if card['name'] == 'BLAST':
                if card['time_state'] == 'PAST': # AoE Damage
                    hit = False
                    for enemy in self.enemies:
                        if np.linalg.norm(np.array(portal['pos']) - enemy['pos']) < 80:
                            enemy['health'] -= 25
                            hit = True
                            self._create_particle(pos=enemy['pos'], color=(255, 200, 100), count=5, life=15)
                    if hit:
                        card['cooldown'] = card['max_cooldown']
                        # sfx: blast_aoe.wav
                        self._create_particle(pos=card_pos, color=(255, 200, 100), count=30, life=20, speed_range=(2, 4))
                
                elif card['time_state'] == 'FUTURE': # Single Target Beam
                    # Target closest enemy
                    targets = [e for e in self.enemies if np.linalg.norm(np.array(portal['pos']) - e['pos']) < 250]
                    if targets:
                        closest_enemy = min(targets, key=lambda e: np.linalg.norm(np.array(card_pos) - e['pos']))
                        closest_enemy['health'] -= 120
                        card['cooldown'] = card['max_cooldown']
                        # sfx: blast_laser.wav
                        self.particles.append({'type': 'beam', 'start': card_pos, 'end': closest_enemy['pos'], 'life': 10, 'max_life': 10, 'color': self.CARD_TIME_STATE_COLORS['FUTURE']})
                        self._create_particle(pos=closest_enemy['pos'], color=(150, 150, 255), count=15, life=20)
            
            elif card['name'] == 'SLOW':
                if card['time_state'] == 'PAST': # Slowing Field
                    # This effect is passive, so we apply it to enemies in range
                    for enemy in self.enemies:
                        if np.linalg.norm(np.array(portal['pos']) - enemy['pos']) < 100:
                            enemy['current_speed'] *= 0.4
                    # Visual effect for the field
                    self.particles.append({'type': 'field', 'pos': portal['pos'], 'radius': 100, 'life': 2, 'max_life': 2, 'color': self.CARD_TIME_STATE_COLORS['PAST']})
                
                elif card['time_state'] == 'FUTURE': # Freeze
                    targets = [e for e in self.enemies if e['frozen_timer'] <= 0 and np.linalg.norm(np.array(portal['pos']) - e['pos']) < 250]
                    if targets:
                        target_enemy = self.np_random.choice(targets)
                        target_enemy['frozen_timer'] = 120 # 4 seconds
                        card['cooldown'] = card['max_cooldown']
                        # sfx: freeze.wav
                        self._create_particle(pos=target_enemy['pos'], color=(200, 200, 255), count=20, life=120, speed_range=(0,0))
            
            elif card['name'] == 'HEAL':
                if card['time_state'] == 'PAST': # Heal Portal
                    if portal['health'] < portal['max_health']:
                        portal['health'] = min(portal['max_health'], portal['health'] + 20)
                        card['cooldown'] = card['max_cooldown']
                        # sfx: heal.wav
                        self._create_particle(pos=portal['pos'], color=self.CARD_TIME_STATE_COLORS['PAST'], count=20, life=30, speed_range=(0.5, 1.5))
                
                elif card['time_state'] == 'FUTURE': # Shield Portal
                    if portal['shield'] < portal['max_shield']:
                        portal['shield'] = min(portal['max_shield'], portal['shield'] + 50)
                        card['cooldown'] = card['max_cooldown']
                        # sfx: shield_up.wav
                        self.particles.append({'type': 'shield_effect', 'pos': portal['pos'], 'life': 30, 'max_life': 30, 'color': self.CARD_TIME_STATE_COLORS['FUTURE']})

        # Check for dead enemies and give rewards
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                self.enemies.remove(enemy)
                self.pending_rewards += 0.1
                # sfx: enemy_die.wav
                self._create_particle(pos=enemy['pos'], color=self.ENEMY_COLORS[enemy['type']], count=30, life=25, speed_range=(1, 4))
    
    def _create_particle(self, pos, color, count=1, life=20, speed_range=(1,2)):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(*speed_range)
            velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'type': 'normal',
                'pos': np.array(pos, dtype=float),
                'vel': velocity,
                'life': life,
                'max_life': life,
                'color': color
            })

    def _update_particles(self):
        for _ in range(len(self.particles)):
            p = self.particles.popleft()
            p['life'] -= 1
            if p['life'] > 0:
                if p['type'] == 'normal':
                    p['pos'] += p['vel']
                    p['vel'] *= 0.95 # friction
                self.particles.append(p)

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        
        if self.wave_num > self.TOTAL_WAVES:
            self.game_over = True
            return True

        if all(p['health'] <= 0 for p in self.portals):
            self.game_over = True
            return True
            
        return False

    # --- Private Helper Methods: Rendering ---

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i), 1)

    def _render_game(self):
        self._render_particles()
        self._render_portals_and_cards()
        self._render_enemies()

    def _render_portals_and_cards(self):
        # Selection indicator
        selected_portal = self.portals[self.selected_portal_idx]
        pulse = abs(math.sin(self.steps * 0.1)) * 5
        self._draw_glowing_circle(self.screen, selected_portal['pos'], 55 + pulse, self.COLOR_PLAYER_CURSOR, 10)

        # Portals
        for i, portal in enumerate(self.portals):
            self._draw_glowing_circle(self.screen, portal['pos'], 30, portal['color'], 15)
            # Health bar
            health_pct = portal['health'] / portal['max_health']
            bar_pos = (portal['pos'][0] - 25, portal['pos'][1] + 35)
            pygame.draw.rect(self.screen, (50, 0, 0), (*bar_pos, 50, 5))
            pygame.draw.rect(self.screen, (0, 255, 0), (*bar_pos, 50 * health_pct, 5))
            
            # Shield bar
            if portal['shield'] > 0:
                shield_pct = portal['shield'] / portal['max_shield']
                self._draw_glowing_circle(self.screen, portal['pos'], 32 + shield_pct*3, self.CARD_TIME_STATE_COLORS['FUTURE'], 5)

        # Placed cards
        portal_card_counts = {i: 0 for i in range(4)}
        for card in self.placed_cards:
            portal_idx = card['portal_idx']
            portal = self.portals[portal_idx]
            slot_idx = portal_card_counts[portal_idx]
            pos = portal['slots'][slot_idx]
            portal_card_counts[portal_idx] += 1
            
            color = self.CARD_TIME_STATE_COLORS[card['time_state']]
            self._draw_glowing_circle(self.screen, pos, 12, color, 8)
            
            # Cooldown indicator
            cooldown_pct = card['cooldown'] / card['max_cooldown']
            if cooldown_pct > 0:
                self._draw_arc(self.screen, pos, 14, (100, 100, 100), 0, 360 * cooldown_pct, 3)

            # Card initial
            text = self.font_card.render(card['name'][0], True, self.COLOR_TEXT)
            self.screen.blit(text, text.get_rect(center=pos))

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            color = self.ENEMY_COLORS[enemy['type']]
            
            if enemy['type'] == 'GRUNT':
                pygame.gfxdraw.aacircle(self.screen, *pos, 8, color)
                pygame.gfxdraw.filled_circle(self.screen, *pos, 8, color)
            elif enemy['type'] == 'SPRINTER':
                points = [(pos[0], pos[1] - 8), (pos[0] - 6, pos[1] + 6), (pos[0] + 6, pos[1] + 6)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            elif enemy['type'] == 'BRUTE':
                 pygame.gfxdraw.box(self.screen, (pos[0]-7, pos[1]-7, 14, 14), color)

            if enemy['frozen_timer'] > 0:
                self._draw_glowing_circle(self.screen, pos, 12, (200, 200, 255, 150), 5)
            
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            bar_pos = (pos[0] - 10, pos[1] - 15)
            pygame.draw.rect(self.screen, (50, 0, 0), (*bar_pos, 20, 3))
            pygame.draw.rect(self.screen, (255, 0, 0), (*bar_pos, 20 * health_pct, 3))

    def _render_particles(self):
        for p in self.particles:
            life_pct = p['life'] / p['max_life']
            
            if p['type'] == 'normal':
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                radius = int(life_pct * 5)
                if radius > 0:
                    pygame.gfxdraw.aacircle(self.screen, *pos, radius, p['color'])
            
            elif p['type'] == 'beam':
                pygame.draw.aaline(self.screen, p['color'], p['start'], p['end'], int(life_pct * 5))
            
            elif p['type'] == 'field':
                self._draw_glowing_circle(self.screen, p['pos'], p['radius'], p['color'] + (int(life_pct * 50),), 0, True)

            elif p['type'] == 'shield_effect':
                self._draw_glowing_circle(self.screen, p['pos'], 32 + (1-life_pct)*15, p['color'] + (int(life_pct * 150),), 0, True)

    def _render_ui(self):
        # Top-left info
        enemies_left = len(self.enemies) + (self.wave_enemy_count - self.wave_enemies_spawned if not self.wave_complete else 0)
        wave_text = f"WAVE: {self.wave_num}/{self.TOTAL_WAVES}"
        enemies_text = f"ENEMIES: {enemies_left}"
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(wave_text, (10, 10), self.font_ui)
        self._draw_text(enemies_text, (10, 30), self.font_ui)
        self._draw_text(score_text, (10, 50), self.font_ui)

        # Wave timer
        if self.wave_complete and self.wave_num < self.TOTAL_WAVES:
            timer_sec = self.wave_timer // self.FPS
            timer_text = f"NEXT WAVE IN: {timer_sec}"
            self._draw_text(timer_text, (self.SCREEN_WIDTH / 2, 10), self.font_ui, center=True)
        
        # Bottom card selection UI
        ui_y = self.SCREEN_HEIGHT - 60
        pygame.draw.rect(self.screen, (0,0,0,150), (0, ui_y, self.SCREEN_WIDTH, 60))
        
        selected_card_name = self.player_hand[self.selected_hand_idx]
        card_def = self.CARD_DEFINITIONS[selected_card_name]
        
        self._draw_text(f"HAND [SHIFT to cycle]: {', '.join(self.player_hand)}", (10, ui_y + 5), self.font_card, color=self.COLOR_TEXT_DIM)
        self._draw_text(f"SELECTED: {selected_card_name}", (10, ui_y + 25), self.font_title)
        
        self._draw_text(f"PAST:", (250, ui_y + 15), self.font_card, color=self.CARD_TIME_STATE_COLORS['PAST'])
        self._draw_text(card_def['past_desc'], (290, ui_y + 15), self.font_card)
        self._draw_text(f"FUTURE:", (250, ui_y + 35), self.font_card, color=self.CARD_TIME_STATE_COLORS['FUTURE'])
        self._draw_text(card_def['future_desc'], (310, ui_y + 35), self.font_card)

        # Game Over Screen
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0, 180))
            self.screen.blit(s, (0,0))
            msg = "VICTORY" if self.wave_num > self.TOTAL_WAVES else "ALL PORTALS LOST"
            self._draw_text(msg, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20), self.font_title, center=True)
            self._draw_text(f"FINAL SCORE: {int(self.score)}", (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20), self.font_ui, center=True)


    def _draw_text(self, text, pos, font, color=None, center=False):
        color = color or self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_size, filled=True, use_alpha=False):
        pos = (int(pos[0]), int(pos[1]))
        if radius <= 0: return

        for i in range(glow_size, 0, -1):
            alpha = int(150 * (1 - i / glow_size))
            if use_alpha:
                c = color
            else:
                c = (*color, alpha)
            
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius + i), c)
        
        if filled:
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius), color)
        else:
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(radius), color)

    def _draw_arc(self, surface, center, radius, color, start_angle_deg, stop_angle_deg, width):
        center = (int(center[0]), int(center[1]))
        radius = int(radius)
        if radius <= 0: return
        
        rect = pygame.Rect(center[0]-radius, center[1]-radius, radius*2, radius*2)
        start_rad = math.radians(start_angle_deg)
        stop_rad = math.radians(stop_angle_deg)
        
        if stop_rad > start_rad:
            pygame.draw.arc(surface, color, rect, start_rad, stop_rad, width)

    # --- Gymnasium Interface Compliance ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_num,
            "portals_health": [p['health'] for p in self.portals],
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# --- Example Usage ---
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Controls ---
    # Arrows: Select Portal (Up, Down, Left, Right)
    # Space: Place selected card
    # L-Shift: Cycle card in hand (if no cards at portal) / Cycle time state of placed card
    # R: Reset environment
    
    obs, info = env.reset()
    done = False
    
    # Setup for manual play
    # To play manually, you need a display. The line below will be commented out.
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # You might need to remove the os.environ setting at the top of the file.
    pygame.display.set_caption("Temporal Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key == pygame.K_q:
                    running = False

        keys = pygame.key.get_pressed()
        
        # Movement
        mov = 0
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        # Actions
        space = keys[pygame.K_SPACE]
        shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [mov, 1 if space else 0, 1 if shift else 0]

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            # Keep showing the game over screen for a bit
            pass

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()