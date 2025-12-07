import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:44:44.488021
# Source Brief: brief_01175.md
# Brief Index: 1175
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
    A Gymnasium environment where the agent places trap cards on a grid to capture
    valuable sea creatures in a deep-sea setting. The goal is to reach a target
    score by strategically using different types of cards before running out of
    placements.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Place trap cards on a grid to capture valuable sea creatures. Use different card types strategically "
        "to attract, repel, or stun creatures before activating your traps to score points."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place the selected trap card. "
        "Press shift to activate all placed traps at once."
    )
    auto_advance = True

    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 10
    GRID_ROWS = 6
    GRID_LEFT = 60
    GRID_TOP = 40
    CELL_SIZE = 50
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (30, 50, 90)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SCORE = (255, 200, 0)
    
    CREATURE_COLORS = {
        'common': (0, 255, 150),
        'uncommon': (255, 150, 255),
        'rare': (255, 100, 100)
    }

    # Game Parameters
    WIN_SCORE = 1000
    MAX_STEPS = 1500
    MAX_CARDS = 25
    INITIAL_CREATURES = 15
    BASE_CREATURE_SPEED = 0.75
    
    # Card Types
    CARD_TYPES = {
        'CAPTURE': {'color': (255, 255, 255), 'symbol': 'C', 'radius': 75},
        'ATTRACT': {'color': (0, 150, 255), 'symbol': 'A', 'radius': 120, 'power': 15.0},
        'REPEL': {'color': (255, 100, 0), 'symbol': 'R', 'radius': 120, 'power': -15.0},
        'STUN': {'color': (200, 50, 255), 'symbol': 'S', 'radius': 100, 'duration': 90}
    }
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_card = pygame.font.SysFont("monospace", 20, bold=True)
        
        # --- Game State Initialization ---
        # These are reset in self.reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.grid = None
        self.creatures = []
        self.particles = []
        self.unlocked_cards = []
        self.current_card_idx = 0
        self.cards_placed_count = 0
        self.space_was_held = False
        self.shift_was_held = False
        self.move_cooldown = 0
        self.bg_particles = []

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        
        self.creatures = []
        self._spawn_creatures(self.INITIAL_CREATURES)
        
        self.particles = []
        if not self.bg_particles:
            self._init_bg_particles(100)

        self.unlocked_cards = ['CAPTURE', 'ATTRACT']
        self.current_card_idx = 0
        self.cards_placed_count = 0
        
        self.space_was_held = False
        self.shift_was_held = False
        self.move_cooldown = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        
        if shift_held and not self.shift_was_held:
            # --- Activate Traps ---
            # Sound: activate_traps.wav
            reward += self._activate_traps()

        self._update_game_state()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Win bonus
            elif self.cards_placed_count >= self.MAX_CARDS:
                reward -= 50.0 # Loss penalty
        
        self.space_was_held = space_held
        self.shift_was_held = shift_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # --- Cursor Movement ---
        if self.move_cooldown <= 0:
            moved = False
            if movement == 1: self.cursor_pos[1] -= 1; moved = True # Up
            elif movement == 2: self.cursor_pos[1] += 1; moved = True # Down
            elif movement == 3: self.cursor_pos[0] -= 1; moved = True # Left
            elif movement == 4: self.cursor_pos[0] += 1; moved = True # Right
            
            if moved:
                self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
                self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)
                self.move_cooldown = 5 # 5-frame cooldown for smoother selection
        
        # --- Place Card ---
        if space_held and not self.space_was_held:
            self._place_card()

    def _place_card(self):
        if self.cards_placed_count >= self.MAX_CARDS:
            # Sound: error.wav
            return

        x, y = self.cursor_pos
        if self.grid[y][x] == '':
            card_type = self.unlocked_cards[self.current_card_idx]
            self.grid[y][x] = card_type
            self.cards_placed_count += 1
            
            # Cycle to next available card
            self.current_card_idx = (self.current_card_idx + 1) % len(self.unlocked_cards)
            
            # Sound: card_place.wav
            pos = self._grid_to_pixel(x, y)
            self._create_particles(pos[0], pos[1], self.CARD_TYPES[card_type]['color'], 10, 2.0)

    def _activate_traps(self):
        reward = 0.0
        placed_cards = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] != '':
                    placed_cards.append({'type': self.grid[r][c], 'pos': self._grid_to_pixel(c, r)})
        
        if not placed_cards:
            return 0.0

        # Phase 1: Stun
        for card in placed_cards:
            if card['type'] == 'STUN':
                props = self.CARD_TYPES['STUN']
                for creature in self.creatures:
                    if creature['pos'].distance_to(card['pos']) < props['radius']:
                        creature['stun_timer'] = props['duration']
                        self._create_particles(creature['pos'].x, creature['pos'].y, props['color'], 5, 1.0, 10)

        # Phase 2: Movement
        creature_forces = [pygame.Vector2(0, 0) for _ in self.creatures]
        for card in placed_cards:
            if card['type'] in ['ATTRACT', 'REPEL']:
                props = self.CARD_TYPES[card['type']]
                for i, creature in enumerate(self.creatures):
                    if creature['stun_timer'] > 0: continue
                    dist_vec = card['pos'] - creature['pos']
                    dist = dist_vec.length()
                    if 0 < dist < props['radius']:
                        force_magnitude = props['power'] * (1 - dist / props['radius'])
                        force = dist_vec.normalize() * force_magnitude
                        creature_forces[i] += force
        
        for i, creature in enumerate(self.creatures):
            creature['pos'] += creature_forces[i]
            self._clamp_creature_position(creature)

        # Phase 3: Capture
        captured_creatures = set()
        for card in placed_cards:
            if card['type'] == 'CAPTURE':
                props = self.CARD_TYPES['CAPTURE']
                for i, creature in enumerate(self.creatures):
                    if i in captured_creatures: continue
                    if creature['pos'].distance_to(card['pos']) < props['radius']:
                        # Sound: creature_capture.wav
                        self.score += creature['value']
                        reward += 1.0 + (4.0 if creature['type'] == 'rare' else 0.0)
                        captured_creatures.add(i)
                        self._create_particles(creature['pos'].x, creature['pos'].y, self.CREATURE_COLORS[creature['type']], 30, 4.0, 25)
        
        if captured_creatures:
            self.creatures = [c for i, c in enumerate(self.creatures) if i not in captured_creatures]
            self._check_unlocks()

        # Cleanup
        self.grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        
        return reward

    def _update_game_state(self):
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
            
        # Update Creatures
        creature_speed = self.BASE_CREATURE_SPEED + (self.score // 200) * 0.1
        for creature in self.creatures:
            if creature['stun_timer'] > 0:
                creature['stun_timer'] -= 1
            else:
                creature['pos'] += creature['vel'] * creature_speed
            
            # Wobble effect
            creature['wobble_angle'] += creature['wobble_speed']
            wobble_offset = pygame.Vector2(
                math.cos(creature['wobble_angle']) * creature['wobble_magnitude'],
                math.sin(creature['wobble_angle']) * creature['wobble_magnitude']
            )
            creature['draw_pos'] = creature['pos'] + wobble_offset

            # Wall bounce
            if not (0 < creature['pos'].x < self.SCREEN_WIDTH): creature['vel'].x *= -1
            if not (0 < creature['pos'].y < self.SCREEN_HEIGHT): creature['vel'].y *= -1
            self._clamp_creature_position(creature)

        # Update Particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            
        # Update BG Particles
        for p in self.bg_particles:
            p['pos'].y += p['vel'].y
            if p['pos'].y > self.SCREEN_HEIGHT:
                p['pos'].y = 0
                p['pos'].x = self.np_random.uniform(0, self.SCREEN_WIDTH)

    def _check_termination(self):
        return (self.score >= self.WIN_SCORE or 
                self.cards_placed_count >= self.MAX_CARDS)

    def _check_unlocks(self):
        if self.score >= 200 and 'REPEL' not in self.unlocked_cards:
            self.unlocked_cards.append('REPEL')
        if self.score >= 500 and 'STUN' not in self.unlocked_cards:
            self.unlocked_cards.append('STUN')

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cards_left": self.MAX_CARDS - self.cards_placed_count,
            "cursor_pos": tuple(self.cursor_pos),
        }

    def _render_background(self):
        # Drifting particles
        for p in self.bg_particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], p['radius'])

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_TOP + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_LEFT, y), (self.GRID_LEFT + self.GRID_WIDTH, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_LEFT + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_TOP), (x, self.GRID_TOP + self.GRID_HEIGHT), 1)

        # Draw creatures
        for creature in self.creatures:
            self._draw_glowing_circle(self.screen, creature['draw_pos'], creature['radius'], self.CREATURE_COLORS[creature['type']])
            if creature['stun_timer'] > 0:
                stun_text = self.font_small.render("Zzz", True, (255,255,255))
                self.screen.blit(stun_text, (creature['draw_pos'].x - 10, creature['draw_pos'].y - 25))

        # Draw placed cards
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                card_type = self.grid[r][c]
                if card_type:
                    props = self.CARD_TYPES[card_type]
                    px, py = self._grid_to_pixel(c, r)
                    rect = pygame.Rect(px - 18, py - 18, 36, 36)
                    pygame.draw.rect(self.screen, props['color'], rect, 0, 4)
                    symbol_surf = self.font_card.render(props['symbol'], True, self.COLOR_BG)
                    self.screen.blit(symbol_surf, symbol_surf.get_rect(center=rect.center))

        # Draw cursor
        cursor_px, cursor_py = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        cursor_rect = pygame.Rect(cursor_px - self.CELL_SIZE / 2, cursor_py - self.CELL_SIZE / 2, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, 5)
        
        # Draw card placement preview
        card_to_place = self.unlocked_cards[self.current_card_idx]
        props = self.CARD_TYPES[card_to_place]
        pygame.gfxdraw.aacircle(self.screen, int(cursor_px), int(cursor_py), props['radius'], (*props['color'], 80))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            if alpha > 0:
                pygame.draw.circle(self.screen, color, p['pos'], int(p['life'] / p['max_life'] * p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 15, 10))

        # Cards Left
        cards_text = self.font_large.render(f"CARDS: {self.MAX_CARDS - self.cards_placed_count}", True, self.COLOR_TEXT)
        self.screen.blit(cards_text, (15, 10))

        # Current Card
        card_type = self.unlocked_cards[self.current_card_idx]
        props = self.CARD_TYPES[card_type]
        ui_text = self.font_small.render(f"NEXT: {card_type}", True, props['color'])
        self.screen.blit(ui_text, (self.GRID_LEFT, self.GRID_TOP + self.GRID_HEIGHT + 10))

    # --- Helper Functions ---
    def _spawn_creatures(self, num_creatures):
        for _ in range(num_creatures):
            roll = self.np_random.random()
            if roll < 0.1:
                c_type, val, radius, color = 'rare', 50, 12, self.CREATURE_COLORS['rare']
            elif roll < 0.4:
                c_type, val, radius, color = 'uncommon', 20, 9, self.CREATURE_COLORS['uncommon']
            else:
                c_type, val, radius, color = 'common', 10, 7, self.CREATURE_COLORS['common']

            self.creatures.append({
                'pos': pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                'draw_pos': pygame.Vector2(0,0),
                'vel': pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize(),
                'type': c_type, 'value': val, 'radius': radius,
                'stun_timer': 0,
                'wobble_angle': self.np_random.uniform(0, 2 * math.pi),
                'wobble_speed': self.np_random.uniform(0.05, 0.1),
                'wobble_magnitude': self.np_random.uniform(2, 5)
            })

    def _init_bg_particles(self, num):
        for _ in range(num):
            self.bg_particles.append({
                'pos': pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                'vel': pygame.Vector2(0, self.np_random.uniform(0.1, 0.3)),
                'radius': self.np_random.integers(1, 3),
                'color': (30, 50, 90, 150)
            })

    def _create_particles(self, x, y, color, count, speed_scale, size=5.0, life=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * speed_scale
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(x, y),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'size': size
            })

    def _clamp_creature_position(self, creature):
        creature['pos'].x = np.clip(creature['pos'].x, 0, self.SCREEN_WIDTH)
        creature['pos'].y = np.clip(creature['pos'].y, 0, self.SCREEN_HEIGHT)
        
    def _grid_to_pixel(self, c, r):
        x = self.GRID_LEFT + c * self.CELL_SIZE + self.CELL_SIZE / 2
        y = self.GRID_TOP + r * self.CELL_SIZE + self.CELL_SIZE / 2
        return int(x), int(y)

    @staticmethod
    def _draw_glowing_circle(surface, pos, radius, color):
        x, y = int(pos.x), int(pos.y)
        glow_color = (*color, 20)
        # Draw multiple semi-transparent circles for a glow effect
        for i in range(radius, radius + 5):
            pygame.gfxdraw.aacircle(surface, x, y, i, glow_color)
        pygame.gfxdraw.filled_circle(surface, x, y, radius, color)
        pygame.gfxdraw.aacircle(surface, x, y, radius, color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play ---
    # The following code is not part of the environment and is for testing purposes only.
    # It will not be part of the final package.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display for manual play
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Deep Sea Trap Cards")
    clock = pygame.time.Clock()
    
    game_over = False
    total_reward = 0
    
    # Action state
    action = [0, 0, 0] # [movement, space, shift]

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        # --- Keyboard to Action Mapping ---
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No-op
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space and Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        game_over = terminated or truncated
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

        if game_over:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Optional: wait a bit before auto-closing
            pygame.time.wait(3000)

    env.close()