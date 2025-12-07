import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:26:24.751082
# Source Brief: brief_00411.md
# Brief Index: 411
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
    A Gymnasium environment for a roguelike card battler.

    The player uses a cursor to select cards from their hand, place them on a grid,
    and then teleport a card on the grid to trigger an area-of-effect attack.
    The goal is to defeat all quantum opponents before the player's health reaches zero.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0] (Movement): 0:None, 1:Up, 2:Down, 3:Left, 4:Right
    - action[1] (Space): 0:Released, 1:Pressed (Confirm)
    - action[2] (Shift): 0:Released, 1:Pressed (Cancel/Skip)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A roguelike card battler where you place cards on a grid and teleport them "
        "to unleash area-of-effect attacks against quantum opponents."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your cursor. "
        "Press space to select/confirm and shift to cancel/skip your turn."
    )
    auto_advance = False


    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 4, 6
    GRID_X_OFFSET, GRID_Y_OFFSET = 20, 20
    CELL_SIZE = 60
    CARD_WIDTH, CARD_HEIGHT = 50, 70
    HAND_Y_POS = 320
    HAND_CARD_WIDTH, HAND_CARD_HEIGHT = 40, 56

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (30, 50, 80)
    COLOR_PLAYER_HP = (0, 150, 255)
    COLOR_ENEMY_HP = (255, 50, 50)
    COLOR_HP_BG = (50, 50, 50)
    COLOR_WHITE = (240, 240, 240)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CARD_ATTACK = (220, 60, 60)
    COLOR_CARD_DEFEND = (60, 120, 220)
    COLOR_ENEMY_IDLE = (180, 100, 255)
    COLOR_ENEMY_ATTACK = (255, 100, 100)
    COLOR_ENEMY_DEFEND = (100, 180, 255)

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
        self.font_small = pygame.font.SysFont("Consolas", 12, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game State Attributes (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.level = None
        self.player = None
        self.enemies = None
        self.grid = None
        self.deck = None
        self.hand = None
        self.particles = None
        self.floating_texts = None
        
        self.game_phase = None
        self.hand_cursor_pos = None
        self.grid_cursor_pos = None
        self.selected_card_hand_idx = None
        self.selected_card_grid_pos = None
        
        self.last_space_press = False
        self.last_shift_press = False
        
        self.pending_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.pending_reward = 0.0

        self.player = {"health": 100, "max_health": 100}
        
        self._create_deck()
        self.hand = []
        self._draw_cards(5)
        
        self.grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self._spawn_enemies()

        self.particles = []
        self.floating_texts = []

        self.game_phase = "PLAYER_SELECT_HAND" # PLAYER_SELECT_HAND, PLAYER_SELECT_GRID_PLACE, PLAYER_SELECT_GRID_TELEPORT_SRC, PLAYER_SELECT_GRID_TELEPORT_DST, ENEMY_TURN, ANIMATION
        self.hand_cursor_pos = 0
        self.grid_cursor_pos = [0, 0]
        self.selected_card_hand_idx = None
        self.selected_card_grid_pos = None
        
        self.last_space_press = False
        self.last_shift_press = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.pending_reward = 0.0
        
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # Detect rising edge for button presses
        space_just_pressed = space_pressed and not self.last_space_press
        shift_just_pressed = shift_pressed and not self.last_shift_press
        self.last_space_press = space_pressed
        self.last_shift_press = shift_pressed

        if self.game_phase.startswith("PLAYER"):
            self._handle_player_action(movement, space_just_pressed, shift_just_pressed)
        
        if self.game_phase == "ENEMY_TURN":
            self._handle_enemy_turn()

        self._update_animations()

        reward = self.pending_reward
        self.score += reward
        
        terminated = self._check_termination()
        if terminated:
            if self.player['health'] > 0:
                reward += 100 # Win bonus
                self.score += 100
            else:
                reward -= 100 # Loss penalty
                self.score -= 100

        truncated = self.steps >= 1000
        if truncated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_player_action(self, movement, space_just_pressed, shift_just_pressed):
        if self.game_phase == "PLAYER_SELECT_HAND":
            if movement == 1: self.hand_cursor_pos = max(0, self.hand_cursor_pos - 1) # Up
            if movement == 2: self.hand_cursor_pos = min(len(self.hand) - 1 if self.hand else 0, self.hand_cursor_pos + 1) # Down
            
            if space_just_pressed and self.hand:
                self.selected_card_hand_idx = self.hand_cursor_pos
                self.game_phase = "PLAYER_SELECT_GRID_PLACE"
                # sfx: UI_confirm
            elif shift_just_pressed: # Skip turn
                self.game_phase = "ENEMY_TURN"
                # sfx: UI_cancel

        elif self.game_phase == "PLAYER_SELECT_GRID_PLACE":
            self._move_grid_cursor(movement)
            if space_just_pressed:
                r, c = self.grid_cursor_pos
                if self.grid[r][c] is None:
                    card_to_place = self.hand.pop(self.selected_card_hand_idx)
                    self.grid[r][c] = card_to_place
                    self.hand_cursor_pos = min(self.hand_cursor_pos, len(self.hand) - 1 if self.hand else 0)
                    self.game_phase = "PLAYER_SELECT_GRID_TELEPORT_SRC"
                    # sfx: card_place
            elif shift_just_pressed:
                self.game_phase = "PLAYER_SELECT_HAND"
                self.selected_card_hand_idx = None
                # sfx: UI_back
        
        elif self.game_phase == "PLAYER_SELECT_GRID_TELEPORT_SRC":
            self._move_grid_cursor(movement)
            if space_just_pressed:
                r, c = self.grid_cursor_pos
                if self.grid[r][c] is not None:
                    self.selected_card_grid_pos = (r, c)
                    self.game_phase = "PLAYER_SELECT_GRID_TELEPORT_DST"
                    # sfx: UI_confirm_special
            elif shift_just_pressed: # Skip teleport phase
                self.game_phase = "ENEMY_TURN"
                # sfx: UI_cancel
        
        elif self.game_phase == "PLAYER_SELECT_GRID_TELEPORT_DST":
            self._move_grid_cursor(movement)
            if space_just_pressed:
                r, c = self.grid_cursor_pos
                if self.grid[r][c] is None and (r,c) != self.selected_card_grid_pos:
                    self._execute_teleport((r, c))
                    self.game_phase = "ENEMY_TURN"
            elif shift_just_pressed:
                self.game_phase = "PLAYER_SELECT_GRID_TELEPORT_SRC"
                self.selected_card_grid_pos = None
                # sfx: UI_back

    def _move_grid_cursor(self, movement):
        if movement == 1: self.grid_cursor_pos[0] = max(0, self.grid_cursor_pos[0] - 1) # Up
        if movement == 2: self.grid_cursor_pos[0] = min(self.GRID_ROWS - 1, self.grid_cursor_pos[0] + 1) # Down
        if movement == 3: self.grid_cursor_pos[1] = max(0, self.grid_cursor_pos[1] - 1) # Left
        if movement == 4: self.grid_cursor_pos[1] = min(self.GRID_COLS - 1, self.grid_cursor_pos[1] + 1) # Right

    def _execute_teleport(self, to_pos):
        from_pos = self.selected_card_grid_pos
        card = self.grid[from_pos[0]][from_pos[1]]
        
        self.grid[from_pos[0]][from_pos[1]] = None
        self.grid[to_pos[0]][to_pos[1]] = card

        # sfx: teleport_whoosh
        self._create_teleport_effect(from_pos, to_pos)

        # Resolve AoE damage
        # sfx: explosion
        self._create_shockwave_effect(to_pos, card['color'])
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = to_pos[0] + dr, to_pos[1] + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                    for enemy in self.enemies:
                        if enemy['pos'] == (nr, nc):
                            damage = card['attack']
                            if enemy['state'] == 'DEFENDING':
                                damage //= 2
                            enemy['health'] -= damage
                            self.pending_reward += damage * 0.1
                            self._create_floating_text(f"-{damage}", self._grid_to_pixel(enemy['pos']), self.COLOR_ENEMY_HP)
                            # sfx: damage_hit

        # Check for defeated enemies
        enemies_alive = []
        for enemy in self.enemies:
            if enemy['health'] > 0:
                enemies_alive.append(enemy)
            else:
                self.pending_reward += 1.0 # Defeat bonus
                self._create_explosion(self._grid_to_pixel(enemy['pos']), enemy['color'])
                # sfx: enemy_defeat
        self.enemies = enemies_alive
        
        self.selected_card_grid_pos = None

    def _handle_enemy_turn(self):
        if not self.enemies:
            self.game_phase = "PLAYER_SELECT_HAND"
            return

        all_cards = [card for row in self.grid for card in row if card]

        for enemy in self.enemies:
            # Update state machine
            enemy['state_timer'] -= 1
            if enemy['state_timer'] <= 0:
                enemy['state_timer'] = 3
                if enemy['state'] == 'IDLE': enemy['state'] = 'ATTACKING'
                elif enemy['state'] == 'ATTACKING': enemy['state'] = 'DEFENDING'
                elif enemy['state'] == 'DEFENDING': enemy['state'] = 'IDLE'
            
            if enemy['state'] == 'ATTACKING':
                target = None
                min_dist = float('inf')

                # Find closest card
                if all_cards:
                    enemy_px = self._grid_to_pixel(enemy['pos'])
                    for r, row in enumerate(self.grid):
                        for c, card in enumerate(row):
                            if card:
                                card_px = self._grid_to_pixel((r,c))
                                dist = math.hypot(enemy_px[0]-card_px[0], enemy_px[1]-card_px[1])
                                if dist < min_dist:
                                    min_dist = dist
                                    target = (r,c)
                
                damage = int(5 * (1.05 ** (self.level - 1)))
                if target:
                    # Attack card
                    r, c = target
                    self.grid[r][c]['defense'] -= damage
                    self._create_floating_text(f"-{damage}", self._grid_to_pixel(target), self.COLOR_ENEMY_ATTACK)
                    if self.grid[r][c]['defense'] <= 0:
                        self._create_explosion(self._grid_to_pixel(target), self.grid[r][c]['color'])
                        self.grid[r][c] = None
                        # sfx: card_destroy
                else:
                    # Attack player
                    self.player['health'] -= damage
                    self.pending_reward -= damage * 0.1
                    self._create_floating_text(f"-{damage}", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30), self.COLOR_PLAYER_HP)
                    # sfx: player_damage

        # End of enemy turn
        self.game_phase = "PLAYER_SELECT_HAND"
        self._draw_cards(1)
        if not self.hand: self._draw_cards(1) # Ensure player always has a card if deck is not empty

    def _check_termination(self):
        if self.player['health'] <= 0:
            self.game_over = True
            return True
        if not self.enemies:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level, "player_health": self.player['health'], "enemies_left": len(self.enemies)}

    # --- Rendering Methods ---

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_entities()
        self._render_hand()
        self._render_cursors()
        self._render_effects()
        self._render_ui()

    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_COLS * self.CELL_SIZE, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_ROWS * self.CELL_SIZE), 1)

    def _render_entities(self):
        # Render cards on grid
        for r, row in enumerate(self.grid):
            for c, card in enumerate(row):
                if card:
                    self._draw_card(card, self._grid_to_pixel((r, c)), (self.CARD_WIDTH, self.CARD_HEIGHT))
        
        # Render enemies
        for enemy in self.enemies:
            px, py = self._grid_to_pixel(enemy['pos'])
            
            # State-based color and shape
            if enemy['state'] == 'IDLE':
                color = self.COLOR_ENEMY_IDLE
                points = [(px, py - 15), (px + 15, py + 10), (px - 15, py + 10)]
            elif enemy['state'] == 'ATTACKING':
                color = self.COLOR_ENEMY_ATTACK
                points = [(px, py - 15), (px + 13, py - 5), (px + 8, py + 15), (px - 8, py + 15), (px - 13, py - 5)]
            else: # DEFENDING
                color = self.COLOR_ENEMY_DEFEND
                points = [(px - 15, py - 15), (px + 15, py - 15), (px + 15, py + 15), (px - 15, py + 15)]

            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            enemy['color'] = color

            # Health bar
            hp_ratio = max(0, enemy['health'] / enemy['max_health'])
            pygame.draw.rect(self.screen, self.COLOR_HP_BG, (px - 20, py + 20, 40, 5))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_HP, (px - 20, py + 20, 40 * hp_ratio, 5))
            
    def _render_hand(self):
        if not self.hand: return
        total_width = len(self.hand) * (self.HAND_CARD_WIDTH + 5) - 5
        start_x = self.SCREEN_WIDTH // 2 - total_width // 2
        for i, card in enumerate(self.hand):
            pos = (start_x + i * (self.HAND_CARD_WIDTH + 5) + self.HAND_CARD_WIDTH // 2, self.HAND_Y_POS + self.HAND_CARD_HEIGHT // 2)
            self._draw_card(card, pos, (self.HAND_CARD_WIDTH, self.HAND_CARD_HEIGHT), is_in_hand=True, is_selected=(i == self.hand_cursor_pos and self.game_phase == "PLAYER_SELECT_HAND"))

    def _draw_card(self, card, center_pos, size, is_in_hand=False, is_selected=False):
        w, h = size
        rect = pygame.Rect(center_pos[0] - w//2, center_pos[1] - h//2, w, h)
        
        border_color = self.COLOR_CURSOR if is_selected else card['color']
        pygame.draw.rect(self.screen, (20, 30, 50), rect)
        pygame.draw.rect(self.screen, border_color, rect, 2 if is_selected else 1)
        
        if is_selected:
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect.inflate(4, 4), 1)

        atk_text = self.font_small.render(f"A:{card['attack']}", True, self.COLOR_WHITE)
        def_text = self.font_small.render(f"D:{card['defense']}", True, self.COLOR_WHITE)
        self.screen.blit(atk_text, (rect.x + 4, rect.y + 4))
        self.screen.blit(def_text, (rect.x + 4, rect.y + h - 16))

    def _render_cursors(self):
        if self.game_phase.startswith("PLAYER_SELECT_GRID"):
            r, c = self.grid_cursor_pos
            px, py = self._grid_to_pixel((r,c))
            rect = pygame.Rect(px - self.CELL_SIZE//2, py - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)
            
            if self.game_phase == "PLAYER_SELECT_GRID_TELEPORT_DST" and self.selected_card_grid_pos:
                sr, sc = self.selected_card_grid_pos
                spx, spy = self._grid_to_pixel((sr, sc))
                pygame.draw.line(self.screen, self.COLOR_CURSOR, (spx, spy), (px, py), 2)

    def _render_effects(self):
        # Particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            radius = int(p['size'] * (p['lifespan'] / p['max_lifespan']))
            if radius > 0:
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

        # Floating texts
        for ft in self.floating_texts:
            ft['pos'][1] -= 0.5
            ft['lifespan'] -= 1
            alpha = int(255 * (ft['lifespan'] / ft['max_lifespan']))
            if alpha > 0:
                text_surf = self.font_medium.render(ft['text'], True, ft['color'])
                text_surf.set_alpha(alpha)
                self.screen.blit(text_surf, (int(ft['pos'][0]), int(ft['pos'][1])))
        self.floating_texts = [ft for ft in self.floating_texts if ft['lifespan'] > 0]

    def _render_ui(self):
        # Player HP Bar
        hp_ratio = max(0, self.player['health'] / self.player['max_health'])
        pygame.draw.rect(self.screen, self.COLOR_HP_BG, (10, self.SCREEN_HEIGHT - 20, 200, 15))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_HP, (10, self.SCREEN_HEIGHT - 20, 200 * hp_ratio, 15))
        hp_text = self.font_medium.render(f"HP: {self.player['health']}/{self.player['max_health']}", True, self.COLOR_WHITE)
        self.screen.blit(hp_text, (15, self.SCREEN_HEIGHT - 22))

        # Score and Phase
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - 150, 5))
        phase_text = self.font_medium.render(f"Phase: {self.game_phase}", True, self.COLOR_WHITE)
        self.screen.blit(phase_text, (self.SCREEN_WIDTH - 250, self.SCREEN_HEIGHT - 22))

    # --- Helper & Logic Methods ---
    
    def _grid_to_pixel(self, pos):
        r, c = pos
        x = self.GRID_X_OFFSET + c * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_Y_OFFSET + r * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def _create_deck(self):
        self.deck = deque()
        for _ in range(10):
            self.deck.append({'type': 'attack', 'attack': random.randint(8, 12), 'defense': random.randint(3, 7), 'color': self.COLOR_CARD_ATTACK})
        for _ in range(10):
            self.deck.append({'type': 'defend', 'attack': random.randint(3, 7), 'defense': random.randint(8, 12), 'color': self.COLOR_CARD_DEFEND})
        random.shuffle(self.deck)
        
    def _draw_cards(self, num):
        for _ in range(num):
            if self.deck and len(self.hand) < 8:
                self.hand.append(self.deck.popleft())

    def _spawn_enemies(self):
        self.enemies = []
        num_enemies = random.randint(2, 3)
        available_cells = []
        for c in range(self.GRID_COLS-2, self.GRID_COLS):
            for r in range(self.GRID_ROWS):
                available_cells.append((r,c))
        
        random.shuffle(available_cells)
        
        for i in range(num_enemies):
            if not available_cells: break
            pos = available_cells.pop()
            base_health = int(50 * (1.05 ** (self.level - 1)))
            self.enemies.append({
                'pos': pos,
                'health': base_health,
                'max_health': base_health,
                'state': 'IDLE',
                'state_timer': random.randint(1,3),
                'color': self.COLOR_ENEMY_IDLE
            })

    def _create_floating_text(self, text, pos, color):
        self.floating_texts.append({
            'text': text, 'pos': [pos[0], pos[1]], 'color': color, 
            'lifespan': 60, 'max_lifespan': 60
        })

    def _create_explosion(self, pos, color):
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [pos[0], pos[1]],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'size': random.uniform(2, 6),
                'lifespan': random.randint(20, 40),
                'max_lifespan': 40
            })

    def _create_shockwave_effect(self, pos, color):
        px, py = self._grid_to_pixel(pos)
        for i in range(1, 6):
            self.particles.append({
                'pos': [px, py], 'vel': [0, 0], 'color': color,
                'size': i * 15, 'lifespan': 5 + i * 2, 'max_lifespan': 5 + i * 2,
                'type': 'shockwave' # Custom behavior could be added
            })

    def _create_teleport_effect(self, from_pos, to_pos):
        start_px = self._grid_to_pixel(from_pos)
        end_px = self._grid_to_pixel(to_pos)
        for i in range(20):
            t = i / 19.0
            pos = (start_px[0] * (1-t) + end_px[0] * t, start_px[1] * (1-t) + end_px[1] * t)
            self.particles.append({
                'pos': list(pos), 'vel': [0,0], 'color': self.COLOR_CURSOR,
                'size': 2, 'lifespan': 10, 'max_lifespan': 10
            })
            
    def _update_animations(self):
        # This is currently handled inside _render_effects for simplicity.
        # A more complex system would update positions here and render separately.
        pass

    def close(self):
        pygame.quit()

# Example usage:
if __name__ == '__main__':
    # To run with display, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Controls ---
    # Arrows: Move cursor
    # Space: Confirm
    # Shift: Cancel / Skip
    # Q: Quit
    
    running = True
    terminated = False
    
    # Use pygame for human interaction
    pygame.display.set_caption("Quantum Card Battle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    # Track held keys for continuous action
    keys_held = {
        pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False,
        pygame.K_SPACE: False, pygame.K_LSHIFT: False, pygame.K_RSHIFT: False
    }

    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if event.key in keys_held:
                    keys_held[event.key] = True
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        # Map held keys to actions
        if keys_held[pygame.K_UP]: action[0] = 1
        elif keys_held[pygame.K_DOWN]: action[0] = 2
        elif keys_held[pygame.K_LEFT]: action[0] = 3
        elif keys_held[pygame.K_RIGHT]: action[0] = 4
        
        if keys_held[pygame.K_SPACE]: action[1] = 1
        if keys_held[pygame.K_LSHIFT] or keys_held[pygame.K_RSHIFT]: action[2] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            if terminated or truncated:
                print("--- GAME OVER ---")
                print(f"Final Score: {info['score']:.2f}, Steps: {info['steps']}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()