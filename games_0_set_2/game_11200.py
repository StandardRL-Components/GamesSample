import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:37:26.870321
# Source Brief: brief_01200.md
# Brief Index: 1200
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Constants ---

# Screen Dimensions
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 400

# Colors
COLOR_BG = (15, 10, 30)
COLOR_GRID = (40, 30, 60)
COLOR_TEXT = (220, 220, 240)
COLOR_TEXT_SHADOW = (10, 5, 20)
COLOR_HP = (40, 200, 80)
COLOR_HP_BG = (20, 80, 40)
COLOR_MANA = (0, 180, 255)
COLOR_MANA_BG = (0, 70, 100)
COLOR_ENEMY = (100, 100, 120)
COLOR_ENEMY_HIT = (255, 100, 100)
COLOR_PROJECTILE = (255, 200, 0)
COLOR_CURSOR = (255, 255, 255)
GEM_COLORS = [
    (255, 50, 50),   # Red
    (50, 255, 50),   # Green
    (50, 100, 255),  # Blue
    (255, 255, 50),  # Yellow
    (200, 50, 255),  # Purple
]

# Game Board
BOARD_SIZE = 8
GEM_SIZE = 32
BOARD_X = 40
BOARD_Y = (SCREEN_HEIGHT - BOARD_SIZE * GEM_SIZE) // 2 # Center vertically

# Enemy Area
ENEMY_AREA_X = 320
ENEMY_AREA_Y = 40
ENEMY_AREA_WIDTH = SCREEN_WIDTH - ENEMY_AREA_X - 20
ENEMY_AREA_HEIGHT = SCREEN_HEIGHT - ENEMY_AREA_Y * 2

# Game Parameters
MAX_STEPS = 3000
NUM_WAVES = 5
MAX_HP = 100
MAX_MANA = 100

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Match gems to power up your attacks and defend against waves of incoming enemies in this puzzle-combat hybrid."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to swap the selected gem with the one in your last moved direction."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Game state variables are initialized in reset()
        self.board = []
        self.cursor_pos = [0, 0]
        self.last_move_dir = (0, 0)
        self.prev_space_held = False
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.player_hp = 0
        self.player_mana = 0
        self.current_wave = 0
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.game_state = 'IDLE' # 'IDLE', 'MATCHING', 'ATTACKING'
        self.state_timer = 0
        self.wave_clear_bonus_pending = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player_hp = MAX_HP
        self.player_mana = 0
        self.current_wave = 0
        self.wave_clear_bonus_pending = False
        
        self.cursor_pos = [BOARD_SIZE // 2, BOARD_SIZE // 2]
        self.last_move_dir = (1, 0) # Default to right
        self.prev_space_held = False
        
        self.enemies.clear()
        self.projectiles.clear()
        self.particles.clear()

        self._init_board()
        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01 # Small penalty to encourage action
        self.steps += 1
        
        # --- Update Game State based on Timers ---
        if self.state_timer > 0:
            self.state_timer -= 1
        else:
            if self.game_state == 'MATCHING':
                self._handle_fall()
                matches = self._find_matches()
                if not matches:
                    self.game_state = 'IDLE'
                    if self.wave_clear_bonus_pending:
                        reward += 5
                        self.score += 500
                        self.wave_clear_bonus_pending = False
                        self._start_next_wave()
                else:
                    reward += self._resolve_matches(matches)
            elif self.game_state == 'ATTACKING':
                 self.game_state = 'IDLE'
            else: # IDLE state
                 self.game_state = 'IDLE'

        # --- Handle Player Action only in IDLE state ---
        if self.game_state == 'IDLE' and not self.game_over:
            movement, space_held_bool, _ = self._unpack_action(action)
            self._handle_movement(movement)
            reward += self._handle_swap(space_held_bool)
            self.prev_space_held = space_held_bool

        # --- Update Game Entities ---
        if not self.game_over:
            self._update_enemies()
            reward += self._update_projectiles()
            self._update_particles()
        
        # --- Check for Automatic Attack ---
        if self.player_mana >= MAX_MANA and not self.game_over:
            self._trigger_attack()
            self.game_state = 'ATTACKING'
            self.state_timer = 15 # Duration of attack visual

        # --- Check Termination Conditions ---
        if self.player_hp <= 0 and not self.game_over:
            self.game_over = True
            self.game_won = False
            reward = -100
        
        if self.current_wave > NUM_WAVES and not self.enemies and not self.game_over:
            self.game_over = True
            self.game_won = True
            reward = 100

        terminated = self.game_over or self.steps >= MAX_STEPS
        truncated = self.steps >= MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Action Handling ---
    def _unpack_action(self, action):
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        return movement, space_held, shift_held

    def _handle_movement(self, movement):
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.last_move_dir = (dx, dy)
            self.cursor_pos[0] = max(0, min(BOARD_SIZE - 1, self.cursor_pos[0] + dx))
            self.cursor_pos[1] = max(0, min(BOARD_SIZE - 1, self.cursor_pos[1] + dy))

    def _handle_swap(self, space_held):
        reward = 0
        is_pressed = space_held and not self.prev_space_held
        if not is_pressed:
            return reward

        cx, cy = self.cursor_pos
        dx, dy = self.last_move_dir
        nx, ny = cx + dx, cy + dy

        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            # Swap gems
            self.board[cy][cx], self.board[ny][nx] = self.board[ny][nx], self.board[cy][cx]
            
            # Check if swap creates a match
            matches = self._find_matches()
            if matches:
                # Valid swap
                # sfx: gem_match.wav
                self.game_state = 'MATCHING'
                reward += self._resolve_matches(matches)
            else:
                # Invalid swap, swap back
                # sfx: invalid_swap.wav
                self.board[cy][cx], self.board[ny][nx] = self.board[ny][nx], self.board[cy][cx]
        return reward

    # --- Game Logic ---
    def _init_board(self):
        self.board = [[self.np_random.integers(0, len(GEM_COLORS)) for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        while self._find_matches():
             self._resolve_matches(self._find_matches(), gain_mana=False)
             self._handle_fall()
        if not self._find_possible_moves():
            self._init_board()

    def _find_matches(self):
        matches = set()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] is None: continue
                # Horizontal
                if c < BOARD_SIZE - 2 and self.board[r][c] == self.board[r][c+1] == self.board[r][c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                # Vertical
                if r < BOARD_SIZE - 2 and self.board[r][c] == self.board[r+1][c] == self.board[r+2][c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _find_possible_moves(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                # Try swapping right
                if c < BOARD_SIZE - 1:
                    self.board[r][c], self.board[r][c+1] = self.board[r][c+1], self.board[r][c]
                    if self._find_matches():
                        self.board[r][c], self.board[r][c+1] = self.board[r][c+1], self.board[r][c]
                        return True
                    self.board[r][c], self.board[r][c+1] = self.board[r][c+1], self.board[r][c]
                # Try swapping down
                if r < BOARD_SIZE - 1:
                    self.board[r][c], self.board[r+1][c] = self.board[r+1][c], self.board[r][c]
                    if self._find_matches():
                        self.board[r][c], self.board[r+1][c] = self.board[r+1][c], self.board[r][c]
                        return True
                    self.board[r][c], self.board[r+1][c] = self.board[r+1][c], self.board[r][c]
        return False

    def _resolve_matches(self, matches, gain_mana=True):
        reward = 0
        mana_gain = 0
        for r, c in matches:
            if self.board[r][c] is not None:
                # sfx: gem_destroy.wav
                self._create_particles(c * GEM_SIZE + BOARD_X + GEM_SIZE//2, 
                                       r * GEM_SIZE + BOARD_Y + GEM_SIZE//2, 
                                       GEM_COLORS[self.board[r][c]])
                self.board[r][c] = None
                if gain_mana:
                    mana_gain += 5
                reward += 1
                self.score += 10
        
        if gain_mana:
            self.player_mana = min(MAX_MANA, self.player_mana + mana_gain)
        
        self.state_timer = 10 # time for gems to fall
        return reward

    def _handle_fall(self):
        for c in range(BOARD_SIZE):
            empty_row = BOARD_SIZE - 1
            for r in range(BOARD_SIZE - 1, -1, -1):
                if self.board[r][c] is not None:
                    if r != empty_row:
                        self.board[empty_row][c] = self.board[r][c]
                        self.board[r][c] = None
                    empty_row -= 1
        
        for c in range(BOARD_SIZE):
            for r in range(BOARD_SIZE):
                if self.board[r][c] is None:
                    self.board[r][c] = self.np_random.integers(0, len(GEM_COLORS))

        if not self._find_possible_moves():
            # sfx: board_shuffle.wav
            self._init_board()

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > NUM_WAVES:
            return
        
        num_enemies = self.current_wave
        base_hp = 100
        base_proj_speed = 2.0
        
        enemy_hp = base_hp * (1 + 0.2 * (self.current_wave - 1))
        proj_speed = base_proj_speed * (1 + 0.1 * (self.current_wave - 1))
        
        for i in range(num_enemies):
            self.enemies.append({
                'x': ENEMY_AREA_X + (i + 1) * (ENEMY_AREA_WIDTH / (num_enemies + 1)),
                'y': ENEMY_AREA_Y,
                'hp': enemy_hp,
                'max_hp': enemy_hp,
                'fire_cooldown': self.np_random.integers(60, 120),
                'proj_speed': proj_speed,
                'hit_timer': 0
            })
            
    def _update_enemies(self):
        cleared_enemies = []
        for enemy in self.enemies:
            enemy['y'] += 0.2 # Slow drift downwards
            if enemy['y'] > SCREEN_HEIGHT - 100:
                enemy['y'] = SCREEN_HEIGHT - 100

            if enemy['hit_timer'] > 0:
                enemy['hit_timer'] -= 1

            enemy['fire_cooldown'] -= 1
            if enemy['fire_cooldown'] <= 0:
                # sfx: enemy_fire.wav
                self.projectiles.append({
                    'x': enemy['x'],
                    'y': enemy['y'] + 20,
                    'speed': enemy['proj_speed']
                })
                enemy['fire_cooldown'] = self.np_random.integers(100, 200)

            if enemy['hp'] <= 0:
                cleared_enemies.append(enemy)
        
        if cleared_enemies:
            # sfx: enemy_explode.wav
            for enemy in cleared_enemies:
                 self._create_particles(enemy['x'], enemy['y'], COLOR_ENEMY_HIT, count=30)
                 self.enemies.remove(enemy)
                 self.score += 100
            
            if not self.enemies:
                self.wave_clear_bonus_pending = True

    def _update_projectiles(self):
        reward = 0
        surviving_projectiles = []
        for proj in self.projectiles:
            proj['y'] += proj['speed']
            if proj['y'] > SCREEN_HEIGHT - 40: # Player hit line
                # sfx: player_hit.wav
                self.player_hp = max(0, self.player_hp - 10)
                reward -= 2
                self._create_particles(proj['x'], SCREEN_HEIGHT-40, COLOR_HP, count=15)
            else:
                surviving_projectiles.append(proj)
        self.projectiles = surviving_projectiles
        return reward

    def _trigger_attack(self):
        # sfx: player_attack.wav
        self.player_mana = 0
        damage = 50 
        for enemy in self.enemies:
            enemy['hp'] -= damage
            enemy['hit_timer'] = 10 
        
        self._create_particles(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, COLOR_MANA, 100, is_attack=True)

    def _create_particles(self, x, y, color, count=20, is_attack=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            if is_attack:
                speed *= 2
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(20, 40),
                'color': color
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw board grid
        for i in range(BOARD_SIZE + 1):
            pygame.draw.line(self.screen, COLOR_GRID, (BOARD_X + i * GEM_SIZE, BOARD_Y), (BOARD_X + i * GEM_SIZE, BOARD_Y + BOARD_SIZE * GEM_SIZE), 1)
            pygame.draw.line(self.screen, COLOR_GRID, (BOARD_X, BOARD_Y + i * GEM_SIZE), (BOARD_X + BOARD_SIZE * GEM_SIZE, BOARD_Y + i * GEM_SIZE), 1)

        # Draw gems
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                gem_type = self.board[r][c]
                if gem_type is not None:
                    color = GEM_COLORS[gem_type]
                    cx, cy = int(BOARD_X + c * GEM_SIZE + GEM_SIZE / 2), int(BOARD_Y + r * GEM_SIZE + GEM_SIZE / 2)
                    radius = int(GEM_SIZE / 2 * 0.8)
                    pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, tuple(min(255, x+50) for x in color))

        # Draw cursor
        if self.game_state == 'IDLE' and not self.game_over:
            cursor_alpha = 128 + 127 * math.sin(self.steps * 0.2)
            cursor_color = (*COLOR_CURSOR, cursor_alpha)
            rect = pygame.Rect(BOARD_X + self.cursor_pos[0] * GEM_SIZE, BOARD_Y + self.cursor_pos[1] * GEM_SIZE, GEM_SIZE, GEM_SIZE)
            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(surf, cursor_color, surf.get_rect(), 3, border_radius=4)
            self.screen.blit(surf, rect.topleft)

        # Draw enemies
        for enemy in self.enemies:
            color = COLOR_ENEMY_HIT if enemy['hit_timer'] > 0 else COLOR_ENEMY
            points = [
                (enemy['x'], enemy['y'] - 15),
                (enemy['x'] - 15, enemy['y'] + 10),
                (enemy['x'] + 15, enemy['y'] + 10)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            hp_ratio = max(0, enemy['hp'] / enemy['max_hp'])
            pygame.draw.rect(self.screen, (100,0,0), (enemy['x']-15, enemy['y']+15, 30, 5))
            pygame.draw.rect(self.screen, (200,0,0), (enemy['x']-15, enemy['y']+15, 30 * hp_ratio, 5))

        # Draw projectiles
        for proj in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj['x']), int(proj['y']), 4, COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, int(proj['x']), int(proj['y']), 4, tuple(min(255, x+50) for x in COLOR_PROJECTILE))
            
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 40.0))))
            color = (*p['color'], alpha)
            surf = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (3, 3), int(p['life'] / 10))
            self.screen.blit(surf, (p['x'] - 3, p['y'] - 3))

    def _render_ui(self):
        bar_width = 250
        # HP
        hp_ratio = max(0, self.player_hp / MAX_HP)
        pygame.draw.rect(self.screen, COLOR_HP_BG, (20, SCREEN_HEIGHT - 30, bar_width, 20), border_radius=4)
        pygame.draw.rect(self.screen, COLOR_HP, (20, SCREEN_HEIGHT - 30, bar_width * hp_ratio, 20), border_radius=4)
        self._draw_text("HP", 20 + bar_width + 10, SCREEN_HEIGHT - 30)
        # Mana
        mana_ratio = max(0, self.player_mana / MAX_MANA)
        pygame.draw.rect(self.screen, COLOR_MANA_BG, (SCREEN_WIDTH - 20 - bar_width, SCREEN_HEIGHT - 30, bar_width, 20), border_radius=4)
        pygame.draw.rect(self.screen, COLOR_MANA, (SCREEN_WIDTH - 20 - bar_width, SCREEN_HEIGHT - 30, bar_width * mana_ratio, 20), border_radius=4)
        self._draw_text("MANA", SCREEN_WIDTH - 20 - bar_width - 60, SCREEN_HEIGHT - 30)
        
        self._draw_text(f"SCORE: {self.score}", 20, 10)
        wave_text = f"WAVE: {self.current_wave}/{NUM_WAVES}" if self.current_wave <= NUM_WAVES else "VICTORY!"
        self._draw_text(wave_text, SCREEN_WIDTH - 150, 10)

        if self.game_over:
            text = "VICTORY!" if self.game_won else "GAME OVER"
            color = COLOR_HP if self.game_won else COLOR_ENEMY_HIT
            self._draw_text(text, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, font=self.font_large, center=True, color=color)

    def _draw_text(self, text, x, y, font=None, color=COLOR_TEXT, center=False):
        if font is None:
            font = self.font_small
        
        shadow = font.render(text, True, COLOR_TEXT_SHADOW)
        main_text = font.render(text, True, color)
        
        pos = (x, y)
        if center:
            pos = main_text.get_rect(center=(x, y))

        self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
        self.screen.blit(main_text, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "hp": self.player_hp,
            "mana": self.player_mana,
            "enemies": len(self.enemies)
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv()
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # The main loop is for interactive testing and visualization.
    # It does not affect the agent's headless operation.
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.set_caption("Gemstone Guardian")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = [0, 0, 0]
    
    while not terminated and not truncated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        
        action = [0, 0, 0]
        
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
        
    env.close()
    print("Game Over!")
    print(f"Final Info: {info}")