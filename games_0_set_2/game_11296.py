import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:45:40.380737
# Source Brief: brief_01296.md
# Brief Index: 1296
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An arcade puzzle/action game where the player matches circuits to power a robot's 
    defenses against waves of malfunctioning bots.

    - **Action Space**: `MultiDiscrete([5, 2, 2])`
      - `actions[0]`: Movement (0:none, 1:up, 2:down, 3:left, 4:right) for cursor control.
      - `actions[1]`: Space button (0:released, 1:held) for selecting/activating.
      - `actions[2]`: Shift button (0:released, 1:held) for switching modes.

    - **Observation Space**: `Box(shape=(400, 640, 3), dtype=uint8)`
      - A 640x400 RGB image of the game screen.

    - **Gameplay Loop**:
      1.  **Circuit Mode**: Match 3+ circuits on the grid to gain Energy.
      2.  **Deploy Mode**: Use Shift to switch modes. Spend Energy on:
          - **Repair**: Restore robot health.
          - **Shield**: Create a temporary shield.
          - **EMP Blast**: Damage all bots on screen.
      3.  **Defend**: Bots spawn in waves from the right and attack the robot.
      4.  **Survive**: Win by surviving all 10 waves. Lose if robot health reaches zero.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your robot from waves of enemies by matching circuits to power your defenses. "
        "Use energy to repair, shield, or unleash an EMP blast."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select/swap circuits or activate defenses. "
        "Press shift to switch between circuit-matching and defense-deployment modes."
    )
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 8, 6
    GRID_CELL_SIZE = 40
    GRID_TOP_LEFT = (WIDTH - GRID_COLS * GRID_CELL_SIZE - 20, 40)
    NUM_CIRCUIT_TYPES = 5
    MAX_STEPS = 3000
    TOTAL_WAVES = 10

    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_GRID_BG = (25, 30, 50)
    COLOR_ROBOT = (0, 255, 128)
    COLOR_SHIELD = (100, 200, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_EMP = (0, 128, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_ACCENT = (255, 255, 0)
    COLOR_HEALTH = (50, 205, 50)
    COLOR_ENERGY = (255, 190, 0)
    CIRCUIT_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255),   # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('consolas', 16, bold=True)
        self.font_medium = pygame.font.SysFont('consolas', 24, bold=True)
        self.font_large = pygame.font.SysFont('consolas', 32, bold=True)
        
        # Initialize all state variables to prevent attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.step_reward = 0.0
        
        self.robot_health = 0
        self.robot_max_health = 100
        self.energy = 0
        self.max_energy = 100
        
        self.wave_number = 0
        self.wave_cooldown = 0
        
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_circuit = None
        self.mode = 'circuit' # 'circuit' or 'deploy'
        self.deploy_cursor_pos = 0
        
        self.bots = []
        self.particles = []
        self.effects = [] # For EMP, etc.
        
        self.shield_health = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.robot_health = self.robot_max_health
        self.energy = 20
        
        self.wave_number = 0
        self.wave_cooldown = 150 # Initial delay before first wave
        
        self._init_grid()
        while not self._check_for_possible_matches():
            self._init_grid()
        self._find_and_clear_matches(reward=False) # Clear initial matches
        
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_circuit = None
        self.mode = 'circuit'
        self.deploy_cursor_pos = 0
        
        self.bots = []
        self.particles = []
        self.effects = []
        
        self.shield_health = 0
        
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.step_reward = 0.0
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        self._handle_input(movement, space_press, shift_press)
        self._update_bots()
        self._update_waves()
        self._update_effects()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            self.game_over = True
            if self.robot_health <= 0:
                self.step_reward -= 100
            elif self.wave_number > self.TOTAL_WAVES:
                self.step_reward += 100
        
        self.score += self.step_reward
        
        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            truncated,
            self._get_info()
        )
        
    # --- Game Logic ---

    def _handle_input(self, movement, space_press, shift_press):
        if shift_press:
            self.mode = 'deploy' if self.mode == 'circuit' else 'circuit'
            self.selected_circuit = None # Clear selection on mode switch

        if self.mode == 'circuit':
            self._handle_circuit_input(movement, space_press)
        else: # self.mode == 'deploy'
            self._handle_deploy_input(movement, space_press)

    def _handle_circuit_input(self, movement, space_press):
        # Move cursor
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        if movement == 2: self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
        if movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        if movement == 4: self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)
        
        if space_press:
            r, c = self.cursor_pos
            if self.selected_circuit is None:
                self.selected_circuit = [r, c]
            else:
                sr, sc = self.selected_circuit
                # Check for adjacency
                if abs(r - sr) + abs(c - sc) == 1:
                    self._swap_circuits(r, c, sr, sc)
                else: # Invalid selection, reset
                    self.selected_circuit = [r, c]

    def _handle_deploy_input(self, movement, space_press):
        menu_items = 3
        if movement == 1: self.deploy_cursor_pos = (self.deploy_cursor_pos - 1 + menu_items) % menu_items
        if movement == 2: self.deploy_cursor_pos = (self.deploy_cursor_pos + 1) % menu_items

        if space_press:
            costs = [25, 40, 75] # Repair, Shield, EMP
            if self.energy >= costs[self.deploy_cursor_pos]:
                self.energy -= costs[self.deploy_cursor_pos]
                self.step_reward += 1.0
                if self.deploy_cursor_pos == 0: # Repair
                    self.robot_health = min(self.robot_max_health, self.robot_health + 25)
                    # sfx: repair_sound
                elif self.deploy_cursor_pos == 1: # Shield
                    self.shield_health = 50
                    # sfx: shield_up_sound
                elif self.deploy_cursor_pos == 2: # EMP
                    self._create_emp_blast()
                    # sfx: emp_blast_sound

    def _swap_circuits(self, r1, c1, r2, c2):
        self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]
        if not self._find_and_clear_matches():
            # No match, swap back
            self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]
        self.selected_circuit = None

    def _find_and_clear_matches(self, reward=True):
        matches = set()
        # Find horizontal matches
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[r][c] is not None and self.grid[r][c] == self.grid[r][c+1] == self.grid[r][c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        
        # Find vertical matches
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if self.grid[r][c] is not None and self.grid[r][c] == self.grid[r+1][c] == self.grid[r+2][c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        
        if not matches:
            return False

        # sfx: match_sound
        for r, c in matches:
            if self.grid[r][c] is not None:
                if reward:
                    self.step_reward += 0.1
                    self.energy = min(self.max_energy, self.energy + 1)
                self._create_particles(r, c)
                self.grid[r][c] = None

        self._apply_gravity_and_refill()
        
        # Chain reaction check
        if self._find_and_clear_matches(reward):
            if reward: self.step_reward += 0.5 # Bonus for chain reaction

        if not self._check_for_possible_matches():
            self._init_grid() # Reshuffle if no moves left
            
        return True

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] is not None:
                    self.grid[empty_row][c], self.grid[r][c] = self.grid[r][c], self.grid[empty_row][c]
                    empty_row -= 1
            for r in range(empty_row, -1, -1):
                self.grid[r][c] = self.np_random.integers(0, self.NUM_CIRCUIT_TYPES)
    
    def _init_grid(self):
        self.grid = [[self.np_random.integers(0, self.NUM_CIRCUIT_TYPES) for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]

    def _check_for_possible_matches(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Check swap right
                if c < self.GRID_COLS - 1:
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                    if self._has_match_at(r, c) or self._has_match_at(r, c+1):
                        self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                        return True
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                # Check swap down
                if r < self.GRID_ROWS - 1:
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
                    if self._has_match_at(r, c) or self._has_match_at(r+1, c):
                        self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
                        return True
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
        return False

    def _has_match_at(self, r, c):
        val = self.grid[r][c]
        if val is None: return False
        # Horizontal
        if c > 0 and c < self.GRID_COLS - 1 and self.grid[r][c-1] == val and self.grid[r][c+1] == val: return True
        if c > 1 and self.grid[r][c-1] == val and self.grid[r][c-2] == val: return True
        if c < self.GRID_COLS - 2 and self.grid[r][c+1] == val and self.grid[r][c+2] == val: return True
        # Vertical
        if r > 0 and r < self.GRID_ROWS - 1 and self.grid[r-1][c] == val and self.grid[r+1][c] == val: return True
        if r > 1 and self.grid[r-1][c] == val and self.grid[r-2][c] == val: return True
        if r < self.GRID_ROWS - 2 and self.grid[r+1][c] == val and self.grid[r+2][c] == val: return True
        return False

    def _update_bots(self):
        robot_hitbox = pygame.Rect(10, self.HEIGHT/2 - 50, 80, 100)
        
        for bot in self.bots[:]:
            bot['pos'][0] -= bot['speed']
            
            bot_rect = pygame.Rect(bot['pos'][0] - bot['size']/2, bot['pos'][1] - bot['size']/2, bot['size'], bot['size'])
            if bot_rect.colliderect(robot_hitbox):
                damage = 5
                if self.shield_health > 0:
                    taken = min(self.shield_health, damage)
                    self.shield_health -= taken
                    damage -= taken
                    # sfx: shield_hit_sound
                
                if damage > 0:
                    self.robot_health -= damage
                    self.step_reward -= 0.5 * damage
                    self.effects.append({'type': 'flash', 'color': self.COLOR_ENEMY, 'duration': 5})
                    # sfx: robot_hit_sound
                
                self.bots.remove(bot)
                continue
            
            if bot['pos'][0] < -bot['size']:
                self.bots.remove(bot)
    
    def _update_waves(self):
        if not self.bots and self.wave_number <= self.TOTAL_WAVES:
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0:
                if self.wave_number > 0: # Don't reward for clearing wave 0
                    self.step_reward += 5.0
                self.wave_number += 1
                if self.wave_number <= self.TOTAL_WAVES:
                    self._spawn_wave()
                    self.wave_cooldown = 200 # Time between waves
    
    def _spawn_wave(self):
        num_bots = 2 + self.wave_number
        base_speed = 1.0 + self.wave_number * 0.1 + self.steps * 0.00005
        base_health = 10 + self.wave_number * 2 + self.steps * 0.001
        
        for _ in range(num_bots):
            self.bots.append({
                'pos': [self.WIDTH + self.np_random.uniform(20, 100), self.np_random.uniform(50, self.HEIGHT - 50)],
                'speed': base_speed + self.np_random.uniform(-0.2, 0.2),
                'health': base_health,
                'max_health': base_health,
                'size': self.np_random.integers(15, 25),
            })

    def _update_effects(self):
        # Particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        # Other effects (EMP, screen flash)
        for e in self.effects[:]:
            e['duration'] -= 1
            if e['type'] == 'emp':
                e['radius'] += 15
                for bot in self.bots:
                    dist = math.hypot(bot['pos'][0] - e['pos'][0], bot['pos'][1] - e['pos'][1])
                    if dist < e['radius'] and not bot.get('hit_by_emp', False):
                        bot['health'] -= 50
                        bot['hit_by_emp'] = True
                        if bot['health'] <= 0:
                            self.step_reward += 2.0
                            # sfx: bot_destroy_sound
                # Prune dead bots
                self.bots = [b for b in self.bots if b['health'] > 0]
            
            if e['duration'] <= 0:
                if e['type'] == 'emp':
                    for bot in self.bots: bot['hit_by_emp'] = False # Reset for next EMP
                self.effects.remove(e)

    def _create_particles(self, r, c):
        grid_x, grid_y = self.GRID_TOP_LEFT
        cell_type = self.grid[r][c]
        if cell_type is None: return
        color = self.CIRCUIT_COLORS[cell_type]
        px = grid_x + c * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE / 2
        py = grid_y + r * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE / 2
        for _ in range(10):
            self.particles.append({
                'pos': [px, py],
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)],
                'life': self.np_random.integers(10, 20),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _create_emp_blast(self):
        self.effects.append({
            'type': 'emp',
            'pos': (10 + 40, self.HEIGHT / 2),
            'radius': 0,
            'duration': 30
        })

    def _check_termination(self):
        return self.robot_health <= 0 or self.wave_number > self.TOTAL_WAVES
        
    # --- Rendering ---
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_grid()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_grid()
        self._render_robot()
        if self.shield_health > 0: self._render_shield()
        self._render_bots()
        self._render_particles()
        self._render_effects()
    
    def _render_background_grid(self):
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID_BG, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID_BG, (0, i), (self.WIDTH, i))

    def _render_grid(self):
        gx, gy = self.GRID_TOP_LEFT
        gw, gh = self.GRID_COLS * self.GRID_CELL_SIZE, self.GRID_ROWS * self.GRID_CELL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (gx - 5, gy - 5, gw + 10, gh + 10), border_radius=5)
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                cell_type = self.grid[r][c]
                if cell_type is not None:
                    color = self.CIRCUIT_COLORS[cell_type]
                    rect = pygame.Rect(gx + c * self.GRID_CELL_SIZE, gy + r * self.GRID_CELL_SIZE, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
                    
                    # Draw glow
                    glow_rect = rect.inflate(8, 8)
                    glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(glow_surf, (*color, 60), glow_surf.get_rect(), border_radius=8)
                    self.screen.blit(glow_surf, glow_rect.topleft)
                    
                    pygame.draw.rect(self.screen, color, rect.inflate(-4, -4), border_radius=5)
    
    def _render_robot(self):
        center_y = self.HEIGHT // 2
        # Body
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, (10, center_y - 50, 80, 100), border_radius=10)
        # Eye
        eye_color = self.COLOR_ENEMY if self.robot_health < self.robot_max_health * 0.3 else (255, 255, 255)
        pygame.draw.circle(self.screen, (0,0,0), (50, center_y - 20), 12)
        pygame.draw.circle(self.screen, eye_color, (50, center_y - 20), 10)
        pygame.draw.circle(self.screen, (255,255,255), (52, center_y - 22), 3)

    def _render_shield(self):
        center = (10 + 40, self.HEIGHT / 2)
        radius = 70
        alpha = int(100 + (self.shield_health / 50) * 100)
        pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), radius, (*self.COLOR_SHIELD, alpha))
        pygame.gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), radius, self.COLOR_SHIELD)

    def _render_bots(self):
        for bot in self.bots:
            pos = (int(bot['pos'][0]), int(bot['pos'][1]))
            size = int(bot['size'])
            rect = pygame.Rect(pos[0] - size//2, pos[1] - size//2, size, size)
            
            # Glow
            glow_rect = rect.inflate(6, 6)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_ENEMY, 80), glow_surf.get_rect(), border_radius=4)
            self.screen.blit(glow_surf, glow_rect.topleft)

            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=3)
            
            # Health bar
            if bot['health'] < bot['max_health']:
                bar_w = 30
                bar_h = 5
                bar_x = pos[0] - bar_w // 2
                bar_y = pos[1] - size // 2 - 10
                health_pct = bot['health'] / bot['max_health']
                pygame.draw.rect(self.screen, (80, 0, 0), (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH, (bar_x, bar_y, bar_w * health_pct, bar_h))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20.0))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))
    
    def _render_effects(self):
        for e in self.effects:
            if e['type'] == 'emp':
                alpha = int(max(0, 255 * (e['duration'] / 20.0)))
                pygame.gfxdraw.filled_circle(self.screen, int(e['pos'][0]), int(e['pos'][1]), int(e['radius']), (*self.COLOR_EMP, alpha // 4))
                pygame.gfxdraw.aacircle(self.screen, int(e['pos'][0]), int(e['pos'][1]), int(e['radius']), (*self.COLOR_EMP, alpha))
            elif e['type'] == 'flash':
                alpha = int(max(0, 255 * (e['duration'] / 5.0)))
                flash_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                flash_surf.fill((*e['color'], alpha))
                self.screen.blit(flash_surf, (0,0))
    
    def _render_ui(self):
        self._render_hud()
        self._render_cursor()
        if self.mode == 'deploy':
            self._render_deploy_menu()
        
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            win_text = "SYSTEM STABILIZED" if self.wave_number > self.TOTAL_WAVES else "SYSTEM FAILURE"
            text_surf = self.font_large.render(win_text, True, self.COLOR_UI_ACCENT)
            self.screen.blit(text_surf, (self.WIDTH/2 - text_surf.get_width()/2, self.HEIGHT/2 - text_surf.get_height()/2))

    def _render_hud(self):
        # Health Bar
        health_pct = max(0, self.robot_health / self.robot_max_health)
        pygame.draw.rect(self.screen, (80, 0, 0), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, 200 * health_pct, 20))
        
        # Energy Bar
        energy_pct = max(0, self.energy / self.max_energy)
        pygame.draw.rect(self.screen, (80, 60, 0), (10, 35, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (10, 35, 200 * energy_pct, 20))
        
        # Score
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Wave
        wave_text = self.font_medium.render(f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 40))
        
        # Mode
        mode_text = self.font_small.render(f"MODE: {self.mode.upper()}", True, self.COLOR_UI_ACCENT)
        self.screen.blit(mode_text, (10, 60))

    def _render_cursor(self):
        if self.mode == 'circuit':
            gx, gy = self.GRID_TOP_LEFT
            r, c = self.cursor_pos
            
            # Main cursor
            cursor_rect = pygame.Rect(gx + c * self.GRID_CELL_SIZE, gy + r * self.GRID_CELL_SIZE, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, cursor_rect, 3, border_radius=5)
            
            # Selected circuit highlight
            if self.selected_circuit:
                sr, sc = self.selected_circuit
                sel_rect = pygame.Rect(gx + sc * self.GRID_CELL_SIZE, gy + sr * self.GRID_CELL_SIZE, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
                pygame.draw.rect(self.screen, (255, 255, 255), sel_rect, 2, border_radius=5)

    def _render_deploy_menu(self):
        menu_items = ["REPAIR (25)", "SHIELD (40)", "EMP BLAST (75)"]
        costs = [25, 40, 75]
        menu_x, menu_y = 10, self.HEIGHT - 100
        
        for i, item in enumerate(menu_items):
            color = self.COLOR_UI_TEXT if self.energy >= costs[i] else (128, 128, 128)
            text_surf = self.font_small.render(item, True, color)
            self.screen.blit(text_surf, (menu_x + 20, menu_y + i * 25))
            if i == self.deploy_cursor_pos:
                pygame.draw.polygon(self.screen, self.COLOR_UI_ACCENT, [(menu_x, menu_y + i * 25 + 7), (menu_x + 10, menu_y + i * 25 + 12), (menu_x, menu_y + i * 25 + 17)])

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "robot_health": self.robot_health,
            "energy": self.energy,
            "wave": self.wave_number,
            "mode": self.mode
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is recommended to use the environment's render_mode if available
    # This block will not be run during evaluation.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display for manual play
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Circuit Defender")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Replace the manual action polling with a more robust system
    # This example uses a simplified action state for demonstration
    action = [0, 0, 0] # [movement, space, shift]
    
    # Use a dictionary to track key presses for continuous actions
    key_state = {
        "up": False, "down": False, "left": False, "right": False,
        "space": False, "shift": False
    }

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: key_state["up"] = True
                if event.key == pygame.K_DOWN: key_state["down"] = True
                if event.key == pygame.K_LEFT: key_state["left"] = True
                if event.key == pygame.K_RIGHT: key_state["right"] = True
                if event.key == pygame.K_SPACE: key_state["space"] = True
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: key_state["shift"] = True
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    total_reward = 0
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP: key_state["up"] = False
                if event.key == pygame.K_DOWN: key_state["down"] = False
                if event.key == pygame.K_LEFT: key_state["left"] = False
                if event.key == pygame.K_RIGHT: key_state["right"] = False
                if event.key == pygame.K_SPACE: key_state["space"] = False
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: key_state["shift"] = False

        # Map key state to environment action
        movement = 0
        if key_state["up"]: movement = 1
        elif key_state["down"]: movement = 2
        elif key_state["left"]: movement = 3
        elif key_state["right"]: movement = 4
        
        space = 1 if key_state["space"] else 0
        shift = 1 if key_state["shift"] else 0
        
        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30)
        
    env.close()
    pygame.quit()