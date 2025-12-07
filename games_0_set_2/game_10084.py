import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:48:37.045192
# Source Brief: brief_00084.md
# Brief Index: 84
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Navigate a shifting geometric gauntlet by matching tiles to unlock
    teleport gates and upgrade your abilities to reach the exit before time runs out.
    This environment prioritizes visual quality and satisfying gameplay mechanics.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a shifting geometric gauntlet by matching tiles to unlock teleport gates "
        "and upgrade your abilities to reach the exit before time runs out."
    )
    user_guide = (
        "Use arrow keys to swap adjacent tiles. Match 3 or more to clear them and activate gates. "
        "Use space to enter an active gate or confirm an upgrade. Use shift to cycle through upgrade choices."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 6, 10
    TILE_SIZE = 36
    TILE_MARGIN = 4
    GRID_WIDTH = GRID_COLS * TILE_SIZE
    GRID_HEIGHT = GRID_ROWS * TILE_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20
    FPS = 30
    MAX_EPISODE_STEPS = 1000
    LEVELS_TO_WIN = 5

    # --- COLORS ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID_BG = (30, 30, 45)
    TILE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 160, 80),  # Orange
    ]
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_GATE_INACTIVE = (100, 100, 120)
    COLOR_GATE_ACTIVE = (255, 255, 255)
    COLOR_EXIT_GATE = (255, 215, 0)
    COLOR_SELECTOR = (50, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 20)

        self.grid = []
        self.avatar_pos = []
        self.score = 0
        self.steps = 0
        self.time_left = 0
        self.max_time = 0
        self.game_over = False
        self.game_state = 'PLAY'
        self.level = 0
        self.gates = []
        self.upgrades = {}
        self.upgrade_choices = []
        self.selected_upgrade_idx = 0
        self.particles = []
        self.animations = []
        self.last_space_held = False
        self.last_shift_held = False
        self.step_reward = 0

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.max_time = 60.0
        self.time_left = self.max_time
        self.game_state = 'PLAY'
        self.upgrades = {}
        self.particles = []
        self.animations = []
        
        self.avatar_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.step_reward = 0

        self._update_time()
        self._handle_input(action)
        self._update_animations()
        self._update_particles()
        
        terminated = self._check_termination()
        
        self.score += self.step_reward
        
        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_time(self):
        if self.game_state == 'PLAY':
            self.time_left -= 1.0 / self.FPS
            # Penalize every 1/10th of a second
            if int(self.time_left * 10) % (self.FPS // 10) == 0:
                 self.step_reward -= 0.01

    def _check_termination(self):
        terminated = False
        if self.time_left <= 0:
            self.step_reward -= 100
            self.game_state = 'DEFEAT'
            terminated = True
        elif self.game_state in ['VICTORY', 'DEFEAT']:
             terminated = True
        elif self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
        self.game_over = terminated
        return terminated

    def _handle_input(self, action):
        if self.animations: return # Prevent input during animations

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        if self.game_state == 'PLAY':
            self._handle_play_input(movement, space_pressed)
        elif self.game_state == 'UPGRADE_CHOICE':
            self._handle_upgrade_input(shift_pressed, space_pressed)

    def _handle_play_input(self, movement, space_pressed):
        if movement != 0:
            dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][movement]
            r, c = self.avatar_pos
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                self._attempt_swap((r, c), (nr, nc))

        if space_pressed:
            for gate in self.gates:
                if tuple(self.avatar_pos) == gate['pos'] and gate['active']:
                    self._activate_gate(gate)
                    break
    
    def _handle_upgrade_input(self, shift_pressed, space_pressed):
        if shift_pressed:
            self.selected_upgrade_idx = (self.selected_upgrade_idx + 1) % len(self.upgrade_choices)
        if space_pressed:
            chosen_upgrade = self.upgrade_choices[self.selected_upgrade_idx]
            self.upgrades[chosen_upgrade] = self.upgrades.get(chosen_upgrade, 0) + 1
            self.step_reward += 10
            
            self.level += 1
            self.max_time = max(30, 60 - (self.level - 1) * 5)
            self.time_left = self.max_time
            self._generate_level()
            self.game_state = 'PLAY'
            # sfx: upgrade_confirm

    def _activate_gate(self, gate):
        if gate['is_exit']:
            self.step_reward += 100
            self.game_state = 'VICTORY'
            # sfx: victory
        else:
            self.step_reward += 10 # Reward for using the gate
            self.avatar_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
            self.game_state = 'UPGRADE_CHOICE'
            self._generate_upgrade_choices()
            # sfx: teleport

    def _generate_upgrade_choices(self):
        all_upgrades = ["Time Bonus", "Score Multiplier", "Tile Scramble"]
        self.upgrade_choices = self.np_random.choice(all_upgrades, size=2, replace=False).tolist()
        self.selected_upgrade_idx = 0

    def _attempt_swap(self, pos1, pos2):
        r1, c1 = pos1; r2, c2 = pos2
        self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]
        self._add_animation('swap', duration=8, data={'pos1': pos1, 'pos2': pos2})

        if not self._find_and_process_matches():
            self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]
            self._add_animation('swap', duration=8, data={'pos1': pos1, 'pos2': pos2}, delay=8)
            # sfx: swap_fail
        else:
            # sfx: match_success
            while True:
                self._refill_grid()
                if not self._find_and_process_matches():
                    break
            if not self._find_possible_moves():
                self._scramble_board()

    def _find_and_process_matches(self):
        matches = self._find_all_matches()
        if matches:
            self.step_reward += len(matches)
            
            if not any(g['active'] for g in self.gates):
                for gate in self.gates:
                    gate['active'] = True
                    self.step_reward += 5
                    # sfx: gate_activate

            for r, c in matches:
                self._create_particles(r, c, self.grid[r][c])
                self.grid[r][c] = -1
            self._add_animation('destroy', duration=10, data={'tiles': list(matches)})
        return matches

    def _refill_grid(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] != -1:
                    if r != empty_row:
                        self.grid[empty_row][c] = self.grid[r][c]
                        self.grid[r][c] = -1
                    empty_row -= 1
            for r in range(empty_row, -1, -1):
                self.grid[r][c] = self.np_random.integers(0, len(self.TILE_COLORS))
    
    def _find_all_matches(self):
        matches = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[r][c] != -1 and self.grid[r][c] == self.grid[r][c+1] == self.grid[r][c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for r in range(self.GRID_ROWS - 2):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] != -1 and self.grid[r][c] == self.grid[r+1][c] == self.grid[r+2][c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _find_possible_moves(self):
        temp_grid = np.copy(self.grid)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Swap right
                if c < self.GRID_COLS - 1:
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                    if self._find_all_matches():
                        self.grid = temp_grid
                        return True
                    self.grid[r][c], self.grid[r][c+1] = self.grid[r][c+1], self.grid[r][c]
                # Swap down
                if r < self.GRID_ROWS - 1:
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
                    if self._find_all_matches():
                        self.grid = temp_grid
                        return True
                    self.grid[r][c], self.grid[r+1][c] = self.grid[r+1][c], self.grid[r][c]
        self.grid = temp_grid
        return False
    
    def _scramble_board(self):
        flat_grid = self.grid.flatten().tolist()
        self.np_random.shuffle(flat_grid)
        self.grid = np.array(flat_grid).reshape((self.GRID_ROWS, self.GRID_COLS))
        while self._find_all_matches() or not self._find_possible_moves():
            self.np_random.shuffle(flat_grid)
            self.grid = np.array(flat_grid).reshape((self.GRID_ROWS, self.GRID_COLS))
        # sfx: scramble

    def _generate_level(self):
        self._create_grid()
        is_exit_level = (self.level >= self.LEVELS_TO_WIN)
        num_gates = 1 if is_exit_level else self.np_random.integers(2, 4)
        
        possible_locs = [(r, c) for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS)]
        gate_indices = self.np_random.choice(len(possible_locs), num_gates, replace=False)
        
        self.gates = []
        for i in gate_indices:
            self.gates.append({
                'pos': possible_locs[i], 'active': False, 'is_exit': is_exit_level
            })
        if is_exit_level: self.gates[0]['is_exit'] = True

    def _create_grid(self):
        self.grid = self.np_random.integers(0, len(self.TILE_COLORS), size=(self.GRID_ROWS, self.GRID_COLS))
        while self._find_all_matches() or not self._find_possible_moves():
            self.grid = self.np_random.integers(0, len(self.TILE_COLORS), size=(self.GRID_ROWS, self.GRID_COLS))

    def _add_animation(self, type, duration, data, delay=0):
        self.animations.append({'type': type, 'duration': duration, 'timer': duration + delay, 'data': data, 'delay': delay})

    def _update_animations(self):
        self.animations = [anim for anim in self.animations if anim['timer'] > 0]
        for anim in self.animations:
            anim['timer'] -= 1

    def _create_particles(self, r, c, color_idx):
        if color_idx < 0: return
        px, py = self._grid_to_pixel(r, c)
        px += self.TILE_SIZE // 2
        py += self.TILE_SIZE // 2
        color = self.TILE_COLORS[color_idx]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixel(self, r, c):
        return self.GRID_X + c * self.TILE_SIZE, self.GRID_Y + r * self.TILE_SIZE

    def _render_text(self, text, x, y, font, color, align="left"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect()
        if align == "center": text_rect.center = (x, y)
        elif align == "right": text_rect.right = x; text_rect.top = y
        else: text_rect.left = x; text_rect.top = y
        self.screen.blit(shadow_surf, (text_rect.x + 1, text_rect.y + 1))
        self.screen.blit(text_surf, text_rect)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT), border_radius=5)
        self._render_gates()
        self._render_tiles()
        self._render_avatar()
        self._render_particles()

    def _render_gates(self):
        for gate in self.gates:
            r, c = gate['pos']
            px, py = self._grid_to_pixel(r, c)
            rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
            if gate['active']:
                color = self.COLOR_EXIT_GATE if gate['is_exit'] else self.COLOR_GATE_ACTIVE
                alpha = 150 + 105 * math.sin(self.steps * 0.2)
                pygame.gfxdraw.box(self.screen, rect, (*color, int(alpha)))
            else:
                pygame.draw.rect(self.screen, self.COLOR_GATE_INACTIVE, rect, 2, border_radius=5)
    
    def _render_tiles(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_idx = self.grid[r][c]
                if color_idx == -1: continue

                px, py = self._grid_to_pixel(r, c)
                size = self.TILE_SIZE - self.TILE_MARGIN
                offset = self.TILE_MARGIN // 2
                rect = pygame.Rect(px + offset, py + offset, size, size)
                color = self.TILE_COLORS[color_idx]

                for anim in self.animations:
                    if anim['timer'] <= anim['delay']: continue
                    progress = (anim['duration'] - (anim['timer'] - anim['delay'])) / anim['duration']
                    if anim['type'] == 'swap':
                        pos1, pos2 = anim['data']['pos1'], anim['data']['pos2']
                        if (r, c) == pos1:
                            px2, py2 = self._grid_to_pixel(pos2[0], pos2[1])
                            rect.x = int(px + (px2 - px) * progress) + offset
                            rect.y = int(py + (py2 - py) * progress) + offset
                        elif (r, c) == pos2:
                            px1, py1 = self._grid_to_pixel(pos1[0], pos1[1])
                            rect.x = int(px + (px1 - px) * progress) + offset
                            rect.y = int(py + (py1 - py) * progress) + offset
                    elif anim['type'] == 'destroy' and (r,c) in anim['data']['tiles']:
                        scale = 1.0 - progress
                        rect.width = int(size * scale)
                        rect.height = int(size * scale)
                        rect.center = (px + self.TILE_SIZE // 2, py + self.TILE_SIZE // 2)
                
                pygame.draw.rect(self.screen, color, rect, border_radius=4)
                
    def _render_avatar(self):
        if self.game_state == 'PLAY':
            r, c = self.avatar_pos
            px, py = self._grid_to_pixel(r, c)
            rect = pygame.Rect(px, py, self.TILE_SIZE, self.TILE_SIZE)
            thickness = int(2 + 1.5 * math.sin(self.steps * 0.25))
            pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, thickness, border_radius=5)
    
    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            size = int(5 * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, (*p['color'], int(alpha)))

    def _render_ui(self):
        self._render_text(f"SCORE: {self.score}", 10, 10, self.font_ui, self.COLOR_TEXT)
        self._render_text(f"LEVEL: {self.level}/{self.LEVELS_TO_WIN}", self.SCREEN_WIDTH // 2, 10, self.font_ui, self.COLOR_TEXT, "center")
        
        time_str = f"TIME: {max(0, math.ceil(self.time_left))}"
        time_color = (255, 100, 100) if self.time_left < 10 else self.COLOR_TEXT
        self._render_text(time_str, self.SCREEN_WIDTH - 10, 10, self.font_ui, time_color, "right")

        if self.game_state == 'UPGRADE_CHOICE':
            self._render_upgrade_screen()
        elif self.game_state == 'VICTORY':
            self._render_text("VICTORY!", self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2, self.font_title, self.COLOR_EXIT_GATE, "center")
        elif self.game_state == 'DEFEAT':
            self._render_text("TIME UP", self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2, self.font_title, (200,50,50), "center")

    def _render_upgrade_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        self._render_text("CHOOSE UPGRADE", self.SCREEN_WIDTH // 2, 100, self.font_title, self.COLOR_TEXT, "center")
        
        for i, choice in enumerate(self.upgrade_choices):
            y_pos = 200 + i * 50
            color = self.COLOR_SELECTOR if i == self.selected_upgrade_idx else self.COLOR_TEXT
            self._render_text(choice, self.SCREEN_WIDTH // 2, y_pos, self.font_ui, color, "center")
        
        self._render_text("SHIFT to cycle, SPACE to select", self.SCREEN_WIDTH // 2, 350, self.font_small, self.COLOR_TEXT, "center")

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level}
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This validation part is removed from the final code but was useful for development
    # try:
    #     env_to_validate = GameEnv()
    #     obs, info = env_to_validate.reset()
    #     assert obs.shape == (GameEnv.SCREEN_HEIGHT, GameEnv.SCREEN_WIDTH, 3)
    #     assert obs.dtype == np.uint8
    #     test_action = env_to_validate.action_space.sample()
    #     obs, reward, term, trunc, info = env_to_validate.step(test_action)
    #     assert obs.shape == (GameEnv.SCREEN_HEIGHT, GameEnv.SCREEN_WIDTH, 3)
    #     print("✓ Implementation validated successfully")
    # except Exception as e:
    #     print(f"Validation failed: {e}")

    # The main execution block for playing the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # We need to create a display for the main loop to work
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc. depending on your OS
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Geometric Gauntlet")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # To prevent rapid-fire actions from single key presses
    action_timer = 0
    ACTION_COOLDOWN = 5 # frames
    
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if action_timer <= 0:
            keys = pygame.key.get_pressed()
            moved = False
            # This mapping corresponds to the action space: 0=No-op, 1=Up, 2=Down, 3=Left, 4=Right
            if keys[pygame.K_UP]: action[0] = 1; moved = True
            elif keys[pygame.K_DOWN]: action[0] = 2; moved = True
            elif keys[pygame.K_LEFT]: action[0] = 3; moved = True
            elif keys[pygame.K_RIGHT]: action[0] = 4; moved = True
            
            if keys[pygame.K_SPACE]: action[1] = 1; moved = True
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1; moved = True
            
            if moved:
                action_timer = ACTION_COOLDOWN
        else:
            action_timer -= 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}. Final info: {info}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(1000) # Pause before restarting
            
        clock.tick(GameEnv.FPS)
        
    env.close()