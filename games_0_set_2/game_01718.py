
# Generated: 2025-08-27T18:04:07.637431
# Source Brief: brief_01718.md
# Brief Index: 1718

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a fruit group. Shift to reshuffle the board (costs points)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match cascading fruits in a grid-based puzzle race against time. Chain combos to reach the target score across 3 stages."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    GRID_COLS = 10
    GRID_ROWS = 8
    
    FRUIT_TYPES = 5
    TIME_PER_STAGE = 60.0
    SCORE_TO_WIN = 5000
    STAGE_THRESHOLDS = [1667, 3334]
    
    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_BG = (40, 55, 71)
    COLOR_GRID_LINES = (60, 75, 91)
    COLOR_UI_TEXT = (236, 240, 241)
    COLOR_SCORE = (241, 196, 15)
    COLOR_PENALTY = (231, 76, 60)
    
    FRUIT_COLORS = [
        (231, 76, 60),   # 1: Red (Apple)
        (46, 204, 113),  # 2: Green (Lime)
        (52, 152, 219),  # 3: Blue (Blueberry)
        (241, 196, 15),  # 4: Yellow (Banana)
        (230, 126, 34)   # 5: Orange (Orange)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self._init_layout()
        self.reset()
        
        # This can be commented out for performance during training
        self.validate_implementation()

    def _init_layout(self):
        """Calculates rendering layout based on constants."""
        self.cell_size = 40
        self.grid_pixel_width = self.GRID_COLS * self.cell_size
        self.grid_pixel_height = self.GRID_ROWS * self.cell_size
        self.grid_x_offset = (self.SCREEN_WIDTH - self.grid_pixel_width) // 2
        self.grid_y_offset = (self.SCREEN_HEIGHT - self.grid_pixel_height) // 2 + 20

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.time_remaining = self.TIME_PER_STAGE
        self.stage = 1
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.particles = []
        self.floating_texts = []
        
        self.last_space_held = False
        self.last_shift_held = False
        self.move_cooldown = 0

        self._create_grid()
        
        return self._get_observation(), self._get_info()

    def _create_grid(self):
        self.grid = [[self._new_fruit() for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        
        while self._find_and_remove_initial_matches() or not self._has_possible_moves():
            self._fill_grid()

    def _new_fruit(self, y_offset=-20):
        fruit_type = self.np_random.integers(1, self.FRUIT_TYPES + 1)
        return {"type": fruit_type, "y_offset": y_offset, "alpha": 255}

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        step_reward = 0

        if not self.game_over:
            # --- Update Timers ---
            self.time_remaining -= 1.0 / self.FPS
            if self.move_cooldown > 0:
                self.move_cooldown -= 1

            # --- Handle Input ---
            self._handle_movement(movement)
            
            space_pressed = space_held and not self.last_space_held
            if space_pressed:
                step_reward += self._attempt_match()
            
            shift_pressed = shift_held and not self.last_shift_held
            if shift_pressed:
                step_reward += self._reshuffle_grid()
            
            self.last_space_held = space_held
            self.last_shift_held = shift_held

            # --- Game Logic ---
            self._update_fruits()
            cascade_made = self._apply_gravity()
            if cascade_made:
                self._fill_grid()
                # Check for chain reactions
                while self._find_and_remove_initial_matches(is_chain_reaction=True):
                    self._apply_gravity()
                    self._fill_grid()
                    # Small reward for chain reactions
                    step_reward += 2
                    self.score += 50
                    self._add_floating_text("Chain!", (320, 150), (255, 255, 0), 40)


            self._update_particles()
            self._update_floating_texts()
            self._check_stage_progression()

        # --- Termination Check ---
        if self.time_remaining <= 0 and not self.game_over:
            self.game_over = True
            self.win = False
            step_reward = -100.0
            self._add_floating_text("TIME'S UP", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 30), self.COLOR_PENALTY, 90, self.font_large)

        if self.score >= self.SCORE_TO_WIN and not self.game_over:
            self.game_over = True
            self.win = True
            step_reward = 100.0
            self._add_floating_text("YOU WIN!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 30), self.COLOR_SCORE, 90, self.font_large)

        terminated = self.game_over
        
        return self._get_observation(), step_reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        if self.move_cooldown == 0 and movement != 0:
            x, y = self.cursor_pos
            if movement == 1: y -= 1  # Up
            elif movement == 2: y += 1  # Down
            elif movement == 3: x -= 1  # Left
            elif movement == 4: x += 1  # Right
            
            self.cursor_pos[0] = np.clip(x, 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(y, 0, self.GRID_ROWS - 1)
            self.move_cooldown = 4  # 4-frame delay

    def _attempt_match(self):
        x, y = self.cursor_pos
        fruit = self.grid[y][x]
        if not fruit: return 0

        connected = self._find_connected_fruits(x, y)
        
        if len(connected) >= 3:
            # Successful match
            # SFX: Match success
            num_matched = len(connected)
            base_score = num_matched * 10
            bonus = 1.5 if num_matched >= 4 else 1.0
            match_score = int(base_score * bonus)
            self.score += match_score
            
            reward = num_matched * 1.0
            if num_matched >= 4:
                reward += 5.0

            for pos in connected:
                fx, fy = pos
                fruit_type = self.grid[fy][fx]["type"]
                self._spawn_particles(fx, fy, self.FRUIT_COLORS[fruit_type - 1])
                self.grid[fy][fx] = None
            
            self._add_floating_text(f"+{match_score}", (self.grid_x_offset + x * self.cell_size, self.grid_y_offset + y * self.cell_size), self.COLOR_SCORE, 30)
            return reward
        else:
            # Failed match
            # SFX: Match fail
            self.score = max(0, self.score - 20)
            self._add_floating_text("-20", (self.grid_x_offset + x * self.cell_size, self.grid_y_offset + y * self.cell_size), self.COLOR_PENALTY, 30)
            return -0.2

    def _find_connected_fruits(self, start_x, start_y):
        if not self.grid[start_y][start_x]:
            return []
            
        target_type = self.grid[start_y][start_x]["type"]
        q = [(start_x, start_y)]
        visited = set(q)
        connected = []

        while q:
            x, y = q.pop(0)
            connected.append((x, y))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS and (nx, ny) not in visited:
                    if self.grid[ny][nx] and self.grid[ny][nx]["type"] == target_type:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return connected

    def _apply_gravity(self):
        moved = False
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] is not None:
                    if r != empty_row:
                        # Animate fall
                        fall_dist = empty_row - r
                        self.grid[r][c]['y_offset'] = -fall_dist * self.cell_size
                        self.grid[empty_row][c] = self.grid[r][c]
                        self.grid[r][c] = None
                        moved = True
                    empty_row -= 1
        return moved
    
    def _fill_grid(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] is None:
                    self.grid[r][c] = self._new_fruit(y_offset=-self.cell_size)

    def _find_and_remove_initial_matches(self, is_chain_reaction=False):
        to_remove = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Check horizontal
                if c < self.GRID_COLS - 2:
                    if self.grid[r][c] and self.grid[r][c+1] and self.grid[r][c+2] and \
                       self.grid[r][c]["type"] == self.grid[r][c+1]["type"] == self.grid[r][c+2]["type"]:
                        to_remove.update([(c, r), (c+1, r), (c+2, r)])
                # Check vertical
                if r < self.GRID_ROWS - 2:
                    if self.grid[r][c] and self.grid[r+1][c] and self.grid[r+2][c] and \
                       self.grid[r][c]["type"] == self.grid[r+1][c]["type"] == self.grid[r+2][c]["type"]:
                        to_remove.update([(c, r), (c, r+1), (c, r+2)])
        
        if not to_remove:
            return False

        for x, y in to_remove:
            if is_chain_reaction:
                fruit_type = self.grid[y][x]['type']
                self._spawn_particles(x, y, self.FRUIT_COLORS[fruit_type-1])
            self.grid[y][x] = None
        return True

    def _has_possible_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] is None: continue
                
                # Create a temporary grid to simulate a match
                temp_grid = [[f.copy() if f else None for f in row] for row in self.grid]
                
                # Simulate a click on this fruit
                original_fruit = temp_grid[r][c]
                q = [(c, r)]
                visited = set(q)
                connected = []
                while q:
                    x, y = q.pop(0)
                    connected.append((x, y))
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS and (nx, ny) not in visited:
                            if temp_grid[ny][nx] and temp_grid[ny][nx]["type"] == original_fruit["type"]:
                                visited.add((nx, ny))
                                q.append((nx, ny))
                
                if len(connected) >= 3:
                    return True
        return False

    def _reshuffle_grid(self):
        # SFX: Reshuffle
        self.score = max(0, self.score - 50)
        
        flat_grid = [self.grid[r][c] for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS)]
        self.np_random.shuffle(flat_grid)
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                self.grid[r][c] = flat_grid[r * self.GRID_COLS + c]
        
        while self._find_and_remove_initial_matches() or not self._has_possible_moves():
            self._fill_grid()

        self._add_floating_text("SHUFFLE!", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.COLOR_UI_TEXT, 60)
        return -5.0

    def _update_fruits(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c]:
                    self.grid[r][c]['y_offset'] *= 0.65
                    if abs(self.grid[r][c]['y_offset']) < 1:
                        self.grid[r][c]['y_offset'] = 0

    def _check_stage_progression(self):
        if self.stage < 3 and self.score >= self.STAGE_THRESHOLDS[self.stage - 1]:
            self.stage += 1
            self.time_remaining = self.TIME_PER_STAGE
            # SFX: Stage complete
            self._add_floating_text(f"STAGE {self.stage}", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.COLOR_SCORE, 60, self.font_large)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage, "time": self.time_remaining}

    def _render_game(self):
        # Grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, 
                         (self.grid_x_offset, self.grid_y_offset, self.grid_pixel_width, self.grid_pixel_height), 
                         border_radius=5)
        
        # Fruits
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                fruit = self.grid[r][c]
                if fruit:
                    cx = self.grid_x_offset + c * self.cell_size + self.cell_size // 2
                    cy = self.grid_y_offset + r * self.cell_size + self.cell_size // 2 + int(fruit['y_offset'])
                    color = self.FRUIT_COLORS[fruit["type"] - 1]
                    radius = self.cell_size // 2 - 4
                    
                    pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, color)

        # Cursor
        cursor_pulse = (math.sin(self.steps * 0.3) + 1) / 2 * 100 + 50
        cursor_color = (255, 255, 255, cursor_pulse)
        cursor_rect = pygame.Rect(
            self.grid_x_offset + self.cursor_pos[0] * self.cell_size,
            self.grid_y_offset + self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        cursor_surface = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, cursor_color, (0, 0, self.cell_size, self.cell_size), 3, border_radius=5)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

        self._render_particles()
        self._render_floating_texts()

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Time
        time_int = max(0, int(self.time_remaining))
        time_color = self.COLOR_PENALTY if time_int < 10 else self.COLOR_UI_TEXT
        time_text = self.font_medium.render(f"TIME: {time_int}", True, time_color)
        time_rect = time_text.get_rect(centerx=self.SCREEN_WIDTH // 2, y=10)
        self.screen.blit(time_text, time_rect)
        
        # Stage
        stage_text = self.font_medium.render(f"STAGE: {self.stage}/3", True, self.COLOR_UI_TEXT)
        stage_rect = stage_text.get_rect(right=self.SCREEN_WIDTH - 20, y=10)
        self.screen.blit(stage_text, stage_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
    # --- Effects ---
    def _spawn_particles(self, grid_x, grid_y, color):
        cx = self.grid_x_offset + grid_x * self.cell_size + self.cell_size // 2
        cy = self.grid_y_offset + grid_y * self.cell_size + self.cell_size // 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({'x': cx, 'y': cy, 'vx': vx, 'vy': vy, 'lifetime': lifetime, 'max_life': lifetime, 'color': color, 'radius': radius})

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # gravity
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_life']))
            color = (*p['color'], alpha)
            pos = (int(p['x']), int(p['y']))
            
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (pos[0] - p['radius'], pos[1] - p['radius']))

    def _add_floating_text(self, text, pos, color, lifetime, font=None):
        if font is None:
            font = self.font_small
        self.floating_texts.append({'text': text, 'x': pos[0], 'y': pos[1], 'vy': -1, 'lifetime': lifetime, 'max_life': lifetime, 'color': color, 'font': font})

    def _update_floating_texts(self):
        for ft in self.floating_texts:
            ft['y'] += ft['vy']
            ft['lifetime'] -= 1
        self.floating_texts = [ft for ft in self.floating_texts if ft['lifetime'] > 0]

    def _render_floating_texts(self):
        for ft in self.floating_texts:
            alpha = int(255 * (ft['lifetime'] / ft['max_life']))
            color = (*ft['color'][:3], alpha)
            text_surf = ft['font'].render(ft['text'], True, color)
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(int(ft['x']), int(ft['y'])))
            self.screen.blit(text_surf, text_rect)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    import pygame
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Fruit Matcher")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print(env.user_guide)
    print(env.game_description)

    while not done:
        # --- Action mapping for human play ---
        movement = 0 # no-op
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()