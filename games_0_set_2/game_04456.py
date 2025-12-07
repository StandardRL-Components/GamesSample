
# Generated: 2025-08-28T02:28:33.742652
# Source Brief: brief_04456.md
# Brief Index: 4456

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the falling fruit. Press space to drop it instantly. Match 3 or more to score."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Match colorful falling fruits in a grid to score points before the stack reaches the top or time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Game parameters
    GRID_WIDTH = 8
    GRID_HEIGHT = 13
    CELL_SIZE = 30
    WIN_SCORE = 1000
    MAX_STEPS = 3600  # 60 seconds at 60 FPS

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_DANGER = (255, 0, 80)
    COLOR_TEXT = (240, 240, 255)
    COLOR_SCORE = (255, 220, 100)
    FRUIT_COLORS = {
        1: {"main": (255, 50, 50), "light": (255, 120, 120)},  # Apple (Red)
        2: {"main": (255, 220, 0), "light": (255, 240, 100)}, # Banana (Yellow)
        3: {"main": (50, 220, 50), "light": (120, 240, 120)},  # Lime (Green)
        4: {"main": (255, 140, 0), "light": (255, 180, 80)},   # Orange (Orange)
        5: {"main": (160, 60, 255), "light": (200, 140, 255)}, # Grape (Purple)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        
        # Grid positioning
        self.grid_pixel_width = self.GRID_WIDTH * self.CELL_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.CELL_SIZE
        self.grid_offset_x = (self.screen_width - self.grid_pixel_width) // 2
        self.grid_offset_y = (self.screen_height - self.grid_pixel_height)

        # Initialize state variables
        self.grid = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_fruit_type = 0
        self.next_fruit_type = 0
        self.fruit_col = 0
        self.fruit_row = 0.0
        self.fall_speed = 0.0
        self.particles = []
        self.pending_matches = []
        self.match_check_delay = 0
        self.gravity_delay = 0
        self.action_lock_timer = 0
        self.last_move_action = 0
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int8)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.pending_matches = []
        self.match_check_delay = 0
        self.gravity_delay = 0
        self.action_lock_timer = 0
        self.last_move_action = 0

        self.next_fruit_type = self.np_random.integers(1, len(self.FRUIT_COLORS) + 1)
        self._spawn_fruit()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(60)

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1
        
        reward = 0
        
        if not self.game_over:
            self.steps += 1
            
            self._update_timers()

            if self.action_lock_timer == 0:
                reward += self._handle_input(movement, space_pressed)
            
            if self.action_lock_timer == 0:
                reward += self._update_falling_fruit()
            
            if self.match_check_delay == 0 and self.pending_matches:
                reward += self._process_matches()
            
            if self.gravity_delay == 0 and not self.pending_matches:
                if self._apply_gravity():
                    self.match_check_delay = 5 
                    self.pending_matches = self._find_all_matches()
                    
            self._update_particles()
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 100
            else:
                reward -= 100
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _update_timers(self):
        self.action_lock_timer = max(0, self.action_lock_timer - 1)
        self.match_check_delay = max(0, self.match_check_delay - 1)
        self.gravity_delay = max(0, self.gravity_delay - 1)

    def _handle_input(self, movement, space_pressed):
        # Allow continuous movement by resetting last_move_action if key is released
        if movement == 0:
            self.last_move_action = 0

        if movement != self.last_move_action:
            if movement == 3: # Left
                self.fruit_col = max(0, self.fruit_col - 1)
            elif movement == 4: # Right
                self.fruit_col = min(self.GRID_WIDTH - 1, self.fruit_col + 1)
        
        if movement in [3,4]:
            self.last_move_action = movement

        if space_pressed:
            return self._hard_drop()
        return 0

    def _update_falling_fruit(self):
        # Difficulty scaling: Fall speed increases over time
        base_fall_speed = 0.03 # cells per frame
        speed_increase = (self.steps / self.MAX_STEPS) * 0.07
        self.fall_speed = base_fall_speed + speed_increase

        self.fruit_row += self.fall_speed
        
        target_row = math.floor(self.fruit_row)
        if (target_row >= self.GRID_HEIGHT - 1 or 
            self.grid[target_row + 1, self.fruit_col] != 0):
            return self._land_fruit()
        return 0

    def _land_fruit(self):
        landed_row = math.floor(self.fruit_row)
        self.grid[landed_row, self.fruit_col] = self.current_fruit_type
        # sfx: fruit_land

        reward = 1.0
        empty_spaces = self.GRID_HEIGHT - 1 - landed_row
        reward -= 0.1 * empty_spaces
        
        matches = self._find_matches_from(landed_row, self.fruit_col)
        if matches:
            self.pending_matches.append(matches)
            self.match_check_delay = 10
        
        self._spawn_fruit()
        return reward

    def _hard_drop(self):
        drop_row = self.GRID_HEIGHT - 1
        while drop_row >= 0 and self.grid[drop_row, self.fruit_col] != 0:
            drop_row -= 1
        
        if drop_row < 0:
            return 0

        self.fruit_row = drop_row
        return self._land_fruit()

    def _spawn_fruit(self):
        self.fruit_col = self.GRID_WIDTH // 2
        self.fruit_row = 0.0
        self.current_fruit_type = self.next_fruit_type
        self.next_fruit_type = self.np_random.integers(1, len(self.FRUIT_COLORS) + 1)
        
        if self.grid[0, self.fruit_col] != 0:
            self.game_over = True
            # sfx: game_over
    
    def _find_matches_from(self, row, col):
        if not (0 <= row < self.GRID_HEIGHT and 0 <= col < self.GRID_WIDTH):
            return []
        
        target_type = self.grid[row, col]
        if target_type == 0:
            return []

        q = [(row, col)]
        visited = set([(row, col)])
        matches = []

        while q:
            r, c = q.pop(0)
            matches.append((r, c))
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH and
                    (nr, nc) not in visited and self.grid[nr, nc] == target_type):
                    visited.add((nr, nc))
                    q.append((nr, nc))
        
        return matches if len(matches) >= 3 else []
        
    def _find_all_matches(self):
        all_matches = []
        visited = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0 and (r, c) not in visited:
                    matches = self._find_matches_from(r, c)
                    if matches:
                        all_matches.append(matches)
                        for pos in matches:
                            visited.add(pos)
        return all_matches

    def _process_matches(self):
        reward = 0
        total_cleared = 0
        
        for group in self.pending_matches:
            match_count = len(group)
            if match_count == 0: continue

            total_cleared += match_count
            
            if match_count == 3: reward += 10; self.score += 10
            elif match_count == 4: reward += 20; self.score += 25
            else: reward += 30; self.score += 50 * (match_count - 4)
            
            fruit_type = self.grid[group[0][0], group[0][1]]
            for r, c in group:
                self._create_particles(r, c, fruit_type)
                self.grid[r, c] = 0
                
        self.pending_matches = []
        if total_cleared > 0:
            # sfx: match_clear
            self.gravity_delay = 15
            self.action_lock_timer = self.gravity_delay
        return reward

    def _apply_gravity(self):
        moved = False
        for c in range(self.GRID_WIDTH):
            empty_slot = -1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] == 0 and empty_slot == -1:
                    empty_slot = r
                elif self.grid[r, c] != 0 and empty_slot != -1:
                    self.grid[empty_slot, c] = self.grid[r, c]
                    self.grid[r, c] = 0
                    moved = True
                    empty_slot -= 1
        return moved
        
    def _check_termination(self):
        return self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS or self.game_over

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid_bg()
        self._render_fruits()
        self._render_falling_fruit()
        self._render_particles()

    def _render_grid_bg(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.grid_offset_x + c * self.CELL_SIZE,
                    self.grid_offset_y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
        
        danger_y = self.grid_offset_y + self.CELL_SIZE
        pygame.draw.line(self.screen, self.COLOR_DANGER, 
                         (self.grid_offset_x, danger_y),
                         (self.grid_offset_x + self.grid_pixel_width, danger_y), 2)

    def _render_fruits(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                fruit_type = self.grid[r, c]
                if fruit_type != 0:
                    self._draw_fruit(c, r, fruit_type)

    def _render_falling_fruit(self):
        if not self.game_over and self.action_lock_timer == 0:
            self._draw_fruit(self.fruit_col, self.fruit_row, self.current_fruit_type, is_falling=True)
            
            drop_row = self.GRID_HEIGHT - 1
            while drop_row >= 0 and self.grid[drop_row, self.fruit_col] != 0:
                drop_row -= 1
            
            if drop_row >= self.fruit_row:
                x = self.grid_offset_x + self.fruit_col * self.CELL_SIZE
                y = self.grid_offset_y + drop_row * self.CELL_SIZE
                rect = pygame.Rect(x + 4, y + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
                pygame.draw.rect(self.screen, self.FRUIT_COLORS[self.current_fruit_type]['light'], rect, 2, border_radius=4)

    def _draw_fruit(self, col, row, fruit_type, is_falling=False):
        radius = self.CELL_SIZE // 2 - 3
        x = int(self.grid_offset_x + col * self.CELL_SIZE + self.CELL_SIZE / 2)
        y = int(self.grid_offset_y + row * self.CELL_SIZE + self.CELL_SIZE / 2)
        
        colors = self.FRUIT_COLORS[fruit_type]
        
        pygame.gfxdraw.filled_circle(self.screen, x + 2, y + 2, radius, (0, 0, 0, 50))
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, colors["main"])
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, colors["main"])

        highlight_x = x - radius // 3
        highlight_y = y - radius // 3
        pygame.gfxdraw.filled_circle(self.screen, highlight_x, highlight_y, radius // 2, colors["light"])

        if is_falling:
            pygame.gfxdraw.aacircle(self.screen, x, y, radius + 2, (255, 255, 255))

    def _create_particles(self, r, c, fruit_type):
        x = self.grid_offset_x + c * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.grid_offset_y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.FRUIT_COLORS[fruit_type]["main"]
        
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "lifetime": random.randint(20, 40), "radius": random.uniform(2, 5),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["x"] += p["vx"]; p["y"] += p["vy"]
            p["vy"] += 0.1; p["lifetime"] -= 1; p["radius"] -= 0.05
            if p["lifetime"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 40))
            color = (*p["color"], alpha)
            pos = (int(p["x"]), int(p["y"]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), color)
            
    def _render_ui(self):
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) / 60)
        time_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_DANGER
        time_text = self.font_large.render(f"{time_left:.1f}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.screen_width - 20, 10))
        self.screen.blit(time_text, time_rect)
        
        next_text = self.font_medium.render("NEXT", True, self.COLOR_TEXT)
        next_rect = next_text.get_rect(center=(self.grid_offset_x / 2, 50))
        self.screen.blit(next_text, next_rect)
        
        box_rect = pygame.Rect(0, 0, self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5)
        box_rect.center = (self.grid_offset_x / 2, 100)
        pygame.draw.rect(self.screen, self.COLOR_GRID, box_rect, 2, 5)
        self._draw_fruit_ui(box_rect.centerx, box_rect.centery, self.next_fruit_type)

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text_str = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            status_text = self.font_large.render(status_text_str, True, self.COLOR_SCORE)
            status_rect = status_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 - 20))
            self.screen.blit(status_text, status_rect)

    def _draw_fruit_ui(self, x, y, fruit_type):
        radius = self.CELL_SIZE // 2
        colors = self.FRUIT_COLORS[fruit_type]
        pygame.gfxdraw.aacircle(self.screen, int(x), int(y), radius, colors["main"])
        pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), radius, colors["main"])
        highlight_x = x - radius // 3
        highlight_y = y - radius // 3
        pygame.gfxdraw.filled_circle(self.screen, int(highlight_x), int(highlight_y), radius // 2, colors["light"])

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"Game terminated at step {_}. Final info: {info}")
            obs, info = env.reset()
    print("Example run completed without errors.")