import os
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to clear a selected group of blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match 3 or more adjacent blocks of the same color to clear them. Clear the entire grid before time runs out!"
    )

    # Frames auto-advance for time-based gameplay and smooth graphics.
    auto_advance = True

    # --- Game Constants ---
    GRID_WIDTH = 12
    GRID_HEIGHT = 10
    BLOCK_SIZE = 36
    GRID_LINE_WIDTH = 2
    
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # This calculation correctly centers the game grid within the screen space
    _GAME_AREA_WIDTH = GRID_WIDTH * (BLOCK_SIZE + GRID_LINE_WIDTH) - GRID_LINE_WIDTH
    GRID_OFFSET_X = (SCREEN_WIDTH - _GAME_AREA_WIDTH) // 2
    GRID_OFFSET_Y = 60

    FPS = 30
    TIME_LIMIT_SECONDS = 60
    
    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (40, 50, 60)
    COLOR_GRID_LINE = (60, 70, 80)
    
    COLORS = {
        1: (255, 80, 80),   # Red
        2: (80, 255, 80),   # Green
        3: (80, 150, 255),  # Blue
        4: (255, 255, 80),  # Yellow
        5: (200, 80, 255),  # Purple
    }
    
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (230, 240, 250)
    COLOR_TIMER_BAR_GOOD = (80, 255, 80)
    COLOR_TIMER_BAR_WARN = (255, 255, 80)
    COLOR_TIMER_BAR_BAD = (255, 80, 80)

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

        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_huge = pygame.font.SysFont("Consolas", 60, bold=True)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int8)
        self.cursor_pos = [0, 0]
        self.score = 0
        self.steps = 0
        self.time_left = 0
        self.game_over = False
        self.last_space_held = False
        self.particles = []
        self.floating_texts = []
        
        # Note: self.reset() is not called in __init__ as per Gymnasium standard practice.
        # The user is expected to call it to get the first observation.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.last_space_held = False
        self.particles = []
        self.floating_texts = []

        self._generate_grid()
        self._ensure_valid_moves()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Input ---
        self._move_cursor(movement)
        
        if space_held and not self.last_space_held:
            cleared_count, cleared_color = self._clear_blocks()
            if cleared_count > 0:
                reward += cleared_count
                self.score += cleared_count * cleared_count # Bonus for larger clusters
                self._apply_gravity()
                self._ensure_valid_moves()
                
                # Add floating text for score
                pos_x, pos_y = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
                self.floating_texts.append({
                    "text": f"+{cleared_count*cleared_count}", "pos": [pos_x, pos_y], 
                    "life": self.FPS, "color": self.COLORS[cleared_color]
                })

        self.last_space_held = space_held
        
        # --- Update Game State ---
        self.steps += 1
        self.time_left = max(0, self.time_left - 1)
        self._update_particles()
        self._update_floating_texts()

        # --- Check Termination ---
        terminated = False
        is_grid_clear = np.all(self.grid == 0)

        if is_grid_clear:
            reward += 100 # Win bonus
            self.game_over = True
            terminated = True
        elif self.time_left <= 0:
            reward -= 100 # Lose penalty
            self.game_over = True
            terminated = True
        
        # Per test requirements, truncated must be False.
        # A time/step limit is a form of termination in this game.
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": self.time_left / self.FPS}

    # --- Game Logic Helpers ---

    def _generate_grid(self):
        self.grid = self.np_random.integers(1, len(self.COLORS) + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int8)

    def _ensure_valid_moves(self):
        while not self._has_valid_moves():
            if np.all(self.grid == 0): break # Grid is clear, no need to shuffle
            
            # Reshuffle remaining blocks
            remaining_blocks = self.grid[self.grid > 0].flatten().tolist()
            self.np_random.shuffle(remaining_blocks)
            
            self.grid.fill(0)
            
            idx = 0
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT - 1, -1, -1):
                    if idx < len(remaining_blocks):
                        self.grid[y][x] = remaining_blocks[idx]
                        idx += 1
                    else:
                        break
            
            self._apply_gravity() # Settle the shuffled blocks

    def _has_valid_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != 0:
                    cluster = self._find_cluster(x, y)
                    if len(cluster) >= 3:
                        return True
        return False

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

    def _find_cluster(self, start_x, start_y):
        if self.grid[start_y, start_x] == 0:
            return set()
        
        target_color = self.grid[start_y, start_x]
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        
        while q:
            x, y = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[ny, nx] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return visited

    def _clear_blocks(self):
        x, y = self.cursor_pos
        cluster = self._find_cluster(x, y)
        
        if len(cluster) < 3:
            return 0, 0
            
        cleared_color = self.grid[y, x]
        for bx, by in cluster:
            self.grid[by, bx] = 0
            self._create_particles(bx, by, cleared_color)
            
        return len(cluster), cleared_color

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            write_y = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y, x] != 0:
                    self.grid[write_y, x], self.grid[y, x] = self.grid[y, x], self.grid[write_y, x]
                    write_y -= 1

    # --- Particle and Effects ---
    
    def _create_particles(self, grid_x, grid_y, color_id):
        px, py = self._grid_to_pixel(grid_x, grid_y)
        px += self.BLOCK_SIZE // 2
        py += self.BLOCK_SIZE // 2
        color = self.COLORS[color_id]
        
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(self.FPS // 2, self.FPS)
            size = self.np_random.uniform(3, 7)
            self.particles.append({"pos": [px, py], "vel": vel, "life": life, "max_life": life, "color": color, "size": size})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _update_floating_texts(self):
        for ft in self.floating_texts:
            ft['pos'][1] -= 1
            ft['life'] -= 1
        self.floating_texts = [ft for ft in self.floating_texts if ft['life'] > 0]

    # --- Rendering Helpers ---

    def _grid_to_pixel(self, x, y):
        px = self.GRID_OFFSET_X + x * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
        py = self.GRID_OFFSET_Y + y * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
        return px, py
    
    def _render_game(self):
        # Draw grid background
        grid_height_pixels = self.GRID_HEIGHT * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self._GAME_AREA_WIDTH, grid_height_pixels)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=5)
        
        # Highlight selected cluster
        selected_cluster = self._find_cluster(self.cursor_pos[0], self.cursor_pos[1])
        if len(selected_cluster) >= 3:
            for x, y in selected_cluster:
                px, py = self._grid_to_pixel(x, y)
                highlight_rect = pygame.Rect(px - 3, py - 3, self.BLOCK_SIZE + 6, self.BLOCK_SIZE + 6)
                # Use a surface for transparency
                s = pygame.Surface((self.BLOCK_SIZE + 6, self.BLOCK_SIZE + 6), pygame.SRCALPHA)
                pygame.draw.rect(s, (255,255,255,100), s.get_rect(), border_radius=10)
                self.screen.blit(s, (px - 3, py - 3))


        # Draw blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_id = self.grid[y, x]
                if color_id != 0:
                    px, py = self._grid_to_pixel(x, y)
                    color = self.COLORS[color_id]
                    
                    rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    darker_color = tuple(max(0, c - 40) for c in color)
                    pygame.draw.rect(self.screen, darker_color, rect.move(0, 2), border_radius=8)
                    pygame.draw.rect(self.screen, color, rect, border_radius=8)
        
        # Draw grid lines on top
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_OFFSET_X + i * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH / 2
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + grid_rect.height), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_OFFSET_Y + i * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH / 2
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + grid_rect.width, y), self.GRID_LINE_WIDTH)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw floating texts
        for ft in self.floating_texts:
            alpha = int(255 * (ft['life'] / self.FPS))
            text_surf = self.font_small.render(ft['text'], True, ft['color'])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (ft['pos'][0], ft['pos'][1]))

        # Draw cursor
        cx, cy = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        cursor_rect = pygame.Rect(cx - 3, cy - 3, self.BLOCK_SIZE + 6, self.BLOCK_SIZE + 6)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=3, border_radius=10)
        
    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 15, 10))
        
        # Timer display
        time_ratio = self.time_left / (self.TIME_LIMIT_SECONDS * self.FPS) if self.time_left > 0 else 0
        
        # Timer bar
        bar_width = 250
        bar_height = 25
        bar_x = 15
        bar_y = 15
        
        if time_ratio > 0.5: bar_color = self.COLOR_TIMER_BAR_GOOD
        elif time_ratio > 0.2: bar_color = self.COLOR_TIMER_BAR_WARN
        else: bar_color = self.COLOR_TIMER_BAR_BAD
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, bar_width * time_ratio, bar_height), border_radius=5)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            is_win = np.all(self.grid == 0)
            message = "YOU WIN!" if is_win else "TIME UP!"
            color = self.COLOR_TIMER_BAR_GOOD if is_win else self.COLOR_TIMER_BAR_BAD
            
            end_text = self.font_huge.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()


# Example usage to run and display the game
if __name__ == "__main__":
    # To run with display, comment out the os.environ line at the top of the file
    # and instantiate with render_mode="human"
    try:
        is_headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
    except (KeyError, NameError):
        is_headless = False

    if is_headless:
        print("Running in headless mode. No visual output will be shown.")
        print("To play the game, comment out the 'os.environ' line at the top of the file.")
        # Run a short automated test in headless mode
        env = GameEnv()
        obs, info = env.reset(seed=42)
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished. Final score: {info['score']}")
                obs, info = env.reset(seed=42)
        env.close()

    else:
        env = GameEnv(render_mode="rgb_array")
        human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption(env.game_description)
        clock = pygame.time.Clock()
        
        obs, info = env.reset()
        done = False
        
        # Game loop
        while not done:
            # --- Human Controls ---
            movement = 0 # no-op
            space_held = 0
            shift_held = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            if keys[pygame.K_r]: # Allow manual reset
                obs, info = env.reset()

            action = [movement, space_held, shift_held]
            
            # --- Gym Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
                # Render one last time to show game over screen
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                human_screen.blit(surf, (0, 0))
                pygame.display.flip()
                # Wait a moment before auto-resetting
                pygame.time.wait(2000)
                obs, info = env.reset()

            # --- Render to human screen ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(env.FPS)
            
        env.close()