
# Generated: 2025-08-27T14:39:23.014208
# Source Brief: brief_00752.md
# Brief Index: 752

        
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

    user_guide = (
        "Controls: Arrow keys to move the selector. Press space to match the highlighted group of fruits."
    )

    game_description = (
        "Match cascading fruits in a grid to reach a target score before time runs out. Form groups of 3 or more."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 10, 8
    GRID_AREA_WIDTH, GRID_AREA_HEIGHT = 360, 360
    CELL_SIZE = GRID_AREA_HEIGHT // GRID_ROWS
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_COLS * CELL_SIZE) // 2 + 50
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_ROWS * CELL_SIZE) // 2

    WIN_SCORE = 1000
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # --- Colors ---
    COLOR_BG = (40, 40, 50)
    COLOR_GRID_BG = (30, 30, 40)
    COLOR_GRID_LINES = (60, 60, 70)
    COLOR_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_HIGHLIGHT = (255, 255, 255)
    FRUIT_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 160, 80),  # Orange
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
        self.font_large = pygame.font.Font(None, 56)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.timer = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.particles = []
        self.last_space_held = False
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.timer = self.MAX_STEPS
        self.game_over = False
        self.win = False
        self.last_space_held = False
        self.particles = []
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self._init_grid()

        return self._get_observation(), self._get_info()

    def _init_grid(self):
        self.grid = self.np_random.integers(0, len(self.FRUIT_COLORS), size=(self.GRID_ROWS, self.GRID_COLS))
        # Ensure no initial matches
        for _ in range(5): # Iterate a few times to clear most starting matches
            matches_found = self._find_and_remove_all_matches(add_score=False)
            if not matches_found:
                break
            self._apply_gravity_and_refill()


    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = False

        if not self.game_over:
            self.timer -= 1
            self.steps += 1

            movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
            
            # --- Handle Input ---
            self._move_cursor(movement)
            
            space_pressed = space_held and not self.last_space_held
            
            if space_pressed:
                reward += self._handle_match_attempt()
            elif movement == 0: # No-op penalty
                reward -= 0.1

            self.last_space_held = space_held

            # --- Update Game State ---
            self._apply_gravity_and_refill()
            self._update_particles()
            
            # --- Check Termination ---
            if self.score >= self.WIN_SCORE:
                self.game_over = True
                self.win = True
                terminated = True
                reward += 100
            elif self.timer <= 0 or self.steps >= self.MAX_STEPS:
                self.game_over = True
                self.win = False
                terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

    def _handle_match_attempt(self):
        cx, cy = self.cursor_pos
        if self.grid[cy][cx] == -1:
            return 0

        group = self._find_connected_group(cx, cy)
        
        if len(group) >= 3:
            # SFX: Match success
            num_matched = len(group)
            reward = num_matched
            if num_matched >= 6:
                reward += 10
            
            self.score += num_matched * (num_matched - 2) * 10

            for r, c in group:
                self._create_particles(c, r, self.FRUIT_COLORS[self.grid[r][c]])
                self.grid[r][c] = -1 # Mark for removal
            return reward
        else:
            # SFX: Match fail
            return 0

    def _find_connected_group(self, start_c, start_r):
        if not (0 <= start_r < self.GRID_ROWS and 0 <= start_c < self.GRID_COLS):
            return []
            
        target_fruit = self.grid[start_r][start_c]
        if target_fruit == -1:
            return []

        q = [(start_r, start_c)]
        visited = set(q)
        group = []

        while q:
            r, c = q.pop(0)
            group.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                    if (nr, nc) not in visited and self.grid[nr][nc] == target_fruit:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return group

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_COLS):
            empty_r = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r][c] != -1:
                    if r != empty_r:
                        self.grid[empty_r][c] = self.grid[r][c]
                        self.grid[r][c] = -1
                    empty_r -= 1
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] == -1:
                    self.grid[r][c] = self.np_random.integers(0, len(self.FRUIT_COLORS))
                    # SFX: Fruit spawn

    def _find_and_remove_all_matches(self, add_score=True):
        matches_found = False
        to_remove = set()
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if (r, c) not in to_remove:
                    group = self._find_connected_group(c, r)
                    if len(group) >= 3:
                        matches_found = True
                        for gr, gc in group:
                            to_remove.add((gr, gc))

        if matches_found and add_score:
            num_matched = len(to_remove)
            self.score += num_matched * (num_matched - 2) * 10
        
        for r, c in to_remove:
            self.grid[r][c] = -1
        
        return matches_found

    def _create_particles(self, c, r, color):
        center_x = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color, 'radius': radius})

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

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_COLS * self.CELL_SIZE, self.GRID_ROWS * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw fruits and highlights
        fruit_radius = int(self.CELL_SIZE * 0.4)
        highlight_group = self._find_connected_group(self.cursor_pos[0], self.cursor_pos[1])
        if len(highlight_group) < 3:
            highlight_group = []

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                fruit_type = self.grid[r][c]
                if fruit_type != -1:
                    center_x = self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
                    center_y = self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                    color = self.FRUIT_COLORS[fruit_type]

                    # Highlight for potential match
                    if (r, c) in highlight_group:
                        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(fruit_radius * 1.2), (*self.COLOR_HIGHLIGHT, 60))
                        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(fruit_radius * 1.2), (*self.COLOR_HIGHLIGHT, 120))
                    
                    # Draw fruit
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, fruit_radius, color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, fruit_radius, color)
                    
                    # Shine effect
                    shine_x = center_x + fruit_radius // 3
                    shine_y = center_y - fruit_radius // 3
                    pygame.gfxdraw.filled_circle(self.screen, shine_x, shine_y, fruit_radius // 4, (255, 255, 255, 120))

        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_COLS * self.CELL_SIZE, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_ROWS * self.CELL_SIZE))

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'] * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)


    def _render_ui(self):
        # --- Score Display ---
        score_surf = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 20))
        
        # --- Target Score ---
        target_surf = self.font_small.render(f"TARGET: {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(target_surf, (20, 55))

        # --- Timer Bar ---
        timer_bar_width = 200
        timer_bar_height = 20
        timer_x = self.SCREEN_WIDTH - timer_bar_width - 20
        timer_y = 20
        
        ratio = max(0, self.timer / self.MAX_STEPS)
        
        # Bar color changes from green to red
        bar_color = (
            int(255 * (1 - ratio)),
            int(200 * ratio),
            40
        )
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, (timer_x, timer_y, timer_bar_width, timer_bar_height))
        pygame.draw.rect(self.screen, bar_color, (timer_x, timer_y, int(timer_bar_width * ratio), timer_bar_height))
        
        timer_text_surf = self.font_small.render("TIME", True, self.COLOR_TEXT)
        self.screen.blit(timer_text_surf, (timer_x - 50, timer_y))


        # --- Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "TIME'S UP!"
            text_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Fruit Cascade")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Event Handling ---
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to Display ---
        # The observation is already the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()