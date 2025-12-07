import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move the selector. Press space to match the selected fruit cluster."
    )

    game_description = (
        "Fast-paced puzzle game. Match clusters of fruit to score points and create cascades before time runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.GRID_X_OFFSET, self.GRID_Y_OFFSET = 180, 20
        self.CELL_SIZE = 36
        self.FPS = 30
        self.MAX_TIME = 60.0
        self.WIN_SCORE = 5000
        self.NUM_FRUIT_TYPES = 5
        self.MIN_MATCH_SIZE = 3

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_BG = (30, 45, 60)
        self.COLOR_GRID_LINE = (50, 65, 80)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.FRUIT_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        
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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.steps = None
        self.timer = None
        self.game_over = None
        self.win = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.particles = None
        self.falling_fruits = None

        # self.reset() is called here to ensure a valid initial state
        # The error was in the reset method itself, not the call from __init__
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # FIX: Initialize all game state variables at the beginning of reset.
        # This ensures that lists like `self.falling_fruits` are ready
        # before being used by helper methods like `_fill_top_rows`.
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.score = 0
        self.steps = 0
        self.timer = self.MAX_TIME
        self.game_over = False
        self.win = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.falling_fruits = []

        self.grid = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        
        # Ensure no initial matches by clearing them and refilling the grid.
        # This loop is now safe because self.falling_fruits is an initialized list.
        while self._find_all_matches():
            self._clear_matches(self._find_all_matches())
            self._apply_gravity()
            self._fill_top_rows()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        if self.game_over:
            if shift_pressed:
                # The environment will be reset on the next step call by the agent's logic,
                # but we call it here to handle the user input for manual play.
                # In an agent-driven scenario, the agent would call reset().
                obs, info = self.reset()
                return obs, 0, False, False, info
        else:
            self.timer = max(0, self.timer - 1.0 / self.FPS)

            # --- Handle Input ---
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

            # --- Process Match Attempt ---
            if space_pressed:
                cluster = self._find_cluster_bfs(self.cursor_pos[0], self.cursor_pos[1])
                
                if len(cluster) >= self.MIN_MATCH_SIZE:
                    # Successful match
                    match_size = len(cluster)
                    reward += match_size  # +1 per fruit
                    if match_size >= 7:
                        reward += 10  # Bonus for large clusters
                    
                    self.score += match_size * 10
                    self._create_particles_for_cluster(cluster)
                    self._clear_matches([cluster])
                    self._apply_gravity()
                    self._fill_top_rows()

                    # --- Chain Reactions ---
                    chain_level = 2
                    while True:
                        new_matches = self._find_all_matches()
                        if not new_matches:
                            break
                        
                        for new_cluster in new_matches:
                            chain_match_size = len(new_cluster)
                            chain_reward = chain_match_size * chain_level
                            reward += chain_reward
                            self.score += chain_match_size * 10 * chain_level
                            self._create_particles_for_cluster(new_cluster)
                        
                        self._clear_matches(new_matches)
                        self._apply_gravity()
                        self._fill_top_rows()
                        chain_level += 1
                else:
                    # Failed match attempt
                    reward = -0.1

        # --- Update Animations ---
        self._update_animations()

        # --- Check Termination Conditions ---
        if not self.game_over:
            if self.timer <= 0:
                self.game_over = True
                terminated = True
                reward = -100
            elif self.score >= self.WIN_SCORE:
                self.game_over = True
                self.win = True
                terminated = True
                reward += 100

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _find_cluster_bfs(self, start_x, start_y):
        if not (0 <= start_x < self.GRID_WIDTH and 0 <= start_y < self.GRID_HEIGHT):
            return []
        
        target_fruit = self.grid[start_x, start_y]
        if target_fruit == 0:
            return []

        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        cluster = []

        while q:
            x, y = q.popleft()
            cluster.append((x, y))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    if self.grid[nx, ny] == target_fruit:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return cluster

    def _find_all_matches(self):
        matches = []
        visited = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) not in visited and self.grid[x, y] != 0:
                    cluster = self._find_cluster_bfs(x, y)
                    if len(cluster) >= self.MIN_MATCH_SIZE:
                        matches.append(cluster)
                    for cx, cy in cluster:
                        visited.add((cx, cy))
        return matches

    def _clear_matches(self, match_list):
        for cluster in match_list:
            for x, y in cluster:
                self.grid[x, y] = 0

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_y = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] != 0:
                    if y != empty_y:
                        self.grid[x, empty_y] = self.grid[x, y]
                        self.grid[x, y] = 0
                    empty_y -= 1

    def _fill_top_rows(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == 0:
                    self.grid[x, y] = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1)
                    # Create a falling animation for this new fruit
                    start_screen_y = self.GRID_Y_OFFSET + (y - 1) * self.CELL_SIZE
                    end_screen_y = self.GRID_Y_OFFSET + y * self.CELL_SIZE
                    self.falling_fruits.append({
                        'pos': [self.GRID_X_OFFSET + x * self.CELL_SIZE, start_screen_y],
                        'end_y': end_screen_y,
                        'type': self.grid[x, y],
                        'vel': 0,
                    })

    def _create_particles_for_cluster(self, cluster):
        # Find the fruit type *before* clearing the grid
        if not cluster: return
        cx, cy = cluster[0]
        fruit_type = self.grid[cx, cy]
        if fruit_type == 0: return # Should not happen if called before _clear_matches

        for x, y in cluster:
            px = self.GRID_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2
            py = self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2
            for _ in range(10): # 10 particles per fruit
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 5)
                self.particles.append({
                    'pos': [px, py],
                    'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                    'radius': self.np_random.uniform(3, 6),
                    'color': self.FRUIT_COLORS[fruit_type - 1],
                    'life': self.np_random.integers(15, 30) # frames
                })

    def _update_animations(self):
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Update falling fruits (visual only, grid is already updated)
        for f in self.falling_fruits[:]:
            f['vel'] += 0.5 # Gravity
            f['pos'][1] += f['vel']
            if f['pos'][1] >= f['end_y']:
                self.falling_fruits.remove(f)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw fruits from grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                fruit_type = self.grid[x, y]
                if fruit_type != 0:
                    is_falling = any(
                        f['pos'][0] == self.GRID_X_OFFSET + x * self.CELL_SIZE and f['end_y'] == self.GRID_Y_OFFSET + y * self.CELL_SIZE
                        for f in self.falling_fruits
                    )
                    if not is_falling:
                        self._draw_fruit(self.screen, x, y, fruit_type)
        
        # Draw falling fruits (visual only)
        for fruit in self.falling_fruits:
            self._draw_fruit_pixel(self.screen, fruit['pos'][0], fruit['pos'][1], fruit['type'])

        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            start = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start, end)
        for i in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            end = (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start, end)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

        # Draw cursor
        if not self.game_over:
            cursor_x = self.GRID_X_OFFSET + self.cursor_pos[0] * self.CELL_SIZE
            cursor_y = self.GRID_Y_OFFSET + self.cursor_pos[1] * self.CELL_SIZE
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            alpha = 100 + pulse * 100
            
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((255, 255, 255, alpha))
            self.screen.blit(s, (cursor_x, cursor_y))
            pygame.draw.rect(self.screen, (255, 255, 255), (cursor_x, cursor_y, self.CELL_SIZE, self.CELL_SIZE), 2)

    def _draw_fruit(self, surface, grid_x, grid_y, fruit_type):
        pixel_x = self.GRID_X_OFFSET + grid_x * self.CELL_SIZE
        pixel_y = self.GRID_Y_OFFSET + grid_y * self.CELL_SIZE
        self._draw_fruit_pixel(surface, pixel_x, pixel_y, fruit_type)

    def _draw_fruit_pixel(self, surface, pixel_x, pixel_y, fruit_type):
        radius = self.CELL_SIZE // 2 - 4
        center_x = int(pixel_x + self.CELL_SIZE // 2)
        center_y = int(pixel_y + self.CELL_SIZE // 2)

        color = self.FRUIT_COLORS[fruit_type - 1]
        
        pygame.gfxdraw.filled_circle(surface, center_x, center_y, radius, color)
        pygame.gfxdraw.aacircle(surface, center_x, center_y, radius, color)
        
        # Simple highlight
        highlight_color = (min(255, color[0]+80), min(255, color[1]+80), min(255, color[2]+80))
        pygame.gfxdraw.filled_circle(surface, center_x - radius//3, center_y - radius//3, radius//3, highlight_color)
        pygame.gfxdraw.aacircle(surface, center_x - radius//3, center_y - radius//3, radius//3, highlight_color)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Timer
        timer_str = f"TIME: {math.ceil(self.timer):02d}"
        timer_color = (255, 100, 100) if self.timer < 10 else self.COLOR_UI_TEXT
        timer_text = self.font_medium.render(timer_str, True, timer_color)
        self.screen.blit(timer_text, (20, 60))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.win:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "TIME'S UP!"
                color = (255, 100, 100)
            
            main_text = self.font_large.render(msg, True, color)
            text_rect = main_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(main_text, text_rect)
            
            restart_text = self.font_small.render("Press SHIFT to restart", True, self.COLOR_UI_TEXT)
            restart_rect = restart_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30))
            self.screen.blit(restart_text, restart_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part is for human testing and visualization. It won't be used by the agent.
    
    obs, info = env.reset()
    done = False
    
    # Re-initialize pygame for display
    pygame.display.set_caption("Fruit Matcher")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Using get_pressed for continuous movement is better for this game style
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the observation from the environment to the display screen
        # Need to transpose it back to pygame's (width, height, channels) format
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # The env handles the restart logic via the shift key action,
            # but we need to keep the loop running to detect the key press.
            # A small delay prevents a tight loop after game over.
            pygame.time.wait(100)

        clock.tick(env.FPS)

    env.close()