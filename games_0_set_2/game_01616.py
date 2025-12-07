import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Fruit Popper: A grid-based puzzle game where the player pops groups of
    three or more adjacent fruits to score points against a timer.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to pop the selected fruit group."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pop groups of 3+ adjacent fruits in a race against time to reach the target score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 8
        self.CELL_SIZE = 50
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2

        self.MAX_STEPS = 600  # 60 seconds / 0.1s per step
        self.WIN_SCORE = 5000
        self.MIN_GROUP_SIZE = 3
        
        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_CURSOR_VALID = (255, 255, 255)
        self.COLOR_CURSOR_INVALID = (100, 100, 120)
        self.FRUIT_COLORS = [
            (255, 87, 87),   # Red
            (87, 187, 255),  # Blue
            (87, 255, 87),   # Green
            (255, 255, 87),  # Yellow
            (255, 87, 255),  # Magenta
            (255, 165, 0),   # Orange
        ]
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Arial", 72, bold=True)

        # --- Game State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        
        # Initialize state variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.particles = []

        # Create a valid starting grid with at least 3 poppable groups
        while True:
            self._initialize_grid()
            if self._count_valid_groups() >= 3:
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0

        self._move_cursor(movement)

        if space_pressed:
            reward = self._pop_selected_group()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100  # Win bonus
            else:
                reward -= 100  # Loss penalty

        self._update_particles()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _initialize_grid(self):
        num_colors = len(self.FRUIT_COLORS)
        self.grid = self.np_random.integers(1, num_colors + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))

    def _count_valid_groups(self):
        count = 0
        checked = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in checked:
                    group = self._find_group(x, y)
                    if len(group) >= self.MIN_GROUP_SIZE:
                        count += 1
                    checked.update(group)
        return count

    def _move_cursor(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

    def _pop_selected_group(self):
        group = self._find_group(self.cursor_pos[0], self.cursor_pos[1])

        if len(group) >= self.MIN_GROUP_SIZE:
            # Sound effect placeholder: pygame.mixer.Sound('pop.wav').play()
            fruit_type = self.grid[self.cursor_pos[0], self.cursor_pos[1]]
            for x, y in group:
                self._create_particles(x, y, fruit_type)
                self.grid[x, y] = 0  # Mark as empty

            self._apply_gravity()
            self._fill_top_rows()

            # Score increases with the square of the group size
            self.score += len(group) * len(group)
            
            # Reward is per-fruit
            return float(len(group))
        
        # Sound effect placeholder: pygame.mixer.Sound('invalid.wav').play()
        return 0.0

    def _find_group(self, start_x, start_y):
        if self.grid[start_x, start_y] == 0:
            return []

        target_color_idx = self.grid[start_x, start_y]
        q = [(start_x, start_y)]
        visited = set(q)
        group = []

        while q:
            x, y = q.pop(0)
            group.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[nx, ny] == target_color_idx:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_slots = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[x, y + empty_slots] = self.grid[x, y]
                    self.grid[x, y] = 0

    def _fill_top_rows(self):
        num_colors = len(self.FRUIT_COLORS)
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] == 0:
                    self.grid[x, y] = self.np_random.integers(1, num_colors + 1)

    def _create_particles(self, grid_x, grid_y, fruit_type):
        px = self.GRID_OFFSET_X + grid_x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
        color = self.FRUIT_COLORS[fruit_type - 1]
        for _ in range(12):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.5, 4.5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': [px, py], 'vel': vel, 'color': color, 'life': 25})

    def _update_particles(self):
        # This needs to be imported for the main block to run.
        # It's not strictly necessary for the headless environment.
        if 'pygame.gfxdraw' not in globals():
            try:
                import pygame.gfxdraw
            except ImportError:
                pass # Continue without gfxdraw if not available

        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity on particles
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        self._render_grid()
        self._render_fruits()
        self._render_cursor_and_selection()
        self._render_particles()

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, py))

    def _render_fruits(self):
        fruit_margin = 4
        fruit_size = self.CELL_SIZE - fruit_margin * 2
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                fruit_type = self.grid[x, y]
                if fruit_type > 0:
                    color = self.FRUIT_COLORS[fruit_type - 1]
                    rect = pygame.Rect(
                        self.GRID_OFFSET_X + x * self.CELL_SIZE + fruit_margin,
                        self.GRID_OFFSET_Y + y * self.CELL_SIZE + fruit_margin,
                        fruit_size, fruit_size
                    )
                    pygame.draw.rect(self.screen, color, rect, border_radius=10)

    def _render_cursor_and_selection(self):
        group = self._find_group(self.cursor_pos[0], self.cursor_pos[1])
        is_valid_group = len(group) >= self.MIN_GROUP_SIZE

        if is_valid_group:
            highlight_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            highlight_surf.fill((255, 255, 255, 60 + int(math.sin(self.steps * 0.5) * 20)))
            for x, y in group:
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                self.screen.blit(highlight_surf, rect.topleft)

        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        cursor_color = self.COLOR_CURSOR_VALID if is_valid_group else self.COLOR_CURSOR_INVALID
        pygame.draw.rect(self.screen, cursor_color, cursor_rect, width=4, border_radius=8)

    def _render_particles(self):
        # This needs to be imported for the main block to run.
        # It's not strictly necessary for the headless environment.
        try:
            import pygame.gfxdraw
            for p in self.particles:
                alpha = max(0, int(255 * (p['life'] / 25.0)))
                color = (*p['color'], alpha)
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, color)
        except ImportError:
            # Fallback if pygame.gfxdraw is not available
            for p in self.particles:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.draw.circle(self.screen, p['color'], pos, 3)


    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 10))

        time_ratio = max(0, 1 - (self.steps / self.MAX_STEPS))
        timer_bar_width, timer_bar_height = 200, 20
        bar_x = self.WIDTH - timer_bar_width - 15
        bar_y = 15

        pygame.draw.rect(self.screen, (50, 50, 70), (bar_x, bar_y, timer_bar_width, timer_bar_height), border_radius=5)
        current_width = int(timer_bar_width * time_ratio)
        bar_color = (0, 255, 0) if time_ratio > 0.5 else ((255, 255, 0) if time_ratio > 0.2 else (255, 0, 0))
        if current_width > 0:
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, current_width, timer_bar_height), border_radius=5)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME'S UP!"
            color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will try to use the dummy driver, but if a display is available,
    # it might open a window.
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Popper")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    while running:
        movement = 0  # No-op
        space_pressed = 0
        shift_pressed = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_pressed = 1
        
        action = [movement, space_pressed, shift_pressed]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Limit human play to 30 FPS

    env.close()