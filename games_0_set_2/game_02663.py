import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode, suitable for server environments
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a tile, "
        "then move to an adjacent tile and press Space again to swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent colored tiles to create matches of 3 or more. "
        "Clear the board before the timer runs out to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    NUM_TILE_TYPES = 5
    MAX_STEPS = 600

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    TILE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECT = (255, 200, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TIMER_BAR = (80, 150, 255)
    COLOR_TIMER_BG = (40, 50, 70)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.particles = []
        self.prev_space_state = 0
        self.np_random = np.random.default_rng()

        # Calculate grid rendering properties
        self.tile_size = 36
        self.grid_width = self.GRID_SIZE * self.tile_size
        self.grid_height = self.GRID_SIZE * self.tile_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2 + 20
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.time_left = self.MAX_STEPS
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_pos = None
        self.particles = []
        self.prev_space_state = 0
        
        # Generate a valid starting board (no initial matches)
        while True:
            self.grid = self.np_random.integers(0, self.NUM_TILE_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_all_matches():
                break
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_button = action[1]
        # shift_button is unused in this game but must be handled by the action space
        
        space_press = space_button == 1 and self.prev_space_state == 0
        self.prev_space_state = space_button
        
        # Update game logic
        self.steps += 1
        self.time_left -= 1
        reward = -0.01  # Step penalty

        # --- Handle player input ---
        # Move cursor
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)

        # Handle selection/swap on Space press
        if space_press:
            if self.selected_pos is None:
                # Select a tile
                self.selected_pos = list(self.cursor_pos)
            else:
                # Attempt to swap with the selected tile
                dist = abs(self.cursor_pos[0] - self.selected_pos[0]) + abs(self.cursor_pos[1] - self.selected_pos[1])
                if dist == 1: # Is adjacent
                    p1 = tuple(self.selected_pos)
                    p2 = tuple(self.cursor_pos)
                    self.grid[p1[1], p1[0]], self.grid[p2[1], p2[0]] = self.grid[p2[1], p2[0]], self.grid[p1[1], p1[0]]

                    combo_multiplier = 1.0
                    total_chain_reward = 0
                    
                    initial_swap_had_match = False
                    while True:
                        matches = self._find_all_matches()
                        if not matches:
                            if not initial_swap_had_match: # No matches on initial swap
                                self.grid[p1[1], p1[0]], self.grid[p2[1], p2[0]] = self.grid[p2[1], p2[0]], self.grid[p1[1], p1[0]]
                            break
                        
                        initial_swap_had_match = True
                        num_cleared = len(matches)
                        match_reward = self._calculate_match_reward(num_cleared)
                        total_chain_reward += match_reward * combo_multiplier
                        
                        self._clear_tiles(matches)
                        self._apply_gravity()
                        self._refill_board()
                        
                        combo_multiplier += 0.5
                    
                    reward += total_chain_reward
                    self.score += total_chain_reward
                
                self.selected_pos = None # Deselect after any swap attempt

        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            # Victory condition is part of _check_termination, check if board is cleared
            if np.all(self.grid == -1):
                reward += 100
                self.score += 100
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _check_termination(self):
        # Using bool() to convert potential numpy.bool_ to a standard Python bool
        is_timed_out = self.time_left <= 0
        is_board_cleared = np.all(self.grid == -1)
        return bool(is_timed_out or is_board_cleared)

    def _find_all_matches(self):
        matched_tiles = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                tile_type = self.grid[r, c]
                if tile_type != -1 and tile_type == self.grid[r, c+1] and tile_type == self.grid[r, c+2]:
                    match_len = 3
                    while c + match_len < self.GRID_SIZE and self.grid[r, c + match_len] == tile_type:
                        match_len += 1
                    for i in range(match_len):
                        matched_tiles.add((c + i, r))
        
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                tile_type = self.grid[r, c]
                if tile_type != -1 and tile_type == self.grid[r+1, c] and tile_type == self.grid[r+2, c]:
                    match_len = 3
                    while r + match_len < self.GRID_SIZE and self.grid[r + match_len, c] == tile_type:
                        match_len += 1
                    for i in range(match_len):
                        matched_tiles.add((c, r + i))
        return matched_tiles

    def _calculate_match_reward(self, num_cleared):
        base_reward = num_cleared
        if num_cleared == 4:
            base_reward += 5
        elif num_cleared >= 5:
            base_reward += 10
        return base_reward

    def _clear_tiles(self, matches):
        for c, r in matches:
            if self.grid[r, c] != -1:
                tile_color = self.TILE_COLORS[self.grid[r, c]]
                for _ in range(5):
                    self.particles.append(Particle(
                        self.grid_offset_x + c * self.tile_size + self.tile_size / 2,
                        self.grid_offset_y + r * self.tile_size + self.tile_size / 2,
                        tile_color,
                        self.np_random
                    ))
                self.grid[r, c] = -1

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
    
    def _refill_board(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x + i * self.tile_size, self.grid_offset_y), (self.grid_offset_x + i * self.tile_size, self.grid_offset_y + self.grid_height), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, self.grid_offset_y + i * self.tile_size), (self.grid_offset_x + self.grid_width, self.grid_offset_y + i * self.tile_size), 1)
            
        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_type = self.grid[r, c]
                if tile_type != -1:
                    color = self.TILE_COLORS[tile_type]
                    rect = pygame.Rect(self.grid_offset_x + c * self.tile_size + 2, self.grid_offset_y + r * self.tile_size + 2, self.tile_size - 4, self.tile_size - 4)
                    pygame.draw.rect(self.screen, color, rect, border_radius=5)
        
        # Draw selection highlight
        if self.selected_pos is not None:
            c, r = self.selected_pos
            rect = pygame.Rect(self.grid_offset_x + c * self.tile_size, self.grid_offset_y + r * self.tile_size, self.tile_size, self.tile_size)
            pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, width=3, border_radius=7)

        # Draw cursor
        c, r = self.cursor_pos
        rect = pygame.Rect(self.grid_offset_x + c * self.tile_size, self.grid_offset_y + r * self.tile_size, self.tile_size, self.tile_size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, width=2, border_radius=7)

        # Update and draw particles
        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles:
            p.update()
            p.draw(self.screen)

    def _render_ui(self):
        # Draw score
        score_text = self.font_large.render(f"{int(self.score):,}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH // 2, 30))
        self.screen.blit(score_text, score_rect)
        
        # Draw timer bar
        timer_width = self.SCREEN_WIDTH - 40
        timer_height = 15
        timer_y = self.SCREEN_HEIGHT - 25
        
        bg_rect = pygame.Rect(20, timer_y, timer_width, timer_height)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BG, bg_rect, border_radius=7)
        
        time_ratio = max(0, self.time_left / self.MAX_STEPS)
        fg_width = int(timer_width * time_ratio)
        if fg_width > 0:
            fg_rect = pygame.Rect(20, timer_y, fg_width, timer_height)
            pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, fg_rect, border_radius=7)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }
        
    def close(self):
        pygame.quit()

class Particle:
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        self.color = color
        self.np_random = np_random
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = self.np_random.uniform(15, 30) # frames
        self.age = 0
        self.size = self.np_random.uniform(4, 8)

    def is_alive(self):
        return self.age < self.lifespan

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1
        self.age += 1
        self.vx *= 0.98

    def draw(self, screen):
        life_ratio = self.age / self.lifespan
        current_size = int(self.size * (1 - life_ratio))
        if current_size <= 0: return

        alpha = int(255 * (1 - life_ratio))
        
        temp_surface = pygame.Surface((current_size * 2, current_size * 2), pygame.SRCALPHA)
        pygame.draw.rect(temp_surface, (*self.color, alpha), (current_size/2, current_size/2, current_size, current_size))
        screen.blit(temp_surface, (int(self.x - current_size), int(self.y - current_size)))

if __name__ == '__main__':
    # The main loop is for human play and visualization, not for training
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Match-3 Puzzle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    print(env.game_description)
    print(env.user_guide)
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
        
        # Only step if there was an action
        if movement or space or shift:
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != -0.01:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}")
                obs, info = env.reset()
                pygame.time.wait(2000)
        else:
            # Re-render without stepping if no action
            obs = env._get_observation()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    env.close()