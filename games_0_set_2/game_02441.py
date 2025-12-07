import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to rake the sand. Hold Shift and use arrow keys to move the stone cursor. Press Space to place a stone."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Create a visually appealing zen garden by raking sand and placing stones to achieve inner peace. Reach an aesthetic score of 80 before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GARDEN_RECT = pygame.Rect(40, 40, 560, 320)

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
        
        # Colors
        self.COLOR_BG = (20, 20, 20)
        self.COLOR_SAND = (210, 180, 140)
        self.COLOR_RAKE_LIGHT = (225, 195, 155)
        self.COLOR_RAKE_DARK = (195, 165, 125)
        self.COLOR_STONE = (80, 85, 90)
        self.COLOR_STONE_SHADOW = (60, 65, 70)
        self.COLOR_WALL = (60, 40, 20)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_SHADOW = (0, 0, 0)
        self.COLOR_CURSOR = (100, 150, 255, 150)
        self.COLOR_CURSOR_INVALID = (255, 100, 100, 150)

        # Fonts
        self.font_ui = pygame.font.Font(None, 32)
        
        # Game parameters
        self.FPS = 30
        self.MAX_SECONDS = 300
        self.MAX_STEPS = self.MAX_SECONDS * self.FPS
        self.WIN_SCORE = 80
        self.RAKE_WIDTH = 40
        self.RAKE_INTENSITY_FADE = 1
        self.STONE_MIN_PROXIMITY = 40
        self.STONE_RADIUS = 12
        self.CURSOR_SPEED = 5
        self.SAND_GRID_SCALE = 10 # Each grid cell is 10x10 pixels
        self.grid_width = self.GARDEN_RECT.width // self.SAND_GRID_SCALE
        self.grid_height = self.GARDEN_RECT.height // self.SAND_GRID_SCALE
        
        # Initialize state variables (will be set in reset)
        self.steps = 0
        self.score = 0
        self.last_score = 0
        self.game_over = False
        self.stones = []
        self.cursor_pos = [0,0]
        self.sand_grid = None
        self.rng = None
        self.space_was_held = False
        self.stone_placements_this_step = 0
        
        self.reset()
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.last_score = 0
        self.game_over = False
        self.stones = []
        self.cursor_pos = [self.GARDEN_RECT.centerx, self.GARDEN_RECT.centery]
        self.space_was_held = True # Prevent placing stone on first frame
        
        # Sand grid stores [rake_direction, intensity]
        self.sand_grid = np.zeros((self.grid_height, self.grid_width, 2), dtype=int)
        
        self._calculate_aesthetic_score() # Initial score calculation

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.game_over = self.steps >= self.MAX_STEPS
        
        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            self.stone_placements_this_step = 0

            # 1. Handle Actions
            if shift_held:
                self._move_cursor(movement)
            else:
                self._rake_sand(movement)
            
            if space_held and not self.space_was_held:
                if self._place_stone():
                    self.stone_placements_this_step += 1
            
            self.space_was_held = space_held

            # 2. Update Game State
            self._update_sand_grid()
            self._calculate_aesthetic_score()

            # 3. Calculate Reward
            reward = self._calculate_reward()

        # 4. Check Termination
        terminated = self.game_over or self.score >= self.WIN_SCORE
        if terminated and not self.game_over: # Win condition
            reward += 100
        elif self.game_over and self.score < self.WIN_SCORE: # Time out
            reward -= 10
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _move_cursor(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1 # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1 # Right
        
        self.cursor_pos[0] += dx * self.CURSOR_SPEED
        self.cursor_pos[1] += dy * self.CURSOR_SPEED
        
        # Clamp cursor to garden bounds
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], self.GARDEN_RECT.left + self.STONE_RADIUS, self.GARDEN_RECT.right - self.STONE_RADIUS)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], self.GARDEN_RECT.top + self.STONE_RADIUS, self.GARDEN_RECT.bottom - self.STONE_RADIUS)

    def _rake_sand(self, movement):
        if movement == 0: return # No-op
        
        rake_len = int(max(self.grid_width, self.grid_height) * 0.75)
        center_gx, center_gy = self.grid_width // 2, self.grid_height // 2
        
        for i in range(rake_len):
            gx, gy = center_gx, center_gy
            if movement == 1: gy -= i # Up
            elif movement == 2: gy += i # Down
            elif movement == 3: gx -= i # Left
            elif movement == 4: gx += i # Right

            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                # Apply rake pattern over a width
                rake_w = self.RAKE_WIDTH // self.SAND_GRID_SCALE
                for w in range(-rake_w // 2, rake_w // 2):
                    if movement in [1, 2]: # Vertical rake
                        wgx = gx + w
                        if 0 <= wgx < self.grid_width:
                            self.sand_grid[gy, wgx] = [movement, 255]
                    elif movement in [3, 4]: # Horizontal rake
                        wgy = gy + w
                        if 0 <= wgy < self.grid_height:
                            self.sand_grid[wgy, gx] = [movement, 255]

    def _place_stone(self):
        # Check proximity to other stones
        for stone_pos in self.stones:
            dist = math.hypot(self.cursor_pos[0] - stone_pos[0], self.cursor_pos[1] - stone_pos[1])
            if dist < self.STONE_MIN_PROXIMITY:
                return False
        
        self.stones.append(tuple(self.cursor_pos))
        return True

    def _update_sand_grid(self):
        self.sand_grid[:, :, 1] = np.maximum(0, self.sand_grid[:, :, 1] - self.RAKE_INTENSITY_FADE)

    def _calculate_aesthetic_score(self):
        # --- Rake Score Component ---
        raked_cells = np.count_nonzero(self.sand_grid[:, :, 1])
        max_raked_cells = self.grid_width * self.grid_height
        rake_score = (raked_cells / max_raked_cells) * 20

        # --- Stone Score Component ---
        stone_score = 0
        n_stones = len(self.stones)
        if n_stones > 0:
            # Bonus for auspicious odd numbers
            if n_stones in [3, 5, 7]:
                stone_score += 25
            
            # Score based on placement (balance, spacing)
            stone_positions = np.array(self.stones)
            center_of_mass = np.mean(stone_positions, axis=0)
            garden_center = np.array(self.GARDEN_RECT.center)
            
            # Balance: reward for center of mass being near garden center
            balance_dist = np.linalg.norm(center_of_mass - garden_center)
            max_balance_dist = np.linalg.norm(np.array(self.GARDEN_RECT.topleft) - garden_center)
            balance_score = (1 - (balance_dist / max_balance_dist)) * 20
            stone_score += balance_score

            # Spacing: penalize being too close, reward good separation
            spacing_score = 0
            if n_stones > 1:
                for i in range(n_stones):
                    for j in range(i + 1, n_stones):
                        dist = np.linalg.norm(stone_positions[i] - stone_positions[j])
                        # Reward for being far apart, up to a point
                        spacing_score += min(dist, 200) / 100
                spacing_score /= (n_stones * (n_stones - 1) / 2) # Normalize
            stone_score += spacing_score * 10
            
            # Rule of Thirds: Reward for placing stones near 1/3 lines
            thirds_score = 0
            thirds_x = [self.GARDEN_RECT.left + self.GARDEN_RECT.width / 3, self.GARDEN_RECT.left + 2 * self.GARDEN_RECT.width / 3]
            thirds_y = [self.GARDEN_RECT.top + self.GARDEN_RECT.height / 3, self.GARDEN_RECT.top + 2 * self.GARDEN_RECT.height / 3]
            for pos in self.stones:
                is_on_third = any(abs(pos[0] - tx) < 20 for tx in thirds_x) or \
                              any(abs(pos[1] - ty) < 20 for ty in thirds_y)
                if is_on_third:
                    thirds_score += 5
            stone_score += thirds_score

        total_score = rake_score + stone_score
        self.score = int(np.clip(total_score, 0, 100))

    def _calculate_reward(self):
        reward = 0
        score_delta = self.score - self.last_score
        
        # Continuous reward for score changes
        if score_delta > 0:
            reward += score_delta * 0.1
        elif score_delta < 0:
            reward -= abs(score_delta) * 0.01

        # Event-based reward for good stone placement
        if self.stone_placements_this_step > 0 and score_delta > 2:
            reward += 5
            
        self.last_score = self.score
        return reward
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw garden boundary
        pygame.draw.rect(self.screen, self.COLOR_WALL, self.GARDEN_RECT.inflate(10, 10))
        # Draw sand base
        pygame.draw.rect(self.screen, self.COLOR_SAND, self.GARDEN_RECT)

        # Draw sand patterns
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                direction, intensity = self.sand_grid[gy, gx]
                if intensity > 0:
                    alpha = int(intensity / 255 * 100)
                    world_x = self.GARDEN_RECT.left + gx * self.SAND_GRID_SCALE
                    world_y = self.GARDEN_RECT.top + gy * self.SAND_GRID_SCALE
                    cell_rect = pygame.Rect(world_x, world_y, self.SAND_GRID_SCALE, self.SAND_GRID_SCALE)
                    
                    # Draw subtle lines to represent raking
                    for i in range(0, self.SAND_GRID_SCALE, 3):
                        if direction in [1, 2]: # Vertical rake -> horizontal lines
                            start_pos = (cell_rect.left, cell_rect.top + i)
                            end_pos = (cell_rect.right, cell_rect.top + i)
                        else: # Horizontal rake -> vertical lines
                            start_pos = (cell_rect.left + i, cell_rect.top)
                            end_pos = (cell_rect.left + i, cell_rect.bottom)
                        
                        color = self.COLOR_RAKE_LIGHT if i % 2 == 0 else self.COLOR_RAKE_DARK
                        # The original code's use of screen.convert_alpha() fails in headless mode.
                        # pygame.gfxdraw.line supports RGBA colors and correctly blends them onto
                        # the target surface without requiring a display.
                        pygame.gfxdraw.line(self.screen, start_pos[0], start_pos[1], end_pos[0], end_pos[1], (*color, alpha))
        
        # Draw stones
        for pos in sorted(self.stones, key=lambda p: p[1]): # Draw from top to bottom for correct overlap
            # Shadow
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]) + 2, int(pos[1]) + 2, self.STONE_RADIUS, self.COLOR_STONE_SHADOW)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]) + 2, int(pos[1]) + 2, self.STONE_RADIUS, self.COLOR_STONE_SHADOW)
            # Stone
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.STONE_RADIUS, self.COLOR_STONE)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.STONE_RADIUS, self.COLOR_STONE)
            # Highlight
            highlight_pos = (int(pos[0] - 3), int(pos[1] - 3))
            highlight_color = (120, 125, 130)
            pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], self.STONE_RADIUS // 2, highlight_color)

        # Draw cursor
        cursor_x, cursor_y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        is_valid = all(math.hypot(cursor_x - s[0], cursor_y - s[1]) >= self.STONE_MIN_PROXIMITY for s in self.stones)
        cursor_color = self.COLOR_CURSOR if is_valid else self.COLOR_CURSOR_INVALID

        cursor_surf = pygame.Surface((self.STONE_RADIUS * 2, self.STONE_RADIUS * 2), pygame.SRCALPHA)
        pygame.gfxdraw.aacircle(cursor_surf, self.STONE_RADIUS, self.STONE_RADIUS, self.STONE_RADIUS -1, cursor_color)
        self.screen.blit(cursor_surf, (cursor_x - self.STONE_RADIUS, cursor_y - self.STONE_RADIUS))

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_UI_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(content, pos)
            
        # Score
        score_text = f"Aesthetic Score: {self.score}"
        draw_text(score_text, self.font_ui, self.COLOR_UI_TEXT, (10, 10))

        # Timer
        time_left = max(0, self.MAX_SECONDS - (self.steps / self.FPS))
        minutes = int(time_left // 60)
        seconds = int(time_left % 60)
        timer_text = f"Time: {minutes:02d}:{seconds:02d}"
        text_width = self.font_ui.size(timer_text)[0]
        draw_text(timer_text, self.font_ui, self.COLOR_UI_TEXT, (self.SCREEN_WIDTH - text_width - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stones_placed": len(self.stones),
            "time_remaining_seconds": (self.MAX_STEPS - self.steps) / self.FPS,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    import time
    
    # Set to 'human' for a playable window
    render_mode = "rgb_array" 
    
    env = GameEnv(render_mode="rgb_array")

    if render_mode == "human":
        # This block is for human play and requires a display.
        # It's not used by the headless verification tests.
        os.environ.pop("SDL_VIDEODRIVER", None)
        pygame.display.init()
        pygame.display.set_caption("Zen Garden")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    obs, info = env.reset()
    done = False
    
    print(env.user_guide)
    
    total_reward = 0
    start_time = time.time()
    
    for _ in range(1000):
        action = [0, 0, 0] # no-op
        
        if render_mode == "human":
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = keys[pygame.K_SPACE]
            shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            
            action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        else: # Simple programmatic agent
            if env.steps < 500: # Spend time placing stones
                action[2] = 1 # Hold shift
                action[0] = env.action_space.nvec[0] # Random movement
                if env.rng.random() < 0.05:
                    action[1] = 1 # Press space
            else: # Spend time raking
                action[2] = 0 # Release shift
                if env.rng.random() < 0.1:
                    action[0] = env.rng.integers(1, 5) # Random rake direction

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if render_mode == "human":
            # Convert observation back to a surface and draw it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(env.FPS)
        
        if done:
            break
            
    end_time = time.time()
    print(f"Episode finished in {end_time - start_time:.2f} seconds.")
    print(f"Final Info: {info}")
    print(f"Total Reward: {total_reward}")
    env.close()