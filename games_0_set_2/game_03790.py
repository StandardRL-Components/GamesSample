
# Generated: 2025-08-28T00:26:19.596815
# Source Brief: brief_03790.md
# Brief Index: 3790

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a time-management farming game.

    The player controls a cursor on a grid-based farm. The goal is to plant seeds,
    wait for them to grow, harvest the ripe crops, and sell them to accumulate
    1000 coins before a 60-second timer runs out. Each action consumes one step
    and one second from the timer.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Press Space to interact with a tile (plant, harvest, or sell)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Cultivate crops on your farm to earn 1000 coins before the timer expires. Plant, grow, harvest, and sell!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the farming game environment.
        """
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 12, 8
        self.TILE_SIZE = 40
        self.GRID_WIDTH = self.GRID_COLS * self.TILE_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.TILE_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # Game parameters
        self.INITIAL_TIMER = 60
        self.WIN_SCORE = 1000
        self.MAX_STEPS = 1000
        self.CROP_GROWTH_TIME = 5
        self.CROP_SALE_VALUE = 25

        # Tile States
        self.STATE_EMPTY = 0
        self.STATE_PLANTED = 1
        self.STATE_RIPE = 2
        self.STATE_HARVESTED = 3

        # Colors
        self.COLOR_BG = (34, 32, 52)
        self.COLOR_GRID = (50, 48, 72)
        self.COLOR_CURSOR = (255, 204, 0)
        self.COLOR_TEXT = (230, 230, 255)
        self.COLOR_UI_BG = (60, 58, 82, 180)
        self.COLOR_SOIL = (87, 65, 51)
        self.COLOR_SPROUT = (132, 222, 85)
        self.COLOR_RIPE = (255, 170, 51)
        self.COLOR_HARVESTED = (153, 102, 51)
        self.COLOR_GOLD = (255, 215, 0)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        
        # State variables (initialized in reset)
        self.grid = None
        self.growth_timers = None
        self.cursor_pos = None
        self.score = None
        self.timer = None
        self.steps = None
        self.game_over = None
        self.win_message = None
        self.last_space_held = None
        self.particles = None
        
        # Initialize state
        self.reset()

        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        
        # Initialize all game state
        self.grid = np.full((self.GRID_ROWS, self.GRID_COLS), self.STATE_EMPTY, dtype=np.uint8)
        self.growth_timers = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.int8)
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.score = 0
        self.timer = self.INITIAL_TIMER
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.last_space_held = False
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        """Processes an action and advances the game state by one step."""
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # 1. Update Cursor Position (with wraparound)
        if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_ROWS) % self.GRID_ROWS
        elif movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_ROWS
        elif movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_COLS) % self.GRID_COLS
        elif movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_COLS
        
        # 2. Handle Interaction (on space press, not hold)
        space_press = space_held and not self.last_space_held
        if space_press:
            reward += self._handle_interaction()
        self.last_space_held = space_held
        
        # 3. Update Game World State
        self._update_crops()
        self._update_particles()
        
        # 4. Update Timers and Step Count
        self.timer = max(0, self.timer - 1)
        self.steps += 1
        
        # 5. Check for Termination Conditions
        terminated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100.0
            self.game_over = True
            self.win_message = "YOU WIN!"
        elif self.timer <= 0:
            terminated = True
            reward -= 100.0
            self.game_over = True
            self.win_message = "TIME'S UP!"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_interaction(self):
        """Performs an action on the tile under the cursor."""
        col, row = self.cursor_pos
        tile_state = self.grid[row, col]
        reward = 0.0

        if tile_state == self.STATE_EMPTY:
            # Plant a seed
            self.grid[row, col] = self.STATE_PLANTED
            self.growth_timers[row, col] = self.CROP_GROWTH_TIME
            reward = 0.1
            self._create_particles(self.cursor_pos, self.COLOR_SPROUT, 10)
            # sfx: plant_seed.wav
        elif tile_state == self.STATE_RIPE:
            # Harvest a crop
            self.grid[row, col] = self.STATE_HARVESTED
            reward = 0.2
            self._create_particles(self.cursor_pos, self.COLOR_RIPE, 15)
            # sfx: harvest.wav
        elif tile_state == self.STATE_HARVESTED:
            # Sell crops
            self.grid[row, col] = self.STATE_EMPTY
            self.score += self.CROP_SALE_VALUE
            reward = 1.0
            self._create_sell_particles(self.cursor_pos)
            # sfx: coin_jingle.wav
        
        return reward

    def _update_crops(self):
        """Updates the growth state of all planted crops."""
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == self.STATE_PLANTED:
                    self.growth_timers[r, c] -= 1
                    if self.growth_timers[r, c] <= 0:
                        self.grid[r, c] = self.STATE_RIPE
                        # sfx: crop_ripe.wav

    def _get_observation(self):
        """Renders the current game state to a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the grid, crops, cursor, and particles."""
        # Draw grid tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.TILE_SIZE,
                    self.GRID_OFFSET_Y + r * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_SOIL, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                # Draw crop based on state
                state = self.grid[r, c]
                center_x, center_y = rect.center
                
                if state == self.STATE_PLANTED:
                    growth_progress = 1 - (self.growth_timers[r, c] / self.CROP_GROWTH_TIME)
                    radius = int(3 + 8 * growth_progress)
                    pygame.draw.circle(self.screen, self.COLOR_SPROUT, (center_x, center_y), radius)
                elif state == self.STATE_RIPE:
                    pygame.draw.circle(self.screen, self.COLOR_RIPE, (center_x, center_y), 15)
                    pygame.draw.circle(self.screen, (255, 255, 255), (center_x - 5, center_y - 5), 2) # a little glint
                elif state == self.STATE_HARVESTED:
                    pygame.draw.rect(self.screen, self.COLOR_HARVESTED, rect.inflate(-10, -10))

        # Draw particles
        self._draw_particles()

        # Draw cursor
        cursor_x = self.GRID_OFFSET_X + self.cursor_pos[0] * self.TILE_SIZE
        cursor_y = self.GRID_OFFSET_Y + self.cursor_pos[1] * self.TILE_SIZE
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.TILE_SIZE, self.TILE_SIZE)
        
        # Pulsing effect for cursor
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        line_width = 2 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width, border_radius=4)
        
    def _render_ui(self):
        """Renders the UI elements like score and timer."""
        # Score display
        score_text = self.font_medium.render(f"COINS: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topleft=(10, 10))
        ui_bg_score = score_rect.inflate(20, 10)
        s = pygame.Surface(ui_bg_score.size, pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, ui_bg_score.topleft)
        self.screen.blit(score_text, score_rect)

        # Timer display
        timer_text = self.font_medium.render(f"TIME: {self.timer}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        ui_bg_timer = timer_rect.inflate(20, 10)
        s = pygame.Surface(ui_bg_timer.size, pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, ui_bg_timer.topleft)
        self.screen.blit(timer_text, timer_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        """Returns a dictionary with game information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "cursor_pos": list(self.cursor_pos),
        }

    # --- Particle System ---
    def _create_particles(self, grid_pos, color, count):
        """Creates a burst of particles at a grid location."""
        px = self.GRID_OFFSET_X + grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifespan = random.randint(15, 30)
            self.particles.append({'x': px, 'y': py, 'vx': vx, 'vy': vy, 'life': lifespan, 'color': color})

    def _create_sell_particles(self, grid_pos):
        """Creates coin particles that fly to the score UI."""
        px = self.GRID_OFFSET_X + grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2
        for _ in range(5):
            lifespan = random.randint(40, 60)
            # The 'vx' and 'vy' will be recalculated each frame for homing
            self.particles.append({'x': px, 'y': py, 'vx': 0, 'vy': 0, 'life': lifespan, 'color': self.COLOR_GOLD, 'type': 'coin'})

    def _update_particles(self):
        """Updates position and lifespan of all active particles."""
        target_pos = (80, 25) # Approximate center of score UI
        for p in self.particles:
            if p.get('type') == 'coin':
                # Homing behavior for coin particles
                dx = target_pos[0] - p['x']
                dy = target_pos[1] - p['y']
                dist = math.hypot(dx, dy)
                if dist < 1: dist = 1
                p['vx'] = (dx / dist) * 4
                p['vy'] = (dy / dist) * 4
            
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity for non-coin particles
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _draw_particles(self):
        """Draws all active particles."""
        for p in self.particles:
            size = max(0, int(p['life'] / 5))
            if p.get('type') == 'coin':
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), 5, p['color'])
                pygame.gfxdraw.aacircle(self.screen, int(p['x']), int(p['y']), 5, p['color'])
            else:
                pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), size)

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a real window to display the game
    pygame.display.set_caption("Farming Game")
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        # Only step if the game is not over
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']}, Timer: {info['timer']}, Reward: {reward:.2f}")

        # Render the observation to the real screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds before closing
            running = False
            
        env.clock.tick(10) # Run at 10 FPS for manual play

    pygame.quit()