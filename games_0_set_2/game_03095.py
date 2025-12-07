
# Generated: 2025-08-28T06:58:01.324520
# Source Brief: brief_03095.md
# Brief Index: 3095

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a retro arcade game.
    The player navigates a grid to collect coins while avoiding obstacles,
    all against a ticking timer.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Collect all yellow coins "
        "before the timer runs out. Avoid red obstacles!"
    )

    # Short, user-facing description of the game
    game_description = (
        "Navigate a grid-based world, collecting coins against the clock to "
        "achieve a high score. Getting too close to obstacles will cost you points."
    )

    # Frames only advance when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 20
        self.GRID_HEIGHT = 15
        self.CELL_SIZE = 20
        self.PLAY_AREA_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.PLAY_AREA_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.OFFSET_X = (self.SCREEN_WIDTH - self.PLAY_AREA_WIDTH) // 2
        self.OFFSET_Y = (self.SCREEN_HEIGHT - self.PLAY_AREA_HEIGHT) // 2
        
        self.MAX_STEPS = 600
        self.NUM_COINS = 50
        self.NUM_OBSTACLES = 15

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25) # Dark blue-black
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_OUTLINE = (200, 220, 255)
        self.COLOR_COIN = (255, 220, 0)
        self.COLOR_OBSTACLE = (200, 50, 50)
        self.COLOR_OBSTACLE_OUTLINE = (120, 30, 30)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 32)
        
        # --- Game State Variables (initialized in reset) ---
        self.player_pos = None
        self.coins = None
        self.obstacles = None
        self.timer = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win_condition = None
        self.particles = None
        
        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.timer = self.MAX_STEPS
        self.game_over = False
        self.win_condition = None
        self.particles = []

        # Generate game elements using the seeded RNG
        all_positions = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        
        # Player
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        all_positions.remove(self.player_pos)

        # Obstacles
        obstacle_indices = self.np_random.choice(len(all_positions), size=self.NUM_OBSTACLES, replace=False)
        self.obstacles = [list(all_positions)[i] for i in obstacle_indices]
        for pos in self.obstacles:
            all_positions.remove(tuple(pos))

        # Coins
        coin_indices = self.np_random.choice(len(all_positions), size=self.NUM_COINS, replace=False)
        self.coins = [list(all_positions)[i] for i in coin_indices]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self.steps += 1
        self.timer -= 1
        
        # Player movement
        px, py = self.player_pos
        if movement == 1:  # Up
            py -= 1
        elif movement == 2:  # Down
            py += 1
        elif movement == 3:  # Left
            px -= 1
        elif movement == 4:  # Right
            px += 1
        
        # Clamp player position to grid bounds
        px = max(0, min(self.GRID_WIDTH - 1, px))
        py = max(0, min(self.GRID_HEIGHT - 1, py))
        self.player_pos = (px, py)

        # --- Calculate Reward ---
        reward = 0

        # Coin collection
        if self.player_pos in self.coins:
            self.coins.remove(self.player_pos)
            reward += 1.0
            self.score += 1
            self._create_particles(self.player_pos, self.COLOR_COIN)
            # Placeholder: pygame.mixer.Sound('coin.wav').play()

        # Obstacle proximity penalty
        is_near_obstacle = False
        for ox, oy in self.obstacles:
            if abs(px - ox) + abs(py - oy) == 1:
                is_near_obstacle = True
                break
        if is_near_obstacle:
            reward -= 0.1

        # --- Check Termination ---
        terminated = False
        if len(self.coins) == 0:
            reward += 100.0
            terminated = True
            self.game_over = True
            self.win_condition = "VICTORY!"
            # Placeholder: pygame.mixer.Sound('win.wav').play()
        elif self.timer <= 0:
            reward -= 100.0
            terminated = True
            self.game_over = True
            self.win_condition = "TIME'S UP!"
            # Placeholder: pygame.mixer.Sound('lose.wav').play()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps_remaining": self.timer,
            "coins_remaining": len(self.coins),
        }

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.OFFSET_X + x * self.CELL_SIZE, self.OFFSET_Y)
            end_pos = (self.OFFSET_X + x * self.CELL_SIZE, self.OFFSET_Y + self.PLAY_AREA_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.OFFSET_X, self.OFFSET_Y + y * self.CELL_SIZE)
            end_pos = (self.OFFSET_X + self.PLAY_AREA_WIDTH, self.OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
            
        # Draw obstacles
        for ox, oy in self.obstacles:
            rect = pygame.Rect(
                self.OFFSET_X + ox * self.CELL_SIZE,
                self.OFFSET_Y + oy * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_OUTLINE, rect)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect.inflate(-4, -4))

        # Draw coins
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
        coin_radius = int(self.CELL_SIZE * 0.3 + pulse * 2)
        for cx, cy in self.coins:
            pos_x = self.OFFSET_X + cx * self.CELL_SIZE + self.CELL_SIZE // 2
            pos_y = self.OFFSET_Y + cy * self.CELL_SIZE + self.CELL_SIZE // 2
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, coin_radius, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, coin_radius, self.COLOR_COIN)

        # Update and draw particles
        self._update_and_draw_particles()

        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(
            self.OFFSET_X + px * self.CELL_SIZE,
            self.OFFSET_Y + py * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-4, -4), border_radius=3)

    def _render_ui(self):
        # Score display
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Timer display
        timer_text = self.font_medium.render(f"TIME: {self.timer}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game over message
        if self.game_over:
            color = self.COLOR_WIN if self.win_condition == "VICTORY!" else self.COLOR_LOSE
            end_text = self.font_large.render(self.win_condition, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            # Draw a semi-transparent background for the text
            s = pygame.Surface(end_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 128))
            self.screen.blit(s, end_rect.topleft)
            self.screen.blit(end_text, end_rect)
            
    def _grid_to_pixel(self, grid_pos):
        gx, gy = grid_pos
        px = self.OFFSET_X + gx * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.OFFSET_Y + gy * self.CELL_SIZE + self.CELL_SIZE // 2
        return px, py

    def _create_particles(self, grid_pos, color):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(10, 20)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': lifetime, 'color': color})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                radius = int((p['life'] / 20) * 4)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), max(0, radius), p['color'])
                active_particles.append(p)
        self.particles = active_particles

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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run pygame in a window
    import os
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Re-initialize pygame for display
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Collector")
    clock = pygame.time.Clock()

    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("Press ESC or close window to quit.")
    print("="*30 + "\n")

    action = np.array([0, 0, 0]) # No-op, no buttons

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                terminated = True

        # --- Map keyboard to MultiDiscrete action ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])

        # Step the environment only on a valid movement key press
        if movement != 0:
            obs, reward, term, trunc, info = env.step(action)
            terminated = term
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for manual play

    print("Game Over!")
    env.close()