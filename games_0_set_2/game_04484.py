
# Generated: 2025-08-28T02:32:33.305473
# Source Brief: brief_04484.md
# Brief Index: 4484

        
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
        "Controls: Use ↑↓←→ arrow keys to move your pixel around the grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-style arcade game. Navigate your pixel to collect all 50 coins "
        "before the 60-step timer runs out. Move efficiently to maximize your score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 32
        self.GRID_HEIGHT = 20
        self.CELL_SIZE = 20
        self.MAX_STEPS = 60
        self.GOAL_COINS = 50

        # Visuals
        self.COLOR_BG = (20, 20, 40)
        self.COLOR_GRID = (40, 40, 80)
        self.COLOR_PLAYER = (255, 0, 128)
        self.COLOR_PLAYER_GLOW = (255, 0, 128, 60)
        self.COLOR_COIN = (255, 220, 0)
        self.COLOR_COIN_SHINE = (255, 255, 180)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_TEXT_SHADOW = (10, 10, 20)

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.player_pos = None
        self.coins = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.last_dist_to_nearest_coin = None
        
        # Initialize state
        self.reset()

        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.coins = set()
        while len(self.coins) < self.GOAL_COINS:
            cx = self.np_random.integers(0, self.GRID_WIDTH)
            cy = self.np_random.integers(0, self.GRID_HEIGHT)
            if (cx, cy) != tuple(self.player_pos):
                self.coins.add((cx, cy))
        
        self.last_dist_to_nearest_coin = self._find_nearest_coin_dist(self.player_pos)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0.0
        
        # 1. Calculate pre-move state for reward shaping
        dist_before_move = self._find_nearest_coin_dist(self.player_pos)

        # 2. Update player position based on action
        new_pos = list(self.player_pos)
        if movement == 1:  # Up
            new_pos[1] -= 1
        elif movement == 2:  # Down
            new_pos[1] += 1
        elif movement == 3:  # Left
            new_pos[0] -= 1
        elif movement == 4:  # Right
            new_pos[0] += 1
        # movement == 0 is a no-op

        # 3. Clamp position to grid boundaries
        self.player_pos[0] = np.clip(new_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(new_pos[1], 0, self.GRID_HEIGHT - 1)
        
        # 4. Calculate movement reward
        dist_after_move = self._find_nearest_coin_dist(self.player_pos)
        if dist_after_move < dist_before_move:
            reward += 1.0  # +1 for moving towards the nearest coin
        elif dist_after_move > dist_before_move:
            reward -= 0.1  # -0.1 for moving away

        # 5. Check for coin collection
        player_pos_tuple = tuple(self.player_pos)
        if player_pos_tuple in self.coins:
            # SFX: Coin collect sound
            self.coins.remove(player_pos_tuple)
            self.score += 1
            reward += 10.0  # +10 for collecting a coin
            # Recalculate nearest coin dist after collection for next step's comparison
            dist_after_move = self._find_nearest_coin_dist(self.player_pos)

        self.last_dist_to_nearest_coin = dist_after_move
        
        # 6. Update step counter
        self.steps += 1
        
        # 7. Check for termination conditions
        terminated = False
        if self.score >= self.GOAL_COINS:
            terminated = True
            self.game_over = True
            reward += 100.0  # +100 for collecting all coins
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            reward -= 50.0  # -50 for timeout
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _find_nearest_coin_dist(self, pos):
        if not self.coins:
            return 0
        
        min_dist_sq = float('inf')
        for coin_pos in self.coins:
            dist_sq = (pos[0] - coin_pos[0])**2 + (pos[1] - coin_pos[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        return math.sqrt(min_dist_sq)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw coins with a shine effect
        for cx, cy in self.coins:
            px = int((cx + 0.5) * self.CELL_SIZE)
            py = int((cy + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.35)
            shine_radius = int(radius * 0.4)
            
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_COIN)
            pygame.gfxdraw.filled_circle(self.screen, px - shine_radius//2, py - shine_radius//2, shine_radius, self.COLOR_COIN_SHINE)

        # Draw player with a glow effect
        px = int((self.player_pos[0] + 0.5) * self.CELL_SIZE)
        py = int((self.player_pos[1] + 0.5) * self.CELL_SIZE)
        radius = int(self.CELL_SIZE * 0.4)
        
        # Draw glow
        glow_radius = int(radius * 1.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (px - glow_radius, py - glow_radius))

        # Draw player circle
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Helper to draw text with a shadow for readability
        def draw_text(text, font, color, pos):
            shadow_pos = (pos[0] + 2, pos[1] + 2)
            text_surface_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surface_shadow, shadow_pos)
            
            text_surface = font.render(text, True, color)
            self.screen.blit(text_surface, pos)

        # Render coin counter
        coin_text = f"COINS: {self.score}/{self.GOAL_COINS}"
        draw_text(coin_text, self.font, self.COLOR_TEXT, (15, 10))

        # Render timer
        time_left = self.MAX_STEPS - self.steps
        timer_text = f"TIME: {time_left}"
        text_width = self.font.size(timer_text)[0]
        draw_text(timer_text, self.font, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_width - 15, 10))
        
        # Render game over message
        if self.game_over:
            end_text = "VICTORY!" if self.score >= self.GOAL_COINS else "TIME UP!"
            text_width, text_height = self.font.size(end_text)
            pos = (self.SCREEN_WIDTH // 2 - text_width // 2, self.SCREEN_HEIGHT // 2 - text_height // 2)
            draw_text(end_text, self.font, self.COLOR_TEXT, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "coins_left": len(self.coins),
            "player_pos": self.player_pos,
        }

    def close(self):
        pygame.quit()

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
        assert self.score == 0
        assert len(self.coins) == self.GOAL_COINS
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert self.steps == 1
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: This is for human play and visualization, not for training.
    # The environment is designed for agent interaction via step() and reset().
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy', 'windows', or 'quartz' depending on your system

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    total_reward = 0

    print(env.user_guide)

    while running:
        action = [0, 0, 0] # Default action: no-op, no buttons pressed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                total_reward = 0
                print("\n--- Game Reset ---")

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Since auto_advance is False, we only step when a key is pressed or for a no-op
            # For a better human play experience, we'll use a clock to step at a regular interval.
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Score: {info['score']}")

            if terminated:
                print(f"--- Episode Finished ---")
                print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control human play speed
        pygame.time.Clock().tick(10) # 10 steps per second

    env.close()