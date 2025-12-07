
# Generated: 2025-08-27T18:05:52.252882
# Source Brief: brief_01732.md
# Brief Index: 1732

        
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

    # Must be a short, user-facing control string:
    user_guide = "Controls: ↑ to jump, ←→ to move."

    # Must be a short, user-facing description of the game:
    game_description = (
        "A procedurally generated platform jumper. Navigate treacherous gaps and collect coins to reach the end."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.LEVEL_END_X = 5000
        self.MAX_STEPS = 1500

        # Player Physics
        self.PLAYER_SPEED = 5
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -14
        self.PLAYER_SIZE = (20, 20)
        
        # Platform Generation
        self.INITIAL_GAP = 50
        self.MIN_PLATFORM_WIDTH = 50
        self.MAX_PLATFORM_WIDTH = 200
        self.MAX_Y_CHANGE = 80

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (40, 80, 120)
        self.COLOR_PLAYER = (60, 170, 255)
        self.COLOR_PLAYER_OUTLINE = (200, 250, 255)
        self.COLOR_PLATFORM = (128, 128, 128)
        self.COLOR_PLATFORM_OUTLINE = (80, 80, 80)
        self.COLOR_COIN = (255, 215, 0)
        self.COLOR_COIN_OUTLINE = (255, 255, 255)
        self.COLOR_TEXT = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # --- Game State Variables ---
        self.player_rect = None
        self.player_vel_y = None
        self.on_ground = None
        self.platforms = None
        self.coins = None
        self.camera_offset_x = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.np_random = None
        
        # Initialize state variables
        self.reset()
        
        # This is a self-check to ensure the implementation matches the spec.
        # It's good practice for complex environments.
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize RNG
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback to a default or unseeded RNG if seed is None
            self.np_random = np.random.default_rng()

        # Initialize player state
        self.player_rect = pygame.Rect(100, self.HEIGHT // 2, *self.PLAYER_SIZE)
        self.player_vel_y = 0
        self.on_ground = False
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_offset_x = 0
        
        # Generate level
        self._generate_initial_platforms()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        player_vel_x = 0
        if movement == 3:  # Left
            player_vel_x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            player_vel_x = self.PLAYER_SPEED

        if movement == 1 and self.on_ground:  # Jump
            self.player_vel_y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump_sound()

        # --- Physics and State Update ---
        self._update_player(player_vel_x)
        self._update_camera()
        self._update_world() # Dynamically generate new platforms

        # --- Interactions and Rewards ---
        reward = 0.1  # Survival reward
        
        # Coin collection
        collected_coins = []
        for coin in self.coins:
            if self.player_rect.colliderect(coin):
                collected_coins.append(coin)
                self.score += 1
                reward += 1.0
                # sfx: coin_pickup_sound()
        self.coins = [c for c in self.coins if c not in collected_coins]
        
        self.steps += 1
        
        # --- Termination Check ---
        terminated = False
        if self.player_rect.top > self.HEIGHT:
            terminated = True
            reward = -100.0  # Fell off
            # sfx: fall_sound()
        elif self.player_rect.left >= self.LEVEL_END_X:
            terminated = True
            reward = 100.0  # Reached the end
            # sfx: level_complete_sound()
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Time limit

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_player(self, vel_x):
        # Horizontal Movement
        self.player_rect.x += vel_x
        # Prevent moving off the left edge of the world
        self.player_rect.left = max(0, self.player_rect.left)

        # Vertical Movement (Gravity)
        self.player_vel_y += self.GRAVITY
        self.player_rect.y += self.player_vel_y
        
        # Collision Detection
        self.on_ground = False
        for plat in self.platforms:
            # Check if player is falling and was previously above the platform
            if self.player_vel_y > 0 and plat.colliderect(self.player_rect):
                # Check for vertical overlap (player's feet are near platform top)
                if self.player_rect.bottom < plat.top + self.player_vel_y + 1:
                     # Check for horizontal overlap
                    if (self.player_rect.right > plat.left and self.player_rect.left < plat.right):
                        self.player_rect.bottom = plat.top
                        self.player_vel_y = 0
                        self.on_ground = True
                        break

    def _update_camera(self):
        # Camera follows player, keeping them in the center of the screen
        target_camera_x = self.player_rect.centerx - self.WIDTH / 2
        # Clamp camera to not show area before the level starts or after it ends
        self.camera_offset_x = max(0, min(target_camera_x, self.LEVEL_END_X - self.WIDTH))

    def _update_world(self):
        # Dynamically generate new platforms if the player is approaching the end of the generated world
        if self.platforms and self.platforms[-1].right - self.camera_offset_x < self.WIDTH + 200:
            self._generate_next_platform()

        # Prune old platforms and coins that are far off-screen to the left
        self.platforms = [p for p in self.platforms if p.right - self.camera_offset_x > -200]
        self.coins = [c for c in self.coins if c.right - self.camera_offset_x > -200]


    def _generate_initial_platforms(self):
        self.platforms = []
        self.coins = []
        
        # Create a starting platform
        start_plat = pygame.Rect(0, self.HEIGHT - 50, 300, 50)
        self.platforms.append(start_plat)
        
        # Generate platforms to fill the initial screen
        while self.platforms[-1].right < self.WIDTH + 200:
            self._generate_next_platform()
            
    def _generate_next_platform(self):
        last_plat = self.platforms[-1]
        
        # Difficulty scaling: gap increases with steps
        gap = self.INITIAL_GAP + (self.steps // 50) * 0.5
        
        # New platform position and size
        new_x = last_plat.right + gap + self.np_random.integers(0, 30)
        
        y_change = self.np_random.integers(-self.MAX_Y_CHANGE, self.MAX_Y_CHANGE)
        new_y = last_plat.top + y_change
        new_y = int(np.clip(new_y, 100, self.HEIGHT - 50)) # Clamp y to be on screen

        new_width = self.np_random.integers(self.MIN_PLATFORM_WIDTH, self.MAX_PLATFORM_WIDTH)
        
        new_plat = pygame.Rect(new_x, new_y, new_width, self.HEIGHT - new_y) # Platform extends to bottom
        self.platforms.append(new_plat)
        
        # Add a coin to the new platform (with 50% probability)
        if self.np_random.random() < 0.5:
            coin_x = new_plat.left + self.np_random.integers(10, new_plat.width - 10)
            coin_y = new_plat.top - 20 # 20px above the platform
            self.coins.append(pygame.Rect(coin_x, coin_y, 15, 15))


    def _get_observation(self):
        # --- Render Background ---
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # --- Render Game Elements (with camera offset) ---
        # Render Platforms
        for plat in self.platforms:
            # Only draw platforms that are on screen
            if plat.right > self.camera_offset_x and plat.left < self.camera_offset_x + self.WIDTH:
                render_rect = plat.move(-self.camera_offset_x, 0)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, render_rect)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, render_rect, 2)
        
        # Render Coins
        coin_bob = math.sin(self.steps * 0.2) * 3
        for coin in self.coins:
            if coin.right > self.camera_offset_x and coin.left < self.camera_offset_x + self.WIDTH:
                render_x = int(coin.centerx - self.camera_offset_x)
                render_y = int(coin.centery + coin_bob)
                radius = coin.width // 2
                pygame.gfxdraw.filled_circle(self.screen, render_x, render_y, radius, self.COLOR_COIN)
                pygame.gfxdraw.aacircle(self.screen, render_x, render_y, radius, self.COLOR_COIN_OUTLINE)
        
        # Render Player
        player_render_rect = self.player_rect.move(-self.camera_offset_x, 0)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_render_rect)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_render_rect, 2)

        # --- Render UI ---
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        progress = (self.player_rect.left / self.LEVEL_END_X) * 100
        progress_text = self.small_font.render(f"Progress: {progress:.1f}%", True, self.COLOR_TEXT)
        self.screen.blit(progress_text, (self.WIDTH - progress_text.get_width() - 10, 10))

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_x": self.player_rect.x,
            "progress_percent": (self.player_rect.left / self.LEVEL_END_X) * 100
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Platform Jumper")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        # elif keys[pygame.K_DOWN]: movement = 2 # No effect
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0.0

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            print("Press 'R' to reset.")

        clock.tick(env.FPS)

    env.close()