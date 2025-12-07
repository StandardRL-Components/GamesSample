
# Generated: 2025-08-28T05:48:59.171846
# Source Brief: brief_05702.md
# Brief Index: 5702

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to jump, ←→ to move."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A procedurally generated pixel-art platformer. Collect coins and reach the exit flag as fast as you can!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG_SKY = (135, 206, 235)
    COLOR_PLAYER = (60, 179, 113)
    COLOR_PLAYER_OUTLINE = (46, 139, 87)
    COLOR_PLATFORM = (139, 69, 19)
    COLOR_PLATFORM_OUTLINE = (92, 51, 23)
    COLOR_COIN = (255, 215, 0)
    COLOR_COIN_OUTLINE = (218, 165, 32)
    COLOR_EXIT_POLE = (128, 128, 128)
    COLOR_EXIT_FLAG = (255, 0, 0)
    COLOR_PIT = (0, 0, 0)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (0, 0, 0)

    # Physics
    GRAVITY = 0.8
    PLAYER_JUMP_STRENGTH = -15
    PLAYER_MOVE_SPEED = 5
    MAX_FALL_SPEED = 20
    PLAYER_FRICTION = 0.85

    # Player
    PLAYER_SIZE = pygame.Vector2(24, 32)
    
    # World Generation
    PLATFORM_HEIGHT = 20
    PLATFORM_WIDTH_MIN = 3
    PLATFORM_WIDTH_MAX = 8
    PLATFORM_TILE_SIZE = 20
    GAP_WIDTH_MIN = 2
    GAP_WIDTH_MAX = 5
    PLATFORM_LEVEL_MIN = 300
    PLATFORM_LEVEL_MAX = 350
    PLATFORM_LEVEL_CHANGE = 40
    COIN_SPAWN_CHANCE = 0.3
    LEVEL_LENGTH = 300 # in tiles

    # Game
    MAX_STEPS = 500
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

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
        
        try:
            self.font_ui = pygame.font.Font(pygame.font.match_font('monospace', bold=True), 24)
        except:
            self.font_ui = pygame.font.SysFont('monospace', 24, bold=True)

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_ground = False
        self.player_land_squash = 0

        self.platforms = []
        self.coins = []
        self.exit_flag_rect = None
        
        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_player_x = 0
        self.coin_spin_timer = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.coin_spin_timer = 0
        
        self.player_pos = pygame.Vector2(150, 150)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_ground = False
        self.last_player_x = self.player_pos.x

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.coins = []
        
        current_x = 0
        current_y = self.PLATFORM_LEVEL_MAX
        
        # Starting platform
        start_platform_width = 10 * self.PLATFORM_TILE_SIZE
        self.platforms.append(pygame.Rect(current_x, current_y, start_platform_width, self.SCREEN_HEIGHT - current_y))
        current_x += start_platform_width

        # Procedural generation loop
        while current_x < self.LEVEL_LENGTH * self.PLATFORM_TILE_SIZE:
            # Add a gap
            gap_width = self.np_random.integers(self.GAP_WIDTH_MIN, self.GAP_WIDTH_MAX + 1) * self.PLATFORM_TILE_SIZE
            current_x += gap_width

            # Change platform height
            current_y += self.np_random.integers(-self.PLATFORM_LEVEL_CHANGE, self.PLATFORM_LEVEL_CHANGE + 1)
            current_y = np.clip(current_y, self.PLATFORM_LEVEL_MIN, self.PLATFORM_LEVEL_MAX)

            # Add a new platform
            platform_width = self.np_random.integers(self.PLATFORM_WIDTH_MIN, self.PLATFORM_WIDTH_MAX + 1) * self.PLATFORM_TILE_SIZE
            platform_rect = pygame.Rect(current_x, current_y, platform_width, self.SCREEN_HEIGHT - current_y)
            self.platforms.append(platform_rect)

            # Add coins above the platform
            for i in range(1, int(platform_width / self.PLATFORM_TILE_SIZE)):
                if self.np_random.random() < self.COIN_SPAWN_CHANCE:
                    coin_x = current_x + i * self.PLATFORM_TILE_SIZE
                    coin_y = current_y - self.PLATFORM_TILE_SIZE * 2
                    self.coins.append(pygame.Rect(coin_x, coin_y, self.PLATFORM_TILE_SIZE, self.PLATFORM_TILE_SIZE))

            current_x += platform_width
        
        # Add exit flag at the end
        exit_x = current_x + 100
        exit_y = current_y - 80
        self.exit_flag_rect = pygame.Rect(exit_x, exit_y, 10, 80)


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        terminated = False
        
        # --- Player Input ---
        target_vel_x = 0
        if movement == 3:  # Left
            target_vel_x = -self.PLAYER_MOVE_SPEED
        elif movement == 4:  # Right
            target_vel_x = self.PLAYER_MOVE_SPEED
        
        self.player_vel.x = target_vel_x

        if movement == 1 and self.player_on_ground:  # Jump
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.player_on_ground = False
            self.player_land_squash = 0
            # sfx: jump

        # --- Physics and Collision ---
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)

        # Move horizontally
        self.player_pos.x += self.player_vel.x
        player_rect = self._get_player_rect()
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                if self.player_vel.x > 0: # Moving right
                    player_rect.right = plat.left
                elif self.player_vel.x < 0: # Moving left
                    player_rect.left = plat.right
                self.player_pos.x = player_rect.x

        # Move vertically
        self.player_pos.y += self.player_vel.y
        player_rect = self._get_player_rect()
        
        was_on_ground = self.player_on_ground
        self.player_on_ground = False
        
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                if self.player_vel.y > 0: # Falling
                    player_rect.bottom = plat.top
                    self.player_vel.y = 0
                    self.player_on_ground = True
                    if not was_on_ground:
                        self.player_land_squash = 5 # Start squash animation
                        # sfx: land
                elif self.player_vel.y < 0: # Hitting ceiling
                    player_rect.top = plat.bottom
                    self.player_vel.y = 0
                self.player_pos.y = player_rect.y

        # --- Reward for progress ---
        progress = self.player_pos.x - self.last_player_x
        reward += progress * 0.1
        self.last_player_x = self.player_pos.x

        # --- Coin Collection ---
        player_rect = self._get_player_rect()
        collected_coins = []
        for coin in self.coins:
            if player_rect.colliderect(coin):
                collected_coins.append(coin)
                self.score += 1
                reward += 1
                # sfx: coin_collect
        self.coins = [c for c in self.coins if c not in collected_coins]

        # --- Termination Conditions ---
        # Fall into pit
        if self.player_pos.y > self.SCREEN_HEIGHT:
            reward = -50
            terminated = True
            self.game_over = True
            # sfx: fall_death

        # Reach exit
        if player_rect.colliderect(self.exit_flag_rect):
            reward = 100
            self.score += 100
            terminated = True
            self.game_over = True
            # sfx: win_level
        
        # Max steps
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_player_rect(self):
        return pygame.Rect(self.player_pos, self.PLAYER_SIZE)

    def _get_observation(self):
        # Update camera
        self.camera_x += (self.player_pos.x - self.camera_x - self.SCREEN_WIDTH / 3) * 0.1
        self.camera_x = max(0, self.camera_x)

        # Update animations
        self.coin_spin_timer = (self.coin_spin_timer + 1) % 60
        if self.player_land_squash > 0:
            self.player_land_squash -= 1

        # Clear screen
        self.screen.fill(self.COLOR_BG_SKY)
        
        # Render all game elements
        self._render_level()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_level(self):
        cam_x = int(self.camera_x)

        # Draw platforms
        for plat in self.platforms:
            if plat.right - cam_x > 0 and plat.left - cam_x < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, (plat.x - cam_x, plat.y, plat.width, plat.height))
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (plat.x - cam_x + 2, plat.y + 2, plat.width - 4, plat.height - 4))

        # Draw coins
        spin_offset = abs(10 - (self.coin_spin_timer % 20)) / 10
        for coin in self.coins:
             if coin.right - cam_x > 0 and coin.left - cam_x < self.SCREEN_WIDTH:
                w = int(coin.width * (1 - spin_offset * 0.5))
                h = coin.height
                x = coin.x - cam_x + int(coin.width * spin_offset * 0.25)
                y = coin.y
                pygame.draw.rect(self.screen, self.COLOR_COIN_OUTLINE, (x, y, w, h))
                pygame.draw.rect(self.screen, self.COLOR_COIN, (x+1, y+1, w-2, h-2))

        # Draw exit flag
        if self.exit_flag_rect:
            if self.exit_flag_rect.right - cam_x > 0 and self.exit_flag_rect.left - cam_x < self.SCREEN_WIDTH:
                # Pole
                pole_rect = self.exit_flag_rect.copy()
                pole_rect.x -= cam_x
                pygame.draw.rect(self.screen, self.COLOR_EXIT_POLE, pole_rect)
                # Flag
                flag_points = [
                    (pole_rect.right, pole_rect.top),
                    (pole_rect.right, pole_rect.top + 30),
                    (pole_rect.right + 40, pole_rect.top + 15)
                ]
                pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_EXIT_FLAG)
                pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_EXIT_FLAG)

    def _render_player(self):
        cam_x = int(self.camera_x)
        
        squash_h = int(self.player_land_squash * 1.5)
        squash_w = int(self.player_land_squash * 1.5)
        
        player_render_rect = pygame.Rect(
            int(self.player_pos.x - cam_x - squash_w / 2),
            int(self.player_pos.y + squash_h),
            int(self.PLAYER_SIZE.x + squash_w),
            int(self.PLAYER_SIZE.y - squash_h)
        )
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_render_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_render_rect.inflate(-4, -4), border_radius=3)

    def _render_ui(self):
        score_text = f"SCORE: {int(self.score)}"
        time_text = f"TIME: {self.MAX_STEPS - self.steps}"
        
        # Score
        shadow_surf = self.font_ui.render(score_text, True, self.COLOR_UI_SHADOW)
        self.screen.blit(shadow_surf, (12, 12))
        text_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Timer
        shadow_surf = self.font_ui.render(time_text, True, self.COLOR_UI_SHADOW)
        self.screen.blit(shadow_surf, (12, 42))
        text_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_x": self.player_pos.x,
            "player_y": self.player_pos.y
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up a window to display the game
    pygame.display.set_caption("Pixel Platformer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    total_steps = 0
    
    while not done:
        # --- Human Controls ---
        movement = 0 # no-op
        keys = pygame.key.get()
        
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_UP]:
            movement = 1
        
        action = [movement, 0, 0] # Movement, space, shift
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        total_steps += 1

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Rendering ---
        # Convert the observation (which is a numpy array) back to a Pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {total_steps}")
    env.close()