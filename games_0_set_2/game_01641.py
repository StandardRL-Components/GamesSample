
# Generated: 2025-08-28T02:13:41.578884
# Source Brief: brief_01641.md
# Brief Index: 1641

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
    user_guide = (
        "Controls: ←→ to run, ↑ or Space to jump. Collect coins and reach the flag!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced pixel art platformer. Jump across platforms, collect coins, and reach the goal while avoiding deadly pits."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        
        # Colors
        self.COLOR_BG = (135, 206, 235)  # Sky Blue
        self.COLOR_PLAYER = (255, 69, 0) # Bright Red-Orange
        self.COLOR_PLATFORM = (100, 100, 100) # Grey
        self.COLOR_PLATFORM_TOP = (120, 120, 120)
        self.COLOR_COIN = (255, 215, 0) # Gold
        self.COLOR_PIT = (0, 0, 0) # Black
        self.COLOR_FLAME_1 = (255, 69, 0)
        self.COLOR_FLAME_2 = (255, 140, 0)
        self.COLOR_FLAGPOLE = (192, 192, 192) # Silver
        self.COLOR_FLAG = (220, 20, 60) # Crimson
        self.COLOR_CLOUD = (255, 255, 255)

        # Physics and game constants
        self.GRAVITY = 0.6
        self.JUMP_STRENGTH = -12
        self.PLAYER_SPEED = 5
        self.FRICTION = 0.8
        self.MAX_VEL_Y = 15
        self.MAX_STEPS = 1000

        # Level definition
        self._define_level()
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.is_grounded = None
        self.coins = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.squash_frames = 0
        self.run_anim_frame = 0

        self.validate_implementation()

    def _define_level(self):
        """Defines the static layout of the level."""
        self.player_start_pos = [50, 250]
        
        # Platforms are defined as [x, y, width, height]
        platform_data = [
            [0, 350, 250, 50], [300, 300, 150, 100], [500, 250, 140, 150],
            [200, 180, 100, 20]
        ]
        self.platforms = [pygame.Rect(p[0], p[1], p[2], p[3]) for p in platform_data]
        
        # Pits are just rectangles
        pit_data = [[250, 380, 50, 20]]
        self.pits = [pygame.Rect(p[0], p[1], p[2], p[3]) for p in pit_data]
        
        # Coins are just positions
        self.master_coins = [
            (150, 320), (180, 320), (210, 320),
            (350, 150), (380, 140), (410, 150),
            (530, 220), (560, 220), (590, 220)
        ]
        
        # Flag position
        self.flag_pos = (600, 190)

        # Background clouds
        self.clouds = [
            pygame.Rect(100, 80, 80, 30), pygame.Rect(120, 70, 40, 20),
            pygame.Rect(300, 120, 100, 40), pygame.Rect(330, 105, 40, 25),
            pygame.Rect(500, 60, 90, 35), pygame.Rect(480, 50, 50, 20)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = list(self.player_start_pos)
        self.player_vel = [0, 0]
        self.player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], 24, 32)
        self.is_grounded = False
        
        self.coins = [pygame.Rect(c[0], c[1], 16, 16) for c in self.master_coins]
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.squash_frames = 0
        self.run_anim_frame = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Unpack action and handle input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 3:  # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel[0] = self.PLAYER_SPEED
        
        # Jump action
        if (movement == 1 or space_held) and self.is_grounded:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.is_grounded = False
            self.squash_frames = -4 # Stretch effect
            # sfx: jump

        # 2. Update game logic
        reward = 0
        
        # Apply friction if not actively moving
        if movement not in [3, 4]:
            self.player_vel[0] *= self.FRICTION

        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        self.player_vel[1] = min(self.player_vel[1], self.MAX_VEL_Y)

        # 3. Move and resolve collisions
        # Move horizontally
        self.player_pos[0] += self.player_vel[0]
        self.player_rect.x = int(self.player_pos[0])
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel[0] > 0: # Moving right
                    self.player_rect.right = plat.left
                elif self.player_vel[0] < 0: # Moving left
                    self.player_rect.left = plat.right
                self.player_pos[0] = self.player_rect.x
                self.player_vel[0] = 0

        # Move vertically
        self.player_pos[1] += self.player_vel[1]
        self.player_rect.y = int(self.player_pos[1])
        self.is_grounded = False
        for plat in self.platforms:
            if self.player_rect.colliderect(plat):
                if self.player_vel[1] > 0: # Moving down
                    if self.player_rect.bottom > plat.top and self.player_pos[1] - self.player_vel[1] + self.player_rect.height <= plat.top + 1:
                        self.player_rect.bottom = plat.top
                        self.player_vel[1] = 0
                        if not self.is_grounded: # Just landed
                            self.squash_frames = 4 # Squash effect
                            # sfx: land
                        self.is_grounded = True
                elif self.player_vel[1] < 0: # Moving up
                    self.player_rect.top = plat.bottom
                    self.player_vel[1] = 0
        self.player_pos[1] = self.player_rect.y

        # Clamp player to screen
        self.player_pos[0] = max(0, min(self.player_pos[0], self.WIDTH - self.player_rect.width))
        self.player_rect.x = int(self.player_pos[0])

        # 4. Check for events (coins, pits, flag)
        # Coin collection
        collected_coins = []
        for coin in self.coins:
            if self.player_rect.colliderect(coin):
                collected_coins.append(coin)
                self.score += 1
                reward += 1
                # sfx: coin_collect
        self.coins = [c for c in self.coins if c not in collected_coins]

        # Pit collision
        for pit in self.pits:
            if self.player_rect.colliderect(pit):
                self.game_over = True
                reward = -100
                # sfx: player_die
        
        # Off-screen bottom
        if self.player_rect.top > self.HEIGHT:
            self.game_over = True
            reward = -100
            # sfx: player_die

        # End flag
        flag_rect = pygame.Rect(self.flag_pos[0], self.flag_pos[1], 10, 60)
        if self.player_rect.colliderect(flag_rect):
            self.game_over = True
            reward = 100
            self.score += 50
            # sfx: level_complete

        # Update step counter and check for termination
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        # Animation updates
        if abs(self.player_vel[0]) > 1 and self.is_grounded:
            self.run_anim_frame += 1
        if self.squash_frames > 0:
            self.squash_frames -= 1
        elif self.squash_frames < 0:
            self.squash_frames += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
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

    def _render_game(self):
        # Draw clouds
        for cloud in self.clouds:
            pygame.draw.rect(self.screen, self.COLOR_CLOUD, cloud, border_radius=10)

        # Draw platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
            top_rect = pygame.Rect(plat.x, plat.y, plat.width, 5)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, top_rect)

        # Draw pits with animated flames
        for pit in self.pits:
            pygame.draw.rect(self.screen, self.COLOR_PIT, pit)
            for i in range(pit.width // 10):
                flame_height = self.np_random.integers(5, 15)
                flame_x = pit.x + i * 10 + self.np_random.integers(-2, 3)
                flame_y = pit.y + pit.height - flame_height
                flame_color = self.COLOR_FLAME_1 if self.steps % 10 < 5 else self.COLOR_FLAME_2
                pygame.draw.rect(self.screen, flame_color, (flame_x, flame_y, 8, flame_height))
        
        # Draw coins with spinning animation
        for coin in self.coins:
            spin_phase = (self.steps + coin.x) % 60 / 60.0
            scale = abs(math.cos(spin_phase * 2 * math.pi))
            width = int(coin.width * scale)
            if width > 1:
                display_coin = pygame.Rect(coin.centerx - width // 2, coin.y, width, coin.height)
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, display_coin)
                pygame.draw.ellipse(self.screen, (255, 255, 150), display_coin, 2)

        # Draw end flag
        pole_rect = pygame.Rect(self.flag_pos[0], self.flag_pos[1], 5, 60)
        flag_points = [
            (self.flag_pos[0] + 5, self.flag_pos[1]),
            (self.flag_pos[0] + 35, self.flag_pos[1] + 10),
            (self.flag_pos[0] + 5, self.flag_pos[1] + 20)
        ]
        pygame.draw.rect(self.screen, self.COLOR_FLAGPOLE, pole_rect)
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)

        # Draw player with animation
        p_rect = self.player_rect.copy()
        if self.squash_frames > 0: # Squash on land
            p_rect.height *= 0.8
            p_rect.width *= 1.2
            p_rect.y += self.player_rect.height * 0.2
        elif self.squash_frames < 0: # Stretch on jump
            p_rect.height *= 1.2
            p_rect.width *= 0.8
        
        p_rect.centerx = self.player_rect.centerx

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, p_rect, border_radius=4)
        
        # Running animation
        if self.is_grounded and abs(self.player_vel[0]) > 1:
            leg_height = 6
            leg_width = 8
            y_pos = p_rect.bottom
            if (self.run_anim_frame // 4) % 2 == 0:
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, (p_rect.left, y_pos, leg_width, leg_height))
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, (p_rect.right - leg_width, y_pos, leg_width, leg_height))
            else:
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, (p_rect.centerx - leg_width, y_pos, leg_width, leg_height))

    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        text_surf = self.font_ui.render(score_text, True, (255, 255, 255))
        shadow_surf = self.font_ui.render(score_text, True, (0, 0, 0))
        self.screen.blit(shadow_surf, (12, 12))
        self.screen.blit(text_surf, (10, 10))

        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        text_surf = self.font_ui.render(steps_text, True, (255, 255, 255))
        shadow_surf = self.font_ui.render(steps_text, True, (0, 0, 0))
        text_width = text_surf.get_width()
        self.screen.blit(shadow_surf, (self.WIDTH - text_width - 8, 12))
        self.screen.blit(text_surf, (self.WIDTH - text_width - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": tuple(self.player_pos),
            "player_vel": tuple(self.player_vel),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Platformer")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        # Action defaults
        movement = 0 # no-op
        space = 0
        shift = 0

        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Get key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        # K_DOWN is 2, but has no action
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control FPS
        clock.tick(30)
        
        if done:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Optional: wait a bit before reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False
            total_reward = 0

    env.close()