
# Generated: 2025-08-27T14:14:09.786244
# Source Brief: brief_00616.md
# Brief Index: 616

        
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
    metadata = {"render_modes": ["rgb_array", "human"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Collect all the coins and reach the green flag!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist platformer. Navigate procedurally generated levels, collect coins, "
        "and reach the goal as fast as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Game Constants ---
        self.MAX_STEPS = 1500
        self.GRAVITY = 0.4
        self.PLAYER_SPEED = 4.0
        self.JUMP_STRENGTH = -10.0
        self.PLAYER_SIZE = 20
        self.FRICTION = 0.9

        # --- Visuals ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLATFORM = (240, 240, 240)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_FLAG = (50, 255, 50)
        self.COLOR_PARTICLE = (255, 255, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.font = pygame.font.Font(None, 36)
        
        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.is_grounded = False
        self.platforms = []
        self.coins = []
        self.particles = []
        self.end_flag = None
        self.camera_x = 0
        self.last_player_x = 0
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([100.0, 300.0])
        self.player_vel = np.array([0.0, 0.0])
        self.last_player_x = self.player_pos[0]
        
        self.is_grounded = False
        self.camera_x = 0
        
        self.particles = []
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.coins = []
        
        # Start platform
        start_platform = pygame.Rect(0, 350, 250, 50)
        self.platforms.append(start_platform)
        
        last_x = start_platform.right
        last_y = start_platform.top
        
        level_width = 4000
        
        while last_x < level_width:
            x_gap = self.np_random.integers(40, 100)
            y_change = self.np_random.integers(-80, 80)
            width = self.np_random.integers(100, 300)
            
            new_x = last_x + x_gap
            new_y = np.clip(last_y + y_change, 150, self.SCREEN_HEIGHT - 50)
            
            platform_rect = pygame.Rect(new_x, new_y, width, self.SCREEN_HEIGHT - new_y)
            self.platforms.append(platform_rect)
            
            # Add a coin above the platform
            coin_pos = (platform_rect.centerx, platform_rect.top - 30)
            self.coins.append(pygame.Rect(coin_pos[0] - 10, coin_pos[1] - 10, 20, 20))
            
            last_x = platform_rect.right
            last_y = platform_rect.top

        # End flag
        flag_x = last_x + 50
        flag_y = last_y - 50
        self.end_flag = pygame.Rect(flag_x, flag_y, 20, 50)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        # --- Handle Input ---
        if movement == 3:  # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel[0] = self.PLAYER_SPEED
        else: # No horizontal input, apply friction
             self.player_vel[0] *= self.FRICTION

        if movement == 1 and self.is_grounded:  # Jump
            self.player_vel[1] = self.JUMP_STRENGTH
            self.is_grounded = False
            # sfx: jump

        # --- Physics and Collision ---
        self.player_vel[1] += self.GRAVITY
        
        # Horizontal movement and collision
        self.player_pos[0] += self.player_vel[0]
        player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        for platform in self.platforms:
            if player_rect.colliderect(platform):
                if self.player_vel[0] > 0: # Moving right
                    player_rect.right = platform.left
                elif self.player_vel[0] < 0: # Moving left
                    player_rect.left = platform.right
                self.player_pos[0] = float(player_rect.left)
                self.player_vel[0] = 0

        # Vertical movement and collision
        self.player_pos[1] += self.player_vel[1]
        player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.is_grounded = False

        for platform in self.platforms:
             if player_rect.colliderect(platform):
                if self.player_vel[1] > 0: # Moving down
                    player_rect.bottom = platform.top
                    self.is_grounded = True
                    self.player_vel[1] = 0
                elif self.player_vel[1] < 0: # Moving up
                    player_rect.top = platform.bottom
                    self.player_vel[1] = 0
                self.player_pos[1] = float(player_rect.top)
        
        # --- Reward Calculation ---
        progress = self.player_pos[0] - self.last_player_x
        reward = progress * 0.1
        self.last_player_x = self.player_pos[0]

        # --- Interactions ---
        player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        for coin in self.coins[:]:
            if player_rect.colliderect(coin):
                self.coins.remove(coin)
                self.score += 1
                reward += 1.0
                # sfx: coin_pickup
                self._create_particles(coin.center)

        # --- Update Game State ---
        self._update_particles()
        self._update_camera()
        self.steps += 1
        
        # --- Termination Check ---
        terminated = False
        if player_rect.colliderect(self.end_flag):
            terminated = True
            reward += 100.0
            # sfx: level_complete
        elif self.player_pos[1] > self.SCREEN_HEIGHT + 50:
            terminated = True
            reward -= 100.0
            # sfx: fall_death
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifetime})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_camera(self):
        target_camera_x = self.player_pos[0] - self.SCREEN_WIDTH / 3
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x = int(self.camera_x)
        
        for platform in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, platform.move(-cam_x, 0))

        for coin in self.coins:
            pygame.gfxdraw.filled_circle(self.screen, coin.centerx - cam_x, coin.centery, 10, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, coin.centerx - cam_x, coin.centery, 10, self.COLOR_COIN)

        pygame.draw.rect(self.screen, self.COLOR_FLAG, self.end_flag.move(-cam_x, 0))
        
        for p in self.particles:
            size = max(0, int(p['life'] / 5))
            p_rect = pygame.Rect(p['pos'][0] - cam_x - size/2, p['pos'][1] - size/2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_PARTICLE, p_rect)

        player_rect_render = pygame.Rect(
            int(self.player_pos[0] - cam_x), 
            int(self.player_pos[1]), 
            self.PLAYER_SIZE, 
            self.PLAYER_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect_render)
        pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in self.COLOR_PLAYER), player_rect_render, 2)

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()