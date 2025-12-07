
# Generated: 2025-08-28T06:03:17.740338
# Source Brief: brief_02797.md
# Brief Index: 2797

        
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

    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Collect coins and reach the green flag!"
    )

    game_description = (
        "Navigate a procedurally generated pixel-art platformer, collecting coins and reaching the level's end. Avoid falling!"
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (20, 30, 50) # Dark Blue
    COLOR_PLATFORM = (139, 69, 19) # Brown
    COLOR_PLATFORM_OUTLINE = (90, 45, 12)
    COLOR_COIN = (255, 223, 0) # Yellow
    COLOR_COIN_RISKY = (255, 120, 0) # Orange
    COLOR_PLAYER = (255, 50, 50) # Red
    COLOR_FLAG = (50, 205, 50) # Green
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_PARTICLE = (255, 255, 150)

    # Physics
    GRAVITY = 0.5
    FRICTION = 0.9
    PLAYER_ACCEL = 1.0
    PLAYER_MAX_SPEED = 5.0
    JUMP_STRENGTH = 11.0
    
    # Player
    PLAYER_SIZE = (24, 32)
    
    # Level
    MAX_STEPS = 2000
    LEVEL_WIDTH_PIXELS = 6400 # 10 screens wide
    INITIAL_LIVES = 3
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.lives = None
        self.game_over = None
        self.game_won = None
        
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        
        self.platforms = None
        self.coins = None
        self.flag_rect = None
        
        self.particles = None
        self.background_stars = None
        
        self.camera_x = None
        self.last_player_x = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.game_won = False
        
        self.player_pos = np.array([100.0, 200.0])
        self.player_vel = np.array([0.0, 0.0])
        self.on_ground = False
        
        self.camera_x = 0
        self.last_player_x = self.player_pos[0]
        
        self.particles = []
        
        self._generate_level()
        
        # Find starting platform and place player on it
        for p in self.platforms:
            if p.left <= self.player_pos[0] <= p.right:
                self.player_pos[1] = p.top - self.PLAYER_SIZE[1]
                break

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.coins = []
        
        # Starting platform
        start_platform = pygame.Rect(50, 300, 200, 50)
        self.platforms.append(start_platform)
        
        current_x = start_platform.right
        last_y = start_platform.y
        
        # Procedural generation loop
        while current_x < self.LEVEL_WIDTH_PIXELS - 500:
            # Difficulty scaling
            progress_ratio = current_x / self.LEVEL_WIDTH_PIXELS
            min_gap = 40 + 80 * progress_ratio
            max_gap = 100 + 100 * progress_ratio
            
            gap = self.np_random.integers(min_gap, max_gap)
            current_x += gap
            
            width = self.np_random.integers(80, 250)
            height = self.np_random.integers(30, 80)
            
            y_diff = self.np_random.integers(-100, 100)
            new_y = np.clip(last_y + y_diff, 150, self.SCREEN_HEIGHT - 50)
            
            platform_rect = pygame.Rect(current_x, new_y, width, height)
            self.platforms.append(platform_rect)
            
            # Add coins
            coin_chance = 0.5 + 0.4 * progress_ratio
            if self.np_random.random() < coin_chance:
                num_coins = self.np_random.integers(1, 5)
                for i in range(num_coins):
                    coin_x = platform_rect.left + (i + 1) * (platform_rect.width / (num_coins + 1))
                    coin_y = platform_rect.top - 30
                    # coin_data: [x, y, type(0=normal, 1=risky), initial_y]
                    self.coins.append([coin_x, coin_y, 0, coin_y])

            # Add risky coins in gaps
            risky_coin_chance = 0.1 + 0.3 * progress_ratio
            if self.np_random.random() < risky_coin_chance and gap > 80:
                 coin_x = current_x - gap / 2
                 coin_y = last_y - self.np_random.integers(20, 60)
                 self.coins.append([coin_x, coin_y, 1, coin_y])

            last_y = new_y
            current_x += width

        # End flag
        last_platform = self.platforms[-1]
        self.flag_rect = pygame.Rect(last_platform.centerx - 10, last_platform.top - 60, 20, 60)

        # Generate static background stars
        self.background_stars = []
        for _ in range(100):
            x = self.np_random.integers(0, self.LEVEL_WIDTH_PIXELS)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            size = self.np_random.integers(1, 3)
            self.background_stars.append((x, y, size))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, _, _ = action
        reward = -0.01  # Small penalty for existing

        # 1. Update Player based on action
        if movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL
        
        if movement == 1 and self.on_ground:  # Jump
            self.player_vel[1] = -self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump

        # 2. Apply Physics
        self.player_vel[0] = np.clip(self.player_vel[0], -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)
        self.player_vel[0] *= self.FRICTION
        if abs(self.player_vel[0]) < 0.1: self.player_vel[0] = 0
            
        self.player_vel[1] += self.GRAVITY
        
        self.player_pos += self.player_vel

        # 3. Collision Detection and Resolution
        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])
        
        for platform in self.platforms:
            if player_rect.colliderect(platform):
                # Check if player was above in the previous step
                prev_player_bottom = player_rect.bottom - self.player_vel[1]
                if self.player_vel[1] > 0 and prev_player_bottom <= platform.top:
                    self.player_pos[1] = platform.top - self.PLAYER_SIZE[1]
                    self.player_vel[1] = 0
                    self.on_ground = True
                    player_rect.bottom = platform.top # Update rect for next checks
                # Collision from below
                elif self.player_vel[1] < 0 and player_rect.top - self.player_vel[1] >= platform.bottom:
                    self.player_pos[1] = platform.bottom
                    self.player_vel[1] = 0
                    player_rect.top = platform.bottom
                # Horizontal collision
                else:
                    if self.player_vel[0] > 0 and player_rect.right - self.player_vel[0] <= platform.left:
                        self.player_pos[0] = platform.left - self.PLAYER_SIZE[0]
                        self.player_vel[0] = 0
                    elif self.player_vel[0] < 0 and player_rect.left - self.player_vel[0] >= platform.right:
                        self.player_pos[0] = platform.right
                        self.player_vel[0] = 0
        
        # World boundaries
        self.player_pos[0] = max(0, min(self.player_pos[0], self.LEVEL_WIDTH_PIXELS - self.PLAYER_SIZE[0]))

        # 4. Collect Coins
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])
        for coin in self.coins[:]:
            coin_rect = pygame.Rect(coin[0] - 5, coin[1] - 5, 10, 10)
            if player_rect.colliderect(coin_rect):
                is_risky = coin[2] == 1
                reward += 5.0 if is_risky else 1.0
                self.score += 50 if is_risky else 10
                self.coins.remove(coin)
                # sfx: coin_collect
                # Create particles
                for _ in range(10):
                    self.particles.append(Particle(coin[0], coin[1], self.np_random))

        # 5. Update Particles
        for p in self.particles[:]:
            p.update()
            if p.is_dead():
                self.particles.remove(p)

        # 6. Reward for progress
        progress = self.player_pos[0] - self.last_player_x
        if progress > 0:
            reward += progress * 0.1
        self.last_player_x = self.player_pos[0]

        # 7. Check Termination Conditions
        terminated = False
        # Fell off screen
        if self.player_pos[1] > self.SCREEN_HEIGHT + 50:
            self.lives -= 1
            reward -= 50
            # sfx: lose_life
            if self.lives <= 0:
                self.game_over = True
                terminated = True
            else: # Respawn
                self.player_pos = np.array([100.0, 200.0])
                self.player_vel = np.array([0.0, 0.0])
                for p in self.platforms:
                    if p.left <= self.player_pos[0] <= p.right:
                        self.player_pos[1] = p.top - self.PLAYER_SIZE[1]
                        break
                self.last_player_x = self.player_pos[0]

        # Reached flag
        if player_rect.colliderect(self.flag_rect):
            reward += 100
            self.game_over = True
            self.game_won = True
            terminated = True
            # sfx: level_complete

        # Max steps
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        # 8. Update Camera
        self.camera_x += (self.player_pos[0] - self.camera_x - self.SCREEN_WIDTH / 2.5) * 0.1
        self.camera_x = max(0, min(self.camera_x, self.LEVEL_WIDTH_PIXELS - self.SCREEN_WIDTH))

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x = int(self.camera_x)

        # Render background stars (parallax)
        for x, y, size in self.background_stars:
            px = (x - cam_x * 0.5) % self.SCREEN_WIDTH
            pygame.draw.rect(self.screen, (80, 100, 140), (int(px), int(y), size, size))

        # Render platforms
        for p in self.platforms:
            if p.right >= cam_x and p.left <= cam_x + self.SCREEN_WIDTH:
                render_rect = p.move(-cam_x, 0)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, render_rect)
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, render_rect, 2)
        
        # Render coins
        for coin in self.coins:
            x, y, type, initial_y = coin
            if cam_x <= x <= cam_x + self.SCREEN_WIDTH:
                # Bobbing animation
                bob_offset = math.sin(self.steps * 0.1 + x) * 5
                render_x, render_y = int(x - cam_x), int(y + bob_offset)
                color = self.COLOR_COIN_RISKY if type == 1 else self.COLOR_COIN
                pygame.gfxdraw.filled_circle(self.screen, render_x, render_y, 6, color)
                pygame.gfxdraw.aacircle(self.screen, render_x, render_y, 6, (255, 255, 255, 100))

        # Render flag
        if self.flag_rect.right >= cam_x and self.flag_rect.left <= cam_x + self.SCREEN_WIDTH:
            pole_rect = self.flag_rect.copy()
            pole_rect.width = 4
            pole_rect.centerx = self.flag_rect.centerx
            pygame.draw.rect(self.screen, (200, 200, 200), pole_rect.move(-cam_x, 0))
            
            flag_points = [
                (self.flag_rect.left - cam_x, self.flag_rect.top),
                (self.flag_rect.left - cam_x, self.flag_rect.top + 20),
                (self.flag_rect.left - cam_x - 30, self.flag_rect.top + 10)
            ]
            pygame.gfxdraw.filled_trigon(self.screen, flag_points[0][0], flag_points[0][1], flag_points[1][0], flag_points[1][1], flag_points[2][0], flag_points[2][1], self.COLOR_FLAG)

        # Render particles
        for p in self.particles:
            p.draw(self.screen, cam_x)

        # Render player
        if not (self.game_over and self.lives <= 0):
            player_render_rect = pygame.Rect(int(self.player_pos[0] - cam_x), int(self.player_pos[1]), self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_render_rect)
            pygame.draw.rect(self.screen, (255, 150, 150), player_render_rect, 2)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER" if self.lives <= 0 else "LEVEL COMPLETE!"
            color = (255, 80, 80) if self.lives <= 0 else (80, 255, 80)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "player_pos": (self.player_pos[0], self.player_pos[1]),
            "game_won": self.game_won
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Reset to get a valid observation
        self.reset()
        
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

class Particle:
    def __init__(self, x, y, np_random):
        self.np_random = np_random
        self.x = x
        self.y = y
        angle = self.np_random.random() * 2 * math.pi
        speed = self.np_random.random() * 2 + 1
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = 20
        self.color = GameEnv.COLOR_PARTICLE
        self.size = self.np_random.integers(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.vx *= 0.95
        self.vy *= 0.95
    
    def is_dead(self):
        return self.lifespan <= 0

    def draw(self, surface, camera_x):
        alpha = max(0, 255 * (self.lifespan / 20))
        temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, self.color + (int(alpha),), (self.size, self.size), self.size)
        surface.blit(temp_surf, (int(self.x - camera_x - self.size), int(self.y - self.size)))