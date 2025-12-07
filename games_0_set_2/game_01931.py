
# Generated: 2025-08-28T03:08:25.481777
# Source Brief: brief_01931.md
# Brief Index: 1931

        
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
        "Controls: Hold [SPACE] to charge a jump. Release to jump. Use ← and → to aim your jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between procedurally generated platforms, collecting coins to reach the target score before falling off-screen in this side-scrolling arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    WIN_SCORE = 50
    MAX_STEPS = 3000

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 10)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (50, 255, 50, 50)
    COLOR_PLATFORM = (150, 150, 170)
    COLOR_COIN = (255, 223, 0)
    COLOR_COIN_GLOW = (255, 223, 0, 60)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE = (200, 200, 255)

    # Player Physics
    GRAVITY = 0.8
    PLAYER_SIZE = 12
    JUMP_CHARGE_RATE = 0.7
    MAX_JUMP_CHARGE = 18
    JUMP_X_SPEED = 8
    AIR_CONTROL_STRENGTH = 0.5
    X_FRICTION = 0.95
    BOUNCE_FACTOR = -0.2 # Small bounce on landing

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # Internal state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.platforms = deque()
        self.coins = deque()
        self.particles = deque()
        self.camera_y = 0
        self.jump_charge = 0
        self.on_platform = False
        self.platform_speed_multiplier = 1.0
        self.highest_platform_y = self.HEIGHT

        # Run validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_y = 0
        self.jump_charge = 0
        self.on_platform = True
        self.platform_speed_multiplier = 1.0
        self.highest_platform_y = self.HEIGHT

        self.player = {
            "x": self.WIDTH // 2,
            "y": self.HEIGHT - 50,
            "vx": 0,
            "vy": 0,
            "rect": pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        }

        self.platforms.clear()
        self.coins.clear()
        self.particles.clear()

        # Generate initial platforms
        start_platform = self._create_platform(
            self.WIDTH // 2, self.HEIGHT - 30, self.WIDTH
        )
        self.platforms.append(start_platform)
        self._generate_initial_platforms()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(self.FPS)
        self.steps += 1
        reward = 0

        # 1. Unpack action and handle player input
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward += self._handle_input(movement, space_held)

        # 2. Update game state
        self._update_player(movement)
        self._update_platforms()
        self._update_particles()
        
        # 3. Handle collisions and interactions
        landed, platform_landed_on = self._handle_collisions()
        if landed:
            reward += 0.1 # Reward for landing
        
        collected_coin = self._handle_coin_collection()
        if collected_coin:
            reward += 1.0 # Reward for collecting a coin
            if self.score > 0 and self.score % 10 == 0:
                 self.platform_speed_multiplier += 0.05 # Increase difficulty

        if not self.on_platform:
            reward -= 0.01 # Small penalty for being in the air (encourages staying on platforms)

        # 4. Update camera and generate new platforms
        self._update_camera()
        self._manage_platforms_and_coins()

        # 5. Check for termination
        terminated = False
        if self.player["y"] > self.camera_y + self.HEIGHT + 50:
            terminated = True
            reward = -100.0
        elif self.score >= self.WIN_SCORE:
            terminated = True
            reward = 100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        # Handle jump charging and execution
        if self.on_platform:
            if space_held:
                self.jump_charge = min(self.MAX_JUMP_CHARGE, self.jump_charge + self.JUMP_CHARGE_RATE)
            elif self.jump_charge > 0:
                # Execute jump
                self.player["vy"] = -self.jump_charge
                if movement == 3: # Left
                    self.player["vx"] = -self.JUMP_X_SPEED
                elif movement == 4: # Right
                    self.player["vx"] = self.JUMP_X_SPEED
                else: # Up or None
                    self.player["vx"] = 0
                
                self.jump_charge = 0
                self.on_platform = False
                # sfx: jump
                self._spawn_particles(self.player["x"], self.player["y"], 10, self.COLOR_PARTICLE)
        return 0

    def _update_player(self, movement):
        # Apply gravity if not on a platform
        if not self.on_platform:
            self.player["vy"] += self.GRAVITY
            
            # Air control
            if movement == 3: # Left
                self.player["vx"] -= self.AIR_CONTROL_STRENGTH
            elif movement == 4: # Right
                self.player["vx"] += self.AIR_CONTROL_STRENGTH

        # Apply friction and update position
        self.player["vx"] *= self.X_FRICTION
        self.player["x"] += self.player["vx"]
        self.player["y"] += self.player["vy"]

        # Keep player within horizontal bounds
        if self.player["x"] < self.PLAYER_SIZE / 2:
            self.player["x"] = self.PLAYER_SIZE / 2
            self.player["vx"] = 0
        if self.player["x"] > self.WIDTH - self.PLAYER_SIZE / 2:
            self.player["x"] = self.WIDTH - self.PLAYER_SIZE / 2
            self.player["vx"] = 0
        
        # Update player rect
        self.player["rect"].center = (self.player["x"], self.player["y"])

    def _handle_collisions(self):
        landed_this_frame = False
        platform_landed_on = None
        
        # Only check for landing if falling
        if self.player["vy"] > 0:
            for platform in self.platforms:
                # Simple AABB collision check
                if self.player["rect"].colliderect(platform["rect"]):
                    # Check if player was above platform in previous frame
                    if self.player["y"] - self.player["vy"] <= platform["rect"].top:
                        self.player["y"] = platform["rect"].top - self.PLAYER_SIZE / 2
                        self.player["vy"] = self.BOUNCE_FACTOR * self.player["vy"]
                        self.on_platform = True
                        landed_this_frame = True
                        platform_landed_on = platform
                        # sfx: land
                        self._spawn_particles(self.player["x"], self.player["y"], 5, self.COLOR_PLATFORM)
                        break
        
        if self.on_platform and platform_landed_on:
             self.player["x"] += platform_landed_on["vx"]
             self.player["y"] = platform_landed_on["rect"].top - self.PLAYER_SIZE / 2

        return landed_this_frame, platform_landed_on
    
    def _handle_coin_collection(self):
        for coin in list(self.coins):
            if self.player["rect"].colliderect(coin["rect"]):
                self.coins.remove(coin)
                self.score += 1
                # sfx: coin_collect
                self._spawn_particles(coin["rect"].centerx, coin["rect"].centery, 15, self.COLOR_COIN)
                return True
        return False

    def _update_platforms(self):
        for p in self.platforms:
            p["t"] += 1
            p["rect"].y = p["y0"] + math.sin(p["t"] * p["freq"] * self.platform_speed_multiplier) * p["amp"]

    def _update_camera(self):
        # Smoothly follow the player upwards
        target_camera_y = self.player["y"] - self.HEIGHT * 0.6
        self.camera_y += (target_camera_y - self.camera_y) * 0.1

    def _manage_platforms_and_coins(self):
        # Remove off-screen bottom platforms
        while self.platforms and self.platforms[0]["rect"].top > self.camera_y + self.HEIGHT:
            self.platforms.popleft()
        
        while self.coins and self.coins[0]["rect"].top > self.camera_y + self.HEIGHT:
            self.coins.popleft()

        # Generate new platforms at the top
        while self.highest_platform_y > self.camera_y - 50:
            self._generate_new_platform()

    def _generate_initial_platforms(self):
        for _ in range(15):
            self._generate_new_platform()
            
    def _generate_new_platform(self):
        last_platform = self.platforms[-1]
        
        # Position new platform relative to the last one
        new_x = last_platform["rect"].centerx + self.np_random.uniform(-150, 150)
        new_x = np.clip(new_x, 50, self.WIDTH - 50)
        
        new_y = last_platform["rect"].y - self.np_random.uniform(70, 120)

        width = self.np_random.uniform(80, 150)
        
        new_platform = self._create_platform(new_x, new_y, width)
        self.platforms.append(new_platform)
        self.highest_platform_y = new_y

        # Add a coin to the platform with 80% probability
        if self.np_random.random() < 0.8:
            coin_rect = pygame.Rect(0, 0, 10, 10)
            coin_rect.center = (new_platform["rect"].centerx, new_platform["rect"].top - 20)
            self.coins.append({"rect": coin_rect})
    
    def _create_platform(self, x, y, width):
        return {
            "rect": pygame.Rect(x - width / 2, y, width, 15),
            "y0": y,
            "amp": self.np_random.uniform(0, 30),
            "freq": self.np_random.uniform(0.01, 0.03),
            "t": self.np_random.uniform(0, 2 * math.pi),
            "vx": 0 # For future features like moving platforms
        }

    def _update_particles(self):
        for p in list(self.particles):
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _spawn_particles(self, x, y, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": self.np_random.integers(10, 20),
                "color": color
            })
            
    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            color = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(
                self.screen,
                int(p["x"]), int(p["y"] - self.camera_y),
                max(0, int(p["life"] / 4)),
                color
            )

        # Render coins
        for coin in self.coins:
            pos = (int(coin["rect"].centerx), int(coin["rect"].centery - self.camera_y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_COIN_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, self.COLOR_COIN)

        # Render platforms
        for p in self.platforms:
            cam_rect = p["rect"].move(0, -self.camera_y)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, cam_rect, border_radius=3)

        # Render player
        player_pos = (int(self.player["x"]), int(self.player["y"] - self.camera_y))
        
        # Player glow
        pygame.gfxdraw.filled_circle(self.screen, player_pos[0], player_pos[1], 20, self.COLOR_PLAYER_GLOW)
        
        # Player body (crouch effect when charging)
        crouch_factor = self.jump_charge / self.MAX_JUMP_CHARGE
        size = self.PLAYER_SIZE
        body_rect = pygame.Rect(
            player_pos[0] - size / 2,
            player_pos[1] - size / 2 + (size * 0.4 * crouch_factor),
            size,
            size * (1 - 0.4 * crouch_factor)
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=2)

        # Player jump charge indicator
        if self.jump_charge > 0:
            charge_width = self.jump_charge / self.MAX_JUMP_CHARGE * 30
            charge_rect = pygame.Rect(player_pos[0] - 15, player_pos[1] + 15, charge_width, 4)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, charge_rect)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Coin icon
        pygame.gfxdraw.filled_circle(self.screen, 80, 30, 10, self.COLOR_COIN)
        pygame.gfxdraw.aacircle(self.screen, 80, 30, 10, self.COLOR_COIN)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to get observation
        obs, info = self.reset()
        
        # Test observation space  
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test reset output
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