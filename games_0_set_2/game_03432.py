
# Generated: 2025-08-27T23:19:44.017046
# Source Brief: brief_03432.md
# Brief Index: 3432

        
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
        "Controls: Use ← and → to aim your jump. Hold SPACE for a high jump or SHIFT for a short hop. Press ↑ to jump straight up."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between procedurally generated platforms, collect coins, and reach a target score before plummeting into the cosmic abyss."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WIN_CONDITION_COINS = 50
        self.MAX_STEPS = 2000

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_HOPPER = (57, 255, 20)
        self.COLOR_PLATFORM = (240, 240, 255)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (200, 200, 255)

        # Physics
        self.GRAVITY = 0.4
        self.JUMP_POWER_HIGH = -11
        self.JUMP_POWER_NORMAL = -9
        self.JUMP_POWER_LOW = -7
        self.HORIZONTAL_SPEED = 5
        self.AIR_FRICTION = 0.98

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.large_font = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Internal state variables
        self.steps = 0
        self.score = 0
        self.coins_collected = 0
        self.game_over = False
        self.game_won = False
        self.np_random = None
        self.hopper = {}
        self.platforms = []
        self.coins = []
        self.stars = []
        self.particles = []
        self.camera_y = 0
        self.platform_speed = 1.0

        # Per-step reward flags
        self.just_landed = False
        self.coin_collected_this_step = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.coins_collected = 0
        self.game_over = False
        self.game_won = False
        self.camera_y = 0
        self.platform_speed = 1.0
        
        # Hopper state
        self.hopper = {
            "pos": pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.75),
            "vel": pygame.Vector2(0, 0),
            "size": pygame.Vector2(24, 24),
            "on_platform": True,
            "squash": 1.0
        }

        # Clear and generate world
        self.platforms.clear()
        self.coins.clear()
        self.stars.clear()
        self.particles.clear()
        self._generate_initial_world()

        # Center camera on hopper
        self.camera_y = self.hopper["pos"].y - self.SCREEN_HEIGHT * 0.5
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        self.just_landed = False
        self.coin_collected_this_step = False

        self._handle_input(action)
        self._update_physics()
        self._update_difficulty()
        self._update_camera()
        self._cull_and_generate_objects()
        
        reward = self._calculate_reward()
        terminated = self._check_termination()

        if terminated:
            if self.game_won:
                reward += 100
            else: # Fell or timed out
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        if not self.hopper["on_platform"]:
            return

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        jump_intent = movement in [1, 3, 4] # Up, Left, Right
        if not jump_intent:
            return

        # Determine jump power
        if space_held:
            jump_power = self.JUMP_POWER_HIGH
        elif shift_held:
            jump_power = self.JUMP_POWER_LOW
        else:
            jump_power = self.JUMP_POWER_NORMAL

        # Apply jump velocity
        self.hopper["vel"].y = jump_power
        if movement == 3: # Left
            self.hopper["vel"].x = -self.HORIZONTAL_SPEED
        elif movement == 4: # Right
            self.hopper["vel"].x = self.HORIZONTAL_SPEED
        
        self.hopper["on_platform"] = False
        self.hopper["squash"] = 1.5 # Stretch for jump
        # sfx: jump
        self._create_particles(self.hopper["pos"] + pygame.Vector2(0, self.hopper["size"].y/2), 15)

    def _update_physics(self):
        # Update hopper
        if not self.hopper["on_platform"]:
            self.hopper["vel"].y += self.GRAVITY
            self.hopper["pos"] += self.hopper["vel"]
            self.hopper["vel"].x *= self.AIR_FRICTION

        # Hopper squash/stretch effect
        self.hopper["squash"] += (1.0 - self.hopper["squash"]) * 0.2

        # Boundary checks
        if self.hopper["pos"].x < self.hopper["size"].x / 2:
            self.hopper["pos"].x = self.hopper["size"].x / 2
            self.hopper["vel"].x *= -0.5
        if self.hopper["pos"].x > self.SCREEN_WIDTH - self.hopper["size"].x / 2:
            self.hopper["pos"].x = self.SCREEN_WIDTH - self.hopper["size"].x / 2
            self.hopper["vel"].x *= -0.5

        self._handle_platform_collisions()
        self._handle_coin_collisions()

        # Update platforms, coins, particles
        for p in self.platforms:
            p["phase"] += 0.05 * self.platform_speed
            p["rect"].y = p["base_y"] + math.sin(p["phase"]) * p["amplitude"]
        for c in self.coins:
            c["angle"] = (c["angle"] + 5) % 360
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"].y += 0.1
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _handle_platform_collisions(self):
        if self.hopper["vel"].y < 0:
            return

        hopper_rect = pygame.Rect(
            self.hopper["pos"].x - self.hopper["size"].x / 2,
            self.hopper["pos"].y - self.hopper["size"].y / 2,
            self.hopper["size"].x,
            self.hopper["size"].y
        )

        for p in self.platforms:
            is_above = hopper_rect.bottom - self.hopper["vel"].y <= p["rect"].top
            if is_above and p["rect"].colliderect(hopper_rect):
                self.hopper["pos"].y = p["rect"].top - self.hopper["size"].y / 2
                self.hopper["vel"].y = 0
                self.hopper["vel"].x = 0
                self.hopper["on_platform"] = True
                self.just_landed = True
                self.hopper["squash"] = 0.6 # Squash on landing
                # sfx: land
                break

    def _handle_coin_collisions(self):
        hopper_rect = pygame.Rect(
            self.hopper["pos"].x - self.hopper["size"].x / 2,
            self.hopper["pos"].y - self.hopper["size"].y / 2,
            self.hopper["size"].x,
            self.hopper["size"].y
        )
        for c in self.coins[:]:
            if hopper_rect.colliderect(c["rect"]):
                self.coins.remove(c)
                self.score += 1
                self.coins_collected += 1
                self.coin_collected_this_step = True
                # sfx: coin_collect
                self._create_particles(c["rect"].center, 10, self.COLOR_COIN)

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.platform_speed = min(3.0, self.platform_speed + 0.01)

    def _update_camera(self):
        target_y = self.hopper["pos"].y - self.SCREEN_HEIGHT * 0.5
        self.camera_y += (target_y - self.camera_y) * 0.1

    def _cull_and_generate_objects(self):
        cull_line = self.camera_y + self.SCREEN_HEIGHT + 100
        self.platforms = [p for p in self.platforms if p["rect"].bottom > self.camera_y - 100]
        self.coins = [c for c in self.coins if c["rect"].bottom > self.camera_y - 100]

        topmost_platform_y = min([p["rect"].y for p in self.platforms] or [self.SCREEN_HEIGHT])
        while topmost_platform_y > self.camera_y - 100:
            new_y = topmost_platform_y - self.np_random.integers(60, 120)
            self._generate_platform(new_y)
            topmost_platform_y = new_y

    def _generate_initial_world(self):
        # Starting platform
        start_y = self.hopper["pos"].y + self.hopper["size"].y / 2 + 5
        start_platform = {
            "rect": pygame.Rect(self.SCREEN_WIDTH/2 - 50, start_y, 100, 20),
            "base_y": start_y, "phase": 0, "amplitude": 0
        }
        self.platforms.append(start_platform)
        
        # Procedural platforms
        current_y = start_y
        for _ in range(20):
            current_y -= self.np_random.integers(60, 120)
            self._generate_platform(current_y)
            
        # Background stars
        for _ in range(150):
            self.stars.append({
                "pos": pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                "depth": self.np_random.uniform(0.1, 0.7),
                "size": self.np_random.uniform(1, 2.5)
            })

    def _generate_platform(self, y_pos):
        last_x = self.platforms[-1]["rect"].centerx if self.platforms else self.SCREEN_WIDTH / 2
        width = self.np_random.integers(60, 140)
        x_offset = self.np_random.integers(-150, 150)
        x_pos = np.clip(last_x + x_offset, width / 2, self.SCREEN_WIDTH - width / 2)
        
        platform = {
            "rect": pygame.Rect(x_pos - width / 2, y_pos, width, 20),
            "base_y": y_pos,
            "phase": self.np_random.uniform(0, 2 * math.pi),
            "amplitude": self.np_random.integers(0, 30) if y_pos < self.SCREEN_HEIGHT * 0.5 else 0
        }
        self.platforms.append(platform)

        if self.np_random.random() < 0.4: # 40% chance of coin
            coin_pos = pygame.Vector2(platform["rect"].centerx, platform["rect"].top - 20)
            self.coins.append({
                "rect": pygame.Rect(coin_pos.x - 10, coin_pos.y - 10, 20, 20),
                "angle": self.np_random.uniform(0, 360)
            })

    def _create_particles(self, pos, count, color=None):
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "lifespan": self.np_random.integers(15, 30),
                "color": color
            })

    def _calculate_reward(self):
        reward = 0.0
        if self.hopper["vel"].y < 0: # Moving up
            reward += 0.1
        else: # Moving down
            reward -= 0.1
        
        if self.coin_collected_this_step:
            reward += 1.0
        
        if self.just_landed:
            reward += 0.5
            
        return reward

    def _check_termination(self):
        if self.coins_collected >= self.WIN_CONDITION_COINS:
            self.game_over = True
            self.game_won = True
        elif self.hopper["pos"].y > self.camera_y + self.SCREEN_HEIGHT + self.hopper["size"].y:
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_background()
        self._render_game_objects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.stars:
            x = star["pos"].x
            y = (star["pos"].y - self.camera_y * star["depth"]) % self.SCREEN_HEIGHT
            size = star["size"]
            brightness = 150 + 105 * star["depth"]
            color = (brightness, brightness, brightness)
            pygame.draw.circle(self.screen, color, (int(x), int(y)), int(size/2))

    def _render_game_objects(self):
        # Render Platforms
        for p in self.platforms:
            draw_rect = p["rect"].copy()
            draw_rect.y -= self.camera_y
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, draw_rect, border_radius=3)

        # Render Coins
        for c in self.coins:
            draw_pos = (int(c["rect"].centerx), int(c["rect"].centery - self.camera_y))
            pygame.gfxdraw.filled_circle(self.screen, draw_pos[0], draw_pos[1], 10, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, draw_pos[0], draw_pos[1], 10, self.COLOR_COIN)
            # Shine effect
            angle_rad = math.radians(c["angle"])
            line_start = (draw_pos[0] + math.cos(angle_rad) * 5, draw_pos[1] + math.sin(angle_rad) * 5)
            line_end = (draw_pos[0] - math.cos(angle_rad) * 5, draw_pos[1] - math.sin(angle_rad) * 5)
            pygame.draw.aaline(self.screen, (255, 255, 150), line_start, line_end, 2)
        
        # Render Particles
        for p in self.particles:
            draw_pos = (int(p["pos"].x), int(p["pos"].y - self.camera_y))
            alpha = max(0, 255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2,2), 2)
            self.screen.blit(temp_surf, (draw_pos[0]-2, draw_pos[1]-2))

        # Render Hopper
        squashed_height = self.hopper["size"].y * self.hopper["squash"]
        squashed_width = self.hopper["size"].x / self.hopper["squash"]
        hopper_draw_pos = (self.hopper["pos"].x, self.hopper["pos"].y - self.camera_y)
        hopper_rect = pygame.Rect(
            hopper_draw_pos[0] - squashed_width / 2,
            hopper_draw_pos[1] - squashed_height / 2,
            squashed_width,
            squashed_height
        )
        pygame.draw.rect(self.screen, self.COLOR_HOPPER, hopper_rect, border_radius=int(squashed_width/4))
    
    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        coins_text = self.font.render(f"COINS: {self.coins_collected}/{self.WIN_CONDITION_COINS}", True, self.COLOR_TEXT)
        text_rect = coins_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(coins_text, text_rect)

        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_HOPPER if self.game_won else (255, 50, 50)
            end_text = self.large_font.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "coins_collected": self.coins_collected
        }

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a separate display for human playing
    live_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Cosmic Hopper")
    
    total_reward = 0
    total_steps = 0
    
    action = env.action_space.sample() # Start with a default action
    action.fill(0) # No-op

    while not done:
        # --- Human Controls ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0

        # Buttons
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        total_steps += 1
        
        # --- Render to Live Display ---
        # The observation is (H, W, C), but pygame wants (W, H) surface
        # And the surfarray is transposed, so we need to fix it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        live_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control framerate
        env.clock.tick(30)

    print(f"Game Over!")
    print(f"Total Steps: {total_steps}")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")

    env.close()