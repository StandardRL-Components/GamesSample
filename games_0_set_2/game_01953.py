
# Generated: 2025-08-28T03:12:22.406387
# Source Brief: brief_01953.md
# Brief Index: 1953

        
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

    user_guide = (
        "Controls: ←→ to run, ↑ to jump. Collect coins and reach the finish line!"
    )

    game_description = (
        "Guide a robot through a procedural obstacle course. "
        "Collect coins and perform risky jumps for a high score."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and rendering setup
        self.W, self.H = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 36)
        self.font_l = pygame.font.Font(None, 48)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Game Constants
        self.MAX_STEPS = 2000
        self.GRAVITY = 0.6
        self.PLAYER_JUMP = -11
        self.PLAYER_SPEED = 5.0
        self.LEVEL_LENGTH = 100 # Number of segments

        # Colors
        self.COLOR_BG = (26, 26, 26)
        self.COLOR_GRID = (40, 40, 40)
        self.COLOR_PLATFORM = (68, 68, 68)
        self.COLOR_ROBOT = (51, 153, 255)
        self.COLOR_ROBOT_EYE = (255, 255, 255)
        self.COLOR_OBSTACLE = (255, 51, 51)
        self.COLOR_COIN = (255, 204, 0)
        self.COLOR_FINISH = (0, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE_COIN = (255, 230, 100)
        self.COLOR_PARTICLE_JUMP = (100, 180, 255, 150)

        # Initialize state variables
        self.player = {}
        self.level = []
        self.obstacles = []
        self.coins = []
        self.particles = deque()
        self.camera_offset_x = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.obstacle_speed_modifier = 1.0
        self.finish_x = 0

        self.reset()
        
        # This check is for development and can be removed in production
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.obstacle_speed_modifier = 1.0
        self.camera_offset_x = 0.0
        self.particles.clear()

        player_start_y = self.H // 2
        self.player = {
            "rect": pygame.Rect(100, player_start_y, 24, 32),
            "vx": 0.0,
            "vy": 0.0,
            "on_ground": False,
            "is_over_gap": False,
        }

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.level = []
        self.obstacles = []
        self.coins = []
        x = 0
        platform_y = self.H * 0.75
        
        # Start platform
        start_width = 300
        self.level.append({"rect": pygame.Rect(x, platform_y, start_width, self.H - platform_y), "type": "platform"})
        x += start_width

        for i in range(self.LEVEL_LENGTH):
            gap_chance = 0.2
            obstacle_chance = 0.3
            coin_chance = 0.6

            if self.np_random.random() < gap_chance and i > 0: # Don't start with a gap
                gap_width = self.np_random.integers(80, 150)
                self.level.append({"rect": pygame.Rect(x, 0, gap_width, self.H), "type": "gap"})
                x += gap_width
            
            platform_width = self.np_random.integers(200, 500)
            platform_dy = self.np_random.integers(-60, 61)
            platform_y = np.clip(platform_y + platform_dy, self.H * 0.5, self.H * 0.85)
            
            platform_rect = pygame.Rect(x, platform_y, platform_width, self.H - platform_y)
            self.level.append({"rect": platform_rect, "type": "platform"})

            # Add coins
            if self.np_random.random() < coin_chance:
                num_coins = self.np_random.integers(3, 7)
                for j in range(num_coins):
                    cx = platform_rect.left + (platform_width / (num_coins + 1)) * (j + 1)
                    cy = platform_rect.top - self.np_random.integers(40, 80)
                    self.coins.append({"rect": pygame.Rect(cx - 8, cy - 8, 16, 16), "angle": self.np_random.random() * math.pi * 2, "initial_y": cy})

            # Add obstacles
            if self.np_random.random() < obstacle_chance and platform_width > 150:
                ox = platform_rect.centerx
                obstacle_type = self.np_random.choice(['gear', 'pendulum'])
                
                if obstacle_type == 'gear':
                    self.obstacles.append({
                        "type": "gear", "pos": (ox, platform_rect.top - 40), 
                        "radius": 35, "angle": self.np_random.random() * math.pi * 2, "speed": self.np_random.uniform(0.02, 0.04)
                    })
                else: # pendulum
                    self.obstacles.append({
                        "type": "pendulum", "pivot": (ox, platform_rect.top - 120), "length": 100,
                        "angle": math.pi / 4, "speed": self.np_random.uniform(0.03, 0.05), "base_angle": self.np_random.random() * math.pi * 2
                    })
            
            x += platform_width
        
        self.finish_x = x + 100

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        reward = -0.01  # Cost of living

        # --- Player Logic ---
        # Horizontal movement
        if movement == 3: # Left
            self.player["vx"] = -self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player["vx"] = self.PLAYER_SPEED
            reward += 0.1
        else:
            self.player["vx"] *= 0.8 # Friction

        # Vertical movement (Jump)
        if movement == 1 and self.player["on_ground"]:
            self.player["vy"] = self.PLAYER_JUMP
            self.player["on_ground"] = False
            # Sound: Jump

        # Apply gravity
        self.player["vy"] += self.GRAVITY
        self.player["vy"] = min(self.player["vy"], 15) # Terminal velocity

        # Update position
        self.player["rect"].x += int(self.player["vx"])
        self.player["rect"].y += int(self.player["vy"])
        
        self.player["on_ground"] = False
        
        # --- Collision Detection & Resolution ---
        is_currently_over_gap = True
        for segment in self.level:
            if segment["type"] == "platform" and self.player["rect"].colliderect(segment["rect"]):
                is_currently_over_gap = False
                if self.player["vy"] > 0 and self.player["rect"].bottom > segment["rect"].top:
                    # Landed on a platform
                    self.player["rect"].bottom = segment["rect"].top
                    self.player["vy"] = 0
                    self.player["on_ground"] = True
                    if self.player["is_over_gap"]: # Just completed a risky jump
                        reward += 5.0
                        self._spawn_particles(self.player["rect"].midbottom, self.COLOR_PARTICLE_JUMP, 20, is_risky_jump=True)
                        # Sound: Powerup/Success
                    self.player["is_over_gap"] = False
        
        if is_currently_over_gap:
            self.player["is_over_gap"] = True

        # Coin collection
        collected_coins = []
        for coin in self.coins:
            if self.player["rect"].colliderect(coin["rect"]):
                collected_coins.append(coin)
                self.score += 10
                reward += 1.0
                self._spawn_particles(coin["rect"].center, self.COLOR_PARTICLE_COIN, 10)
                # Sound: Coin collect

        self.coins = [c for c in self.coins if c not in collected_coins]

        # --- Update Game State ---
        self.steps += 1
        if self.steps % 200 == 0:
            self.obstacle_speed_modifier += 0.05
        
        self._update_obstacles()
        self._update_particles()
        
        # --- Termination Conditions ---
        terminated = False
        
        # 1. Fall off map
        if self.player["rect"].top > self.H + 50:
            terminated = True
            reward -= 100
            # Sound: Fall/Failure
        
        # 2. Hit obstacle
        for obs in self.obstacles:
            if self._check_obstacle_collision(obs):
                terminated = True
                reward -= 100
                # Sound: Explosion/Hit
                break
        
        # 3. Reach finish line
        if self.player["rect"].centerx > self.finish_x:
            terminated = True
            win_bonus = 100 * (self.MAX_STEPS - self.steps) / self.MAX_STEPS
            reward += win_bonus
            self.score += 1000
            # Sound: Victory
        
        # 4. Max steps
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_obstacles(self):
        for obs in self.obstacles:
            speed = obs["speed"] * self.obstacle_speed_modifier
            if obs["type"] == 'gear':
                obs["angle"] += speed
            elif obs["type"] == 'pendulum':
                obs["angle"] = math.pi / 3 * math.sin(obs["base_angle"] + self.steps * speed)
    
    def _check_obstacle_collision(self, obs):
        if obs["type"] == 'gear':
            for i in range(4):
                angle = obs["angle"] + i * math.pi / 2
                arm_len = obs["radius"]
                p1 = (obs["pos"][0] - math.cos(angle) * arm_len, obs["pos"][1] - math.sin(angle) * arm_len)
                p2 = (obs["pos"][0] + math.cos(angle) * arm_len, obs["pos"][1] + math.sin(angle) * arm_len)
                if self.player["rect"].clipline(p1, p2):
                    return True
        elif obs["type"] == 'pendulum':
            bob_pos = (
                obs["pivot"][0] + math.sin(obs["angle"]) * obs["length"],
                obs["pivot"][1] + math.cos(obs["angle"]) * obs["length"]
            )
            bob_rect = pygame.Rect(bob_pos[0] - 12, bob_pos[1] - 12, 24, 24)
            if self.player["rect"].colliderect(bob_rect):
                return True
        return False

    def _spawn_particles(self, pos, color, count, is_risky_jump=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            if is_risky_jump:
                vy = -abs(vy) # Force upwards
            self.particles.append({
                "pos": list(pos), "vel": [vx, vy], "life": self.np_random.integers(20, 40), "color": color
            })

    def _update_particles(self):
        for i in range(len(self.particles) -1, -1, -1):
            p = self.particles[i]
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity on particles
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.camera_offset_x = self.player["rect"].centerx - self.W / 3

        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        grid_size = 50
        start_x = -int(self.camera_offset_x * 0.5) % grid_size
        for x in range(start_x - grid_size, self.W, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.H))
        for y in range(0, self.H, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.W, y))

        # --- World Elements (Platforms, Obstacles, Coins) ---
        for segment in self.level:
            screen_rect = segment["rect"].move(-self.camera_offset_x, 0)
            if screen_rect.right < 0 or screen_rect.left > self.W:
                continue
            if segment["type"] == "platform":
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect)

        for coin in self.coins:
            screen_pos = (int(coin["rect"].centerx - self.camera_offset_x), int(coin["rect"].centery))
            if -20 < screen_pos[0] < self.W + 20:
                coin["angle"] += 0.1
                bob = math.sin(coin["angle"] * 2) * 3
                size_factor = (math.sin(coin["angle"]) + 1.1) / 2.1
                w = int(coin["rect"].width * size_factor)
                h = coin["rect"].height
                r = pygame.Rect(screen_pos[0] - w // 2, screen_pos[1] - h // 2 + bob, w, h)
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, r)
                pygame.gfxdraw.aaellipse(self.screen, r.centerx, r.centery, r.width // 2, r.height // 2, self.COLOR_COIN)
        
        for obs in self.obstacles:
            if obs["type"] == 'gear':
                screen_pos = (int(obs["pos"][0] - self.camera_offset_x), int(obs["pos"][1]))
                if -50 < screen_pos[0] < self.W + 50:
                    for i in range(4):
                        angle = obs["angle"] + i * math.pi / 2
                        p1 = (screen_pos[0] - math.cos(angle) * obs["radius"], screen_pos[1] - math.sin(angle) * obs["radius"])
                        p2 = (screen_pos[0] + math.cos(angle) * obs["radius"], screen_pos[1] + math.sin(angle) * obs["radius"])
                        pygame.draw.aaline(self.screen, self.COLOR_OBSTACLE, p1, p2, 2)
            elif obs["type"] == 'pendulum':
                pivot_screen = (int(obs["pivot"][0] - self.camera_offset_x), obs["pivot"][1])
                if -150 < pivot_screen[0] < self.W + 150:
                    bob_pos = (
                        pivot_screen[0] + math.sin(obs["angle"]) * obs["length"],
                        pivot_screen[1] + math.cos(obs["angle"]) * obs["length"]
                    )
                    pygame.draw.aaline(self.screen, self.COLOR_PLATFORM, pivot_screen, bob_pos)
                    pygame.gfxdraw.filled_circle(self.screen, int(bob_pos[0]), int(bob_pos[1]), 12, self.COLOR_OBSTACLE)
                    pygame.gfxdraw.aacircle(self.screen, int(bob_pos[0]), int(bob_pos[1]), 12, self.COLOR_OBSTACLE)

        # --- Finish Line ---
        finish_screen_x = int(self.finish_x - self.camera_offset_x)
        if 0 < finish_screen_x < self.W:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_screen_x, 0), (finish_screen_x, self.H), 3)
            finish_text = self.font_m.render("FINISH", True, self.COLOR_FINISH)
            self.screen.blit(finish_text, (finish_screen_x + 10, 20))

        # --- Particles ---
        for p in self.particles:
            screen_pos = (int(p["pos"][0] - self.camera_offset_x), int(p["pos"][1]))
            life_ratio = p["life"] / 40.0
            radius = int(life_ratio * 5)
            if radius > 0:
                color = p["color"]
                if len(color) == 4: # RGBA
                    surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(surf, (*color[:3], int(color[3] * life_ratio)), (radius, radius), radius)
                    self.screen.blit(surf, (screen_pos[0] - radius, screen_pos[1] - radius))
                else: # RGB
                    pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, color)

        # --- Player ---
        player_screen_rect = self.player["rect"].copy()
        player_screen_rect.centerx = self.W / 3
        
        # Bobbing animation
        bob = 0
        if self.player["on_ground"] and abs(self.player["vx"]) > 0.1:
            bob = abs(math.sin(self.steps * 0.5)) * 4
        
        body_rect = pygame.Rect(player_screen_rect.left, player_screen_rect.top + bob, player_screen_rect.width, player_screen_rect.height - bob)
        eye_y = body_rect.top + 10
        eye_x_offset = 5 * (1 if self.player["vx"] >= 0 else -1)
        eye_pos = (body_rect.centerx + eye_x_offset, eye_y)
        
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, body_rect, border_radius=4)
        pygame.draw.circle(self.screen, self.COLOR_ROBOT_EYE, eye_pos, 4)

        # --- UI ---
        score_text = self.font_m.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        timer_text = self.font_m.render(f"TIME: {self.steps}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.W - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_x": self.player["rect"].x,
            "player_y": self.player["rect"].y,
            "finish_x": self.finish_x,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Procedural Platformer")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # --- Human Controls ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    action = [0, 0, 0]
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action = [0, 0, 0]
        
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
            action[0] = 0 # No-op for movement
        
        # Other buttons
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the rendered frame
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # and surfarray.make_surface expects (W, H, C)
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Optional: wait a bit before closing or resetting
            pygame.time.wait(2000)

        clock.tick(30) # Run at 30 FPS

    env.close()