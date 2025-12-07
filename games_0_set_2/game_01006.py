
# Generated: 2025-08-27T15:28:34.352796
# Source Brief: brief_01006.md
# Brief Index: 1006

        
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

    user_guide = (
        "Controls: ←→ to run, ↑ or Space to jump. Collect coins and reach the green flag!"
    )

    game_description = (
        "A minimalist pixel-art platformer. Jump across procedurally generated platforms, "
        "collect coins for points, and reach the goal flag to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLATFORM = (100, 100, 120)
        self.COLOR_COIN = (255, 220, 0)
        self.COLOR_FLAG = (80, 220, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE = (255, 255, 150)

        # Game Physics & Constants
        self.GRAVITY = 0.6
        self.MAX_FALL_SPEED = 10
        self.PLAYER_JUMP_STRENGTH = -11
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = -0.15
        self.PLAYER_MAX_SPEED = 6.0
        self.MAX_STEPS = 1000

        # Game State (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_size = np.array([24, 24])
        self.on_ground = False
        self.squash_stretch_factor = 1.0
        self.squash_timer = 0
        
        self.platforms = []
        self.coins = []
        self.particles = []
        self.flag_pos = np.array([0, 0])
        self.flag_size = (10, 30)
        self.last_dist_to_flag = 0.0

        # Initialize state by calling reset
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.particles.clear()
        self._generate_level()

        start_platform = self.platforms[0]
        self.player_pos = np.array([start_platform.centerx, start_platform.top - self.player_size[1]], dtype=float)
        self.player_vel = np.array([0.0, 0.0])
        self.on_ground = True
        self.squash_stretch_factor = 1.0
        self.squash_timer = 0

        self.last_dist_to_flag = abs(self.player_pos[0] - self.flag_pos[0])

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms.clear()
        self.coins.clear()

        # Start platform
        start_plat_width = self.np_random.integers(100, 150)
        start_plat = pygame.Rect(50, self.screen_height - 50, start_plat_width, 20)
        self.platforms.append(start_plat)

        # Procedurally generate subsequent platforms
        current_x = start_plat.right
        current_y = start_plat.top
        num_platforms = 12

        for i in range(num_platforms):
            if current_x > self.screen_width - 100:
                break
            
            gap_x = self.np_random.integers(30, 90)
            gap_y = self.np_random.integers(-60, 60)
            plat_width = self.np_random.integers(80, 160)
            
            next_x = current_x + gap_x
            next_y = np.clip(current_y + gap_y, 100, self.screen_height - 50)
            
            new_plat = pygame.Rect(next_x, next_y, plat_width, 20)
            self.platforms.append(new_plat)
            
            # Add a coin above the new platform
            coin_pos = (new_plat.centerx, new_plat.top - 30)
            self.coins.append({"pos": np.array(coin_pos, dtype=float), "radius": 8})
            
            current_x = new_plat.right
            current_y = new_plat.top

        # Place the flag on the last platform
        last_plat = self.platforms[-1]
        self.flag_pos = np.array([last_plat.centerx, last_plat.top - self.flag_size[1]])


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = self._update_player(movement, space_held)
        self._update_physics()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        self._update_particles()
        
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over: # Max steps reached
             # No specific penalty for timeout, but no win bonus
             pass

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement, space_held):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL
        
        # Friction
        if abs(self.player_vel[0]) > 0:
            self.player_vel[0] += self.PLAYER_FRICTION * np.sign(self.player_vel[0])
            if abs(self.player_vel[0]) < 0.2:
                self.player_vel[0] = 0
        
        self.player_vel[0] = np.clip(self.player_vel[0], -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)

        # Jumping
        if (movement == 1 or space_held) and self.on_ground:
            self.player_vel[1] = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            # Sound: Jump
            # Jump dust particles
            for _ in range(5):
                p_vel = np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(0.5, 1.5)])
                self.particles.append({
                    "pos": self.player_pos + np.array([0, self.player_size[1]/2]),
                    "vel": p_vel, "radius": self.np_random.uniform(2, 4),
                    "color": (150, 150, 150), "lifetime": 20
                })

        # Calculate movement reward
        current_dist_to_flag = abs(self.player_pos[0] - self.flag_pos[0])
        reward = 0
        if current_dist_to_flag < self.last_dist_to_flag:
            reward += 0.1
        elif current_dist_to_flag > self.last_dist_to_flag:
            reward -= 0.2
        self.last_dist_to_flag = current_dist_to_flag
        
        return reward

    def _update_physics(self):
        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        self.player_vel[1] = min(self.player_vel[1], self.MAX_FALL_SPEED)

        # Update position
        self.player_pos += self.player_vel

        # Screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_size[0]/2, self.screen_width - self.player_size[0]/2)
        
        # Squash and stretch animation
        if self.squash_timer > 0:
            self.squash_timer -= 1
            self.squash_stretch_factor = 1.0 + 0.3 * (self.squash_timer / 5.0)
        elif not self.on_ground:
            # Stretch while in air
            stretch = -self.player_vel[1] / self.PLAYER_JUMP_STRENGTH
            self.squash_stretch_factor = 1.0 + np.clip(stretch, 0, 0.3)
        else:
            self.squash_stretch_factor = 1.0

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(
            self.player_pos[0] - self.player_size[0]/2,
            self.player_pos[1] - self.player_size[1]/2,
            self.player_size[0], self.player_size[1]
        )

        # Platform collisions
        was_on_ground = self.on_ground
        self.on_ground = False
        for plat in self.platforms:
            if player_rect.colliderect(plat) and self.player_vel[1] >= 0:
                # Check if player was above the platform in the previous frame
                if (player_rect.bottom - self.player_vel[1]) <= plat.top + 1:
                    self.player_pos[1] = plat.top - self.player_size[1]/2
                    self.player_vel[1] = 0
                    self.on_ground = True
                    if not was_on_ground: # Just landed
                        self.squash_timer = 5
                        # Sound: Land
                    break
        
        # Coin collisions
        for coin in self.coins[:]:
            dist = np.linalg.norm(self.player_pos - coin["pos"])
            if dist < self.player_size[0]/2 + coin["radius"]:
                self.coins.remove(coin)
                self.score += 1
                reward += 1.0
                # Sound: Coin collect
                # Coin collection particles
                for _ in range(10):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 4)
                    p_vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
                    self.particles.append({
                        "pos": coin["pos"].copy(), "vel": p_vel,
                        "radius": self.np_random.uniform(3, 6),
                        "color": self.COLOR_PARTICLE, "lifetime": 25
                    })

        # Flag collision
        flag_rect = pygame.Rect(self.flag_pos[0] - self.flag_size[0]/2, self.flag_pos[1], *self.flag_size)
        if player_rect.colliderect(flag_rect):
            self.game_over = True
            self.score += 100
            reward += 100.0
            # Sound: Victory
        
        # Fall off screen
        if self.player_pos[1] > self.screen_height + self.player_size[1]:
            self.game_over = True
            reward -= 50.0
            # Sound: Fall/Failure

        return reward
        
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"][1] += 0.1 # particle gravity
            p["lifetime"] -= 1
            p["radius"] *= 0.97
            if p["lifetime"] <= 0 or p["radius"] < 0.5:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=3)

        # Draw coins with a subtle pulse
        pulse = math.sin(pygame.time.get_ticks() * 0.005) * 0.1 + 0.9
        for coin in self.coins:
            pos = (int(coin["pos"][0]), int(coin["pos"][1]))
            radius = int(coin["radius"] * pulse)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_COIN)
            
        # Draw flag
        flag_pole_rect = pygame.Rect(int(self.flag_pos[0] - 2), int(self.flag_pos[1]), 4, self.flag_size[1])
        pygame.draw.rect(self.screen, self.COLOR_FLAG, flag_pole_rect)
        flag_points = [
            (int(self.flag_pos[0]), int(self.flag_pos[1])),
            (int(self.flag_pos[0]), int(self.flag_pos[1] + 15)),
            (int(self.flag_pos[0] + 20), int(self.flag_pos[1] + 7.5))
        ]
        pygame.gfxdraw.aapolygon(self.screen, flag_points, self.COLOR_FLAG)
        pygame.gfxdraw.filled_polygon(self.screen, flag_points, self.COLOR_FLAG)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifetime"] / 20.0))))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), color)
        
        # Draw player with squash/stretch
        w = self.player_size[0] / self.squash_stretch_factor
        h = self.player_size[1] * self.squash_stretch_factor
        player_draw_rect = pygame.Rect(
            int(self.player_pos[0] - w / 2),
            int(self.player_pos[1] - h / 2 + self.player_size[1] * (1-self.squash_stretch_factor)/2),
            int(w), int(h)
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_draw_rect, border_radius=4)


    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos.tolist(),
            "player_vel": self.player_vel.tolist(),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for visualization
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display window
    pygame.display.set_caption("Pixel Platformer")
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    running = True
    total_reward = 0
    
    # Use a dictionary to track held keys for smoother controls
    keys_held = {
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_UP: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
        pygame.K_RSHIFT: False
    }

    while running:
        # Map pygame events to gymnasium action
        movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        space = 0    # 0=released, 1=held
        shift = 0    # 0=released, 1=held

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        if keys_held[pygame.K_UP]: movement = 1
        # Down (key 2) is a no-op
        if keys_held[pygame.K_LEFT]: movement = 3
        if keys_held[pygame.K_RIGHT]: movement = 4
        
        if keys_held[pygame.K_SPACE]: space = 1
        if keys_held[pygame.K_LSHIFT] or keys_held[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
        
        env.clock.tick(30) # Control the frame rate for human play

    env.close()