import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:22:31.105554
# Source Brief: brief_01657.md
# Brief Index: 1657
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a swinging pendulum to collect gems.

    **Gameplay:**
    - The player controls the angular acceleration of a pendulum.
    - The goal is to collect 50 gems within 60 seconds.
    - Collecting a gem increases the pendulum's length, allowing it to reach farther gems.
    - Colliding with the side walls decreases the pendulum's length and penalizes the agent.

    **Action Space (MultiDiscrete([5, 2, 2])):**
    - `action[0]`: Movement (3=left push, 4=right push, others=no push)
    - `action[1]`: Space button (unused)
    - `action[2]`: Shift button (unused)

    **Observation Space:**
    - A 640x400 RGB image of the game screen.

    **Rewards:**
    - +100 for winning (collecting 50 gems).
    - -10 for losing (timeout).
    - +1 for each gem collected.
    - -5 for hitting a wall.
    - +0.1 for moving closer to the nearest gem (shaping reward).
    - -0.1 for moving farther from the nearest gem (shaping reward).
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control a swinging pendulum to collect gems. "
        "Collecting gems extends your reach, but hitting walls will shorten it."
    )
    user_guide = (
        "Controls: Use the ← and → arrow keys to push the pendulum and swing towards the gems."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    GEMS_TO_WIN = 50

    # --- Colors ---
    COLOR_BG = (15, 20, 30)
    COLOR_PENDULUM_BOB = (255, 60, 60)
    COLOR_PENDULUM_BOB_GLOW = (255, 60, 60, 50)
    COLOR_PENDULUM_ROD = (200, 200, 220)
    COLOR_GEM = (50, 255, 100)
    COLOR_GEM_GLOW = (50, 255, 100, 60)
    COLOR_WALL = (100, 100, 110)
    COLOR_SPARK = (255, 220, 50)
    COLOR_ARC = (40, 50, 60)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (0, 0, 0)

    # --- Physics ---
    GRAVITY = 0.005
    DAMPING = 0.999
    PLAYER_FORCE = 0.0015
    MIN_LENGTH = 50
    MAX_LENGTH = 350
    LENGTH_GAIN_PER_GEM = 6
    LENGTH_LOSS_PER_WALL_HIT = 40
    INITIAL_LENGTH = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # --- Game State Initialization ---
        self.pivot = None
        self.length = None
        self.angle = None
        self.angular_velocity = None
        self.gems = None
        self.particles = None
        self.score = None
        self.steps = None
        self.game_over = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pivot = (self.SCREEN_WIDTH // 2, 50)
        self.length = self.INITIAL_LENGTH
        self.angle = math.pi / 2  # Start pointing straight down
        self.angular_velocity = 0.0
        
        self.gems = self._spawn_gems()
        self.particles = []
        
        self.score = 0
        self.steps = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        
        # --- Pre-update state for reward shaping ---
        bob_pos_before = self._get_bob_position()
        dist_to_nearest_gem_before = self._get_dist_to_nearest_gem(bob_pos_before)

        # --- Update Game Logic ---
        self._update_pendulum(movement)
        
        bob_pos_after = self._get_bob_position()
        
        # --- Handle Collisions ---
        wall_hit = self._handle_wall_collisions(bob_pos_after)
        gem_collected = self._handle_gem_collisions(bob_pos_after)

        # --- Update Particles ---
        self._update_particles()
        
        # --- Calculate Reward ---
        dist_to_nearest_gem_after = self._get_dist_to_nearest_gem(bob_pos_after)
        
        reward = self._calculate_reward(
            gem_collected, wall_hit, dist_to_nearest_gem_before, dist_to_nearest_gem_after
        )
        
        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = False # This env does not truncate
        if terminated:
            self.game_over = True
            if self.score >= self.GEMS_TO_WIN:
                reward += 100  # Victory bonus
            else:
                reward += -10  # Timeout penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _spawn_gems(self):
        gems = []
        # Spawn 10 easy gems
        for _ in range(10):
            r = self.np_random.uniform(self.MIN_LENGTH, self.INITIAL_LENGTH + 20)
            a = self.np_random.uniform(math.pi / 4, 3 * math.pi / 4)
            x = self.pivot[0] + r * math.cos(a)
            y = self.pivot[1] + r * math.sin(a)
            gems.append([x, y])

        # Spawn remaining gems
        for _ in range(self.GEMS_TO_WIN - 10):
            r = self.np_random.uniform(self.MIN_LENGTH, self.MAX_LENGTH - 20)
            a = self.np_random.uniform(0, math.pi) # Spawn in a semicircle below pivot
            x = self.pivot[0] + r * math.cos(a)
            y = self.pivot[1] + r * math.sin(a)
            # Ensure gems are not inside walls
            x = np.clip(x, 10, self.SCREEN_WIDTH - 10)
            y = np.clip(y, self.pivot[1], self.SCREEN_HEIGHT - 10)
            gems.append([x, y])
        return gems

    def _update_pendulum(self, movement):
        # Calculate gravitational acceleration
        # The formula for tangential acceleration is -g * sin(theta_from_vertical)
        # Since our angle is from horizontal-right, theta_from_vertical = self.angle - PI/2
        # sin(self.angle - PI/2) = -cos(self.angle)
        # So, acceleration is g * cos(self.angle).
        # We divide by length to model a simple pendulum: a = (g/L) * cos(angle)
        gravitational_accel = (self.GRAVITY / (self.length / 100)) * math.cos(self.angle)

        # Apply player force
        player_accel = 0
        if movement == 3:  # Left
            player_accel = -self.PLAYER_FORCE
        elif movement == 4:  # Right
            player_accel = self.PLAYER_FORCE
        
        # Update angular velocity and angle
        self.angular_velocity += gravitational_accel + player_accel
        self.angular_velocity *= self.DAMPING
        self.angle += self.angular_velocity

    def _handle_wall_collisions(self, bob_pos):
        hit_wall = False
        bob_x, bob_y = bob_pos
        if bob_x < 10 or bob_x > self.SCREEN_WIDTH - 10:
            # # Sound: wall_thud.wav
            self.length = max(self.MIN_LENGTH, self.length - self.LENGTH_LOSS_PER_WALL_HIT)
            self.angular_velocity *= -0.5  # Reverse and dampen velocity
            self._create_sparks((bob_x, bob_y))
            hit_wall = True
        return hit_wall

    def _handle_gem_collisions(self, bob_pos):
        collected_gem = False
        bob_x, bob_y = bob_pos
        for gem in self.gems[:]:
            dist = math.hypot(bob_x - gem[0], bob_y - gem[1])
            if dist < 15: # Bob radius (10) + Gem radius (5)
                # # Sound: gem_collect.wav
                self.gems.remove(gem)
                self.score += 1
                self.length = min(self.MAX_LENGTH, self.length + self.LENGTH_GAIN_PER_GEM)
                collected_gem = True
                break # Only collect one gem per frame
        return collected_gem

    def _calculate_reward(self, gem_collected, wall_hit, dist_before, dist_after):
        reward = 0
        if gem_collected:
            reward += 1.0
        if wall_hit:
            reward -= 5.0
        
        # Shaping reward for moving towards the nearest gem
        if dist_before is not None and dist_after is not None:
            if dist_after < dist_before:
                reward += 0.1
            else:
                reward -= 0.1
        
        return reward

    def _check_termination(self):
        return self.steps >= self.MAX_STEPS or self.score >= self.GEMS_TO_WIN

    def _get_bob_position(self):
        x = self.pivot[0] + self.length * math.cos(self.angle)
        y = self.pivot[1] + self.length * math.sin(self.angle)
        return x, y

    def _get_dist_to_nearest_gem(self, pos):
        if not self.gems:
            return None
        
        return min(math.hypot(pos[0] - gem[0], pos[1] - gem[1]) for gem in self.gems)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": (self.MAX_STEPS - self.steps) / self.FPS}

    def _render_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_game(self):
        # --- Walls ---
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 10)

        # --- Swing Arc ---
        arc_rect = (self.pivot[0] - self.length, self.pivot[1] - self.length, self.length * 2, self.length * 2)
        try:
            pygame.draw.arc(self.screen, self.COLOR_ARC, arc_rect, 0, math.pi, 2)
        except (ValueError, TypeError): # Can happen if length is negative/zero
            pass

        # --- Gems ---
        for gx, gy in self.gems:
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, int(gx), int(gy), 10, self.COLOR_GEM_GLOW)
            # Gem body
            pygame.gfxdraw.filled_circle(self.screen, int(gx), int(gy), 6, self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, int(gx), int(gy), 6, self.COLOR_GEM)

        # --- Pendulum Rod ---
        bob_pos = self._get_bob_position()
        pygame.draw.aaline(self.screen, self.COLOR_PENDULUM_ROD, self.pivot, bob_pos, 1)

        # --- Pendulum Bob ---
        bob_x, bob_y = int(bob_pos[0]), int(bob_pos[1])
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, bob_x, bob_y, 15, self.COLOR_PENDULUM_BOB_GLOW)
        # Bob body
        pygame.gfxdraw.filled_circle(self.screen, bob_x, bob_y, 10, self.COLOR_PENDULUM_BOB)
        pygame.gfxdraw.aacircle(self.screen, bob_x, bob_y, 10, self.COLOR_PENDULUM_BOB)

        # --- Particles ---
        self._render_particles()

    def _render_ui(self):
        # --- Score Display ---
        score_text = f"GEMS: {self.score}/{self.GEMS_TO_WIN}"
        self._render_text(score_text, self.font_small, self.COLOR_TEXT, (20, 20))

        # --- Timer Display ---
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        self._render_text(timer_text, self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - 150, 20))

        # --- Game Over Message ---
        if self.game_over:
            if self.score >= self.GEMS_TO_WIN:
                end_text = "VICTORY!"
                color = self.COLOR_GEM
            else:
                end_text = "TIME UP!"
                color = self.COLOR_PENDULUM_BOB
            
            text_surf = self.font_large.render(end_text, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            
            shadow_surf = self.font_large.render(end_text, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surf.get_rect(center=(self.SCREEN_WIDTH/2 + 3, self.SCREEN_HEIGHT/2 + 3))

            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)

    def _create_sparks(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append([list(pos), vel, lifespan, self.COLOR_SPARK])

    def _update_particles(self):
        for p in self.particles[:]:
            p[0][0] += p[1][0]  # Update x
            p[0][1] += p[1][1]  # Update y
            p[2] -= 1           # Decrease lifespan
            if p[2] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            pos, _, lifespan, color = p
            size = max(0, int(lifespan / 6))
            pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), size)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This part of the script is for human interaction and will not run in the test environment.
    # It requires a display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a separate display screen for manual play
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pendulum Swing")
    
    total_reward = 0
    
    while not done:
        # --- Action Mapping for Manual Play ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        # The action space is MultiDiscrete, but we only control the first part here
        action = [movement, 0, 0]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Rendering for Manual Play ---
        # The observation is a numpy array, convert it back to a Pygame surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        env.clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()