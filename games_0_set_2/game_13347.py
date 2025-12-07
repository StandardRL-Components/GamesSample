import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:53:31.034482
# Source Brief: brief_03347.md
# Brief Index: 3347
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Tilt the board to guide your ball into the moving targets. "
        "Chain hits to build your score multiplier, but be careful not to miss!"
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to tilt the surface and guide the ball."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1800 # 60 seconds at 30 FPS
    WIN_SCORE = 1000
    MAX_MISSES = 3

    # Colors (Neon Inspired)
    COLOR_BG = (15, 15, 25)
    COLOR_BALL = (0, 191, 255) # Deep Sky Blue
    COLOR_BALL_GLOW = (0, 100, 150)
    COLOR_TARGET = (255, 140, 0) # Dark Orange
    COLOR_TARGET_GLOW = (150, 80, 0)
    COLOR_PARTICLE = (255, 255, 0) # Yellow
    COLOR_BOUNDARY = (200, 200, 255, 100)
    COLOR_SCORE = (50, 205, 50) # Lime Green
    COLOR_MULTIPLIER = (186, 85, 211) # Medium Orchid
    COLOR_TIMER = (220, 20, 60) # Crimson

    # Physics
    BALL_RADIUS = 12
    TARGET_RADIUS = 18
    TILT_INCREMENT = 0.005 # Radians, approx 0.3 degrees
    MAX_TILT = 0.1 # Radians, approx 5.7 degrees
    GRAVITY_SCALE = 0.4
    BALL_FRICTION = 0.998
    NUM_TARGETS = 5
    INITIAL_TARGET_SPEED = 1.5
    TARGET_SPEED_INCREASE_INTERVAL = 500
    TARGET_SPEED_INCREASE_AMOUNT = 0.2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_multiplier = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ball_pos = np.array([0.0, 0.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.targets = []
        self.particles = []
        self.tilt = np.array([0.0, 0.0]) # [x_tilt, y_tilt]
        self.chain_multiplier = 1
        self.consecutive_misses = 0
        self.target_speed = self.INITIAL_TARGET_SPEED
        
        # self.reset() # reset is called by the wrapper/runner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Reset ball to center with a small random velocity
        self.ball_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        self.ball_vel = self.np_random.uniform(low=-1.5, high=1.5, size=2)

        # Reset tilt and multiplier
        self.tilt = np.array([0.0, 0.0])
        self.chain_multiplier = 1
        self.consecutive_misses = 0
        self.target_speed = self.INITIAL_TARGET_SPEED

        # Clear lists
        self.particles.clear()
        self.targets.clear()

        # Spawn initial targets
        for _ in range(self.NUM_TARGETS):
            self._spawn_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- Action Handling ---
        movement = action[0]
        self._handle_actions(movement)

        # --- Game Logic ---
        self._update_physics()
        hit_info = self._handle_collisions()

        if hit_info["target_hit"]:
            # --- Event: Target Hit ---
            self.score += 10 * self.chain_multiplier
            self.chain_multiplier += 1
            self.consecutive_misses = 0
            self._create_particles(hit_info["pos"], 30)
            self._spawn_target()
            
            # Rewards
            reward += 0.1  # Continuous feedback
            reward += 1.0  # Multiplier increase bonus

        elif hit_info["wall_hit"]:
            # --- Event: Wall Hit (Miss) ---
            if self.chain_multiplier > 1:
                self.chain_multiplier = 1
                self.consecutive_misses = 1
            else:
                self.consecutive_misses += 1
        
        self._update_particles()
        
        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % self.TARGET_SPEED_INCREASE_INTERVAL == 0:
            self.target_speed += self.TARGET_SPEED_INCREASE_AMOUNT

        # --- Termination Check ---
        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not truncated:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1:   # Up
            self.tilt[1] -= self.TILT_INCREMENT
        elif movement == 2: # Down
            self.tilt[1] += self.TILT_INCREMENT
        elif movement == 3: # Left
            self.tilt[0] -= self.TILT_INCREMENT
        elif movement == 4: # Right
            self.tilt[0] += self.TILT_INCREMENT
        
        # Clamp tilt to max values
        self.tilt = np.clip(self.tilt, -self.MAX_TILT, self.MAX_TILT)

    def _update_physics(self):
        # Apply gravity based on tilt
        gravity = self.tilt * self.GRAVITY_SCALE
        self.ball_vel += gravity
        
        # Apply friction
        self.ball_vel *= self.BALL_FRICTION
        
        # Update ball position
        self.ball_pos += self.ball_vel

        # Update target positions
        for target in self.targets:
            target['pos'] += target['vel']

    def _handle_collisions(self):
        hit_info = {"target_hit": False, "wall_hit": False, "pos": None}

        # Ball vs Walls
        if self.ball_pos[0] <= self.BALL_RADIUS:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= -0.9
            hit_info["wall_hit"] = True
        elif self.ball_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_pos[0] = self.SCREEN_WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -0.9
            hit_info["wall_hit"] = True
        
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= -0.9
            hit_info["wall_hit"] = True
        elif self.ball_pos[1] >= self.SCREEN_HEIGHT - self.BALL_RADIUS:
            self.ball_pos[1] = self.SCREEN_HEIGHT - self.BALL_RADIUS
            self.ball_vel[1] *= -0.9
            hit_info["wall_hit"] = True

        # Ball vs Targets
        for i in range(len(self.targets) - 1, -1, -1):
            target = self.targets[i]
            dist = np.linalg.norm(self.ball_pos - target['pos'])
            if dist < self.BALL_RADIUS + self.TARGET_RADIUS:
                hit_info["target_hit"] = True
                hit_info["pos"] = target['pos'].copy()
                del self.targets[i]
                # Simple momentum transfer
                self.ball_vel *= -0.95
                break # Handle one hit per frame

        # Targets vs Walls
        for target in self.targets:
            if target['pos'][0] <= self.TARGET_RADIUS or target['pos'][0] >= self.SCREEN_WIDTH - self.TARGET_RADIUS:
                target['vel'][0] *= -1
            if target['pos'][1] <= self.TARGET_RADIUS or target['pos'][1] >= self.SCREEN_HEIGHT - self.TARGET_RADIUS:
                target['vel'][1] *= -1
            target['pos'] = np.clip(target['pos'], self.TARGET_RADIUS, [self.SCREEN_WIDTH - self.TARGET_RADIUS, self.SCREEN_HEIGHT - self.TARGET_RADIUS])

        return hit_info

    def _spawn_target(self):
        # Ensure new targets don't spawn on top of the ball or other targets
        for _ in range(100): # Try 100 times to avoid infinite loop
            pos = self.np_random.uniform(
                low=[self.TARGET_RADIUS, self.TARGET_RADIUS],
                high=[self.SCREEN_WIDTH - self.TARGET_RADIUS, self.SCREEN_HEIGHT - self.TARGET_RADIUS],
                size=2
            )
            # Check distance from ball
            if np.linalg.norm(pos - self.ball_pos) < self.BALL_RADIUS + self.TARGET_RADIUS + 20:
                continue
            # Check distance from other targets
            if any(np.linalg.norm(pos - t['pos']) < 2 * self.TARGET_RADIUS for t in self.targets):
                continue
            
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)]) * self.target_speed
            self.targets.append({'pos': pos, 'vel': vel})
            return
        # If we failed to spawn, just place it randomly
        pos = self.np_random.uniform(
            low=[self.TARGET_RADIUS, self.TARGET_RADIUS],
            high=[self.SCREEN_WIDTH - self.TARGET_RADIUS, self.SCREEN_HEIGHT - self.TARGET_RADIUS],
            size=2
        )
        angle = self.np_random.uniform(0, 2 * math.pi)
        vel = np.array([math.cos(angle), math.sin(angle)]) * self.target_speed
        self.targets.append({'pos': pos, 'vel': vel})


    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True
        if self.consecutive_misses >= self.MAX_MISSES:
            return True
        # Failsafe for a stuck ball
        if self.steps > 100 and np.linalg.norm(self.ball_vel) < 0.1:
            return True
        return False

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifetime = self.np_random.integers(20, 40)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': lifetime})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Air resistance
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw boundaries
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 2)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 40))
            color = (*self.COLOR_PARTICLE, int(alpha))
            try:
                # Use a temporary surface for alpha blending
                temp_surf = pygame.Surface((int(p['life']/5), int(p['life']/5)), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (int(p['life']/10), int(p['life']/10)), int(p['life']/10))
                self.screen.blit(temp_surf, (p['pos'] - int(p['life']/10)).astype(int))
            except (ValueError, pygame.error):
                # Ignore errors from particles going off-screen or having zero size
                pass


        # Draw targets with glow
        for target in self.targets:
            pos_int = target['pos'].astype(int)
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.TARGET_RADIUS + 4, (*self.COLOR_TARGET_GLOW, 80))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.TARGET_RADIUS, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.TARGET_RADIUS, self.COLOR_TARGET)

        # Draw ball with glow
        pos_int = self.ball_pos.astype(int)
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS + 6, (*self.COLOR_BALL_GLOW, 100))
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 30.0 # Assuming 30fps for display
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TIMER)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH // 2 - timer_text.get_width() // 2, 10))

        # Multiplier
        if self.chain_multiplier > 1:
            multiplier_text = self.font_multiplier.render(f"x{self.chain_multiplier}", True, self.COLOR_MULTIPLIER)
            text_rect = multiplier_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 5))
            self.screen.blit(multiplier_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "chain_multiplier": self.chain_multiplier,
            "consecutive_misses": self.consecutive_misses,
            "ball_speed": np.linalg.norm(self.ball_vel)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # We need to unset the dummy video driver to see the display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tilt-a-Bounce")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Run at 30 FPS for manual play
        
    env.close()