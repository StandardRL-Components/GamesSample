import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:19:49.346536
# Source Brief: brief_00358.md
# Brief Index: 358
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    game_description = (
        "Control a set of glowing balls, bouncing them off targets to score points before time runs out or you lose all your balls."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to apply force to all active balls. Hit targets to score points."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1800 # 60 seconds at 30 FPS
        self.WIN_SCORE = 1500
        self.NUM_TARGETS = 5
        self.NUM_BALLS = 3
        
        # Physics constants
        self.BALL_RADIUS = 12
        self.TARGET_RADIUS = 15
        self.ACCELERATION = 0.3
        self.FRICTION = 0.99
        self.BOUNCE_DAMPING = 0.95
        
        # Color Palette (Neon)
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_BORDER = (200, 220, 255)
        self.BALL_COLORS = [(255, 50, 50), (50, 255, 50), (50, 150, 255)]
        self.COLOR_TARGET = (50, 200, 255)
        self.COLOR_UI = (255, 255, 255)
        
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
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Initialize state variables to ensure they exist before reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls = []
        self.targets = []
        self.particles = []
        
        # Initialize state
        # self.reset() # reset is called by the wrapper/runner
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_lost = 0

        # Initialize balls
        self.balls = []
        for i in range(self.NUM_BALLS):
            self.balls.append({
                'pos': pygame.Vector2(
                    self.np_random.uniform(self.WIDTH * 0.4, self.WIDTH * 0.6),
                    self.np_random.uniform(self.HEIGHT * 0.4, self.HEIGHT * 0.6)
                ),
                'vel': pygame.Vector2(
                    self.np_random.uniform(-1, 1),
                    self.np_random.uniform(-1, 1)
                ).normalize() * 2,
                'color': self.BALL_COLORS[i],
                'radius': self.BALL_RADIUS,
                'active': True
            })

        # Initialize targets
        self.targets = [self._spawn_target() for _ in range(self.NUM_TARGETS)]
        
        # Clear particles
        self.particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        # 1. Apply player action
        self._apply_player_action(action)
        
        # 2. Update game state
        target_hit_reward = self._update_balls()
        reward += target_hit_reward
        self._update_particles()
        
        # 3. Check for termination
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        # 4. Apply terminal rewards
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100  # Win bonus
            elif self.balls_lost >= self.NUM_BALLS:
                reward -= 100 # Lose penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _apply_player_action(self, action):
        movement = action[0]
        accel_vec = pygame.Vector2(0, 0)
        
        if movement == 1: accel_vec.y = -self.ACCELERATION  # Up
        elif movement == 2: accel_vec.y = self.ACCELERATION   # Down
        elif movement == 3: accel_vec.x = -self.ACCELERATION  # Left
        elif movement == 4: accel_vec.x = self.ACCELERATION   # Right

        for ball in self.balls:
            if ball['active']:
                ball['vel'] += accel_vec

    def _update_balls(self):
        total_reward = 0
        speed_multiplier = 1.0 + (self.score / 100.0) * 0.01

        for ball in self.balls:
            if not ball['active']:
                continue

            # Apply friction and update position
            ball['vel'] *= self.FRICTION
            ball['pos'] += ball['vel'] * speed_multiplier

            # Wall collision
            if ball['pos'].x - ball['radius'] < 0 or ball['pos'].x + ball['radius'] > self.WIDTH:
                ball['active'] = False
                self.balls_lost += 1
                self._create_particles(ball['pos'], (200, 200, 200), 50)
                continue
            if ball['pos'].y - ball['radius'] < 0 or ball['pos'].y + ball['radius'] > self.HEIGHT:
                ball['active'] = False
                self.balls_lost += 1
                self._create_particles(ball['pos'], (200, 200, 200), 50)
                continue

            # Target collision
            for i, target_pos in enumerate(self.targets):
                dist = ball['pos'].distance_to(target_pos)
                if dist < ball['radius'] + self.TARGET_RADIUS:
                    self.score += 10
                    total_reward += 10
                    
                    # Bounce effect
                    normal = (ball['pos'] - target_pos).normalize()
                    ball['vel'] = ball['vel'].reflect(normal) * self.BOUNCE_DAMPING
                    
                    # Move ball slightly out of target to prevent sticking
                    ball['pos'] += normal * (ball['radius'] + self.TARGET_RADIUS - dist)
                    
                    self._create_particles(target_pos, self.COLOR_TARGET, 30)
                    self.targets[i] = self._spawn_target()
        
        return total_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= 0.1

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.game_over = True
        elif self.balls_lost >= self.NUM_BALLS:
            self.game_over = True
        return self.game_over

    def _spawn_target(self):
        padding = self.TARGET_RADIUS + 10
        return pygame.Vector2(
            self.np_random.uniform(padding, self.WIDTH - padding),
            self.np_random.uniform(padding, self.HEIGHT - padding)
        )
        
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(2, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_remaining": self.NUM_BALLS - self.balls_lost,
        }

    def _render_game(self):
        # Draw border
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Draw particles
        for p in self.particles:
            if p['radius'] > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'].x), int(p['pos'].y),
                    int(p['radius']),
                    (*p['color'], int(255 * (p['lifespan'] / 30.0)))
                )
        
        # Draw targets
        for pos in self.targets:
            self._draw_glow_circle(self.screen, self.COLOR_TARGET, pos, self.TARGET_RADIUS, 8)

        # Draw balls
        for ball in self.balls:
            if ball['active']:
                self._draw_glow_circle(self.screen, ball['color'], ball['pos'], ball['radius'], 15)

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        seconds_left = time_left / self.metadata['render_fps']
        timer_text = self.font.render(f"TIME: {seconds_left:.1f}", True, self.COLOR_UI)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        # Remaining balls
        for i in range(self.NUM_BALLS):
            color = self.BALL_COLORS[i] if i >= self.balls_lost else (50, 50, 50)
            center_x = self.WIDTH // 2 - (self.NUM_BALLS - 1) * 15 + i * 30
            pygame.gfxdraw.filled_circle(self.screen, center_x, 25, 8, color)
            pygame.gfxdraw.aacircle(self.screen, center_x, 25, 8, self.COLOR_BORDER)


    def _draw_glow_circle(self, surface, color, center, radius, max_glow):
        center_int = (int(center.x), int(center.y))
        
        # Draw concentric circles for glow effect
        for i in range(max_glow, 0, -1):
            alpha = int(100 * (1 - (i / max_glow))**2)
            glow_radius = radius + i
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], glow_radius, glow_color)
        
        # Draw main circle
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], radius, color)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the headless environment
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit() # Quit the dummy instance
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bouncing Balls")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement_action = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.metadata['render_fps'])
        
    env.close()