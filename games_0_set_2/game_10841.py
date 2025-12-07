import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:04:32.622309
# Source Brief: brief_00841.md
# Brief Index: 841
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
        "Nudge colored balls into targets to score points. Switch between balls and manage their momentum to clear the board before time runs out."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to nudge the active ball. Press Shift to cycle between balls."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.WIN_SCORE = 500
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_TARGET = (255, 200, 0)
        self.BALL_COLORS = [(255, 50, 50), (50, 255, 50), (80, 80, 255)]
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_PARTICLE_HIT = (255, 220, 100)
        self.COLOR_PARTICLE_COLLIDE = (200, 200, 220)

        # Physics constants
        self.BALL_RADIUS = 15
        self.TARGET_RADIUS = 20
        self.NUDGE_STRENGTH = 0.4
        self.FRICTION = 0.995
        self.MAX_SPEED = 7.0
        self.BALL_COLLISION_KICK = 2.0

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
        try:
            self.font_large = pygame.font.SysFont("Consolas", 30)
            self.font_small = pygame.font.SysFont("Consolas", 20)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 28)

        # State variables (will be initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.balls = []
        self.targets = []
        self.particles = []
        self.active_ball_index = 0
        self.shift_was_held = False
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed) # For python's random module
            # self.np_random = np.random.default_rng(seed) # If using numpy's RNG

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.active_ball_index = 0
        self.shift_was_held = False
        self.particles.clear()

        self._spawn_balls()
        self._spawn_targets(count=3)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _spawn_balls(self):
        self.balls.clear()
        spawn_positions = []
        for i in range(3):
            while True:
                pos = np.array([
                    random.uniform(self.BALL_RADIUS * 2, self.WIDTH - self.BALL_RADIUS * 2),
                    random.uniform(self.BALL_RADIUS * 2, self.HEIGHT - self.BALL_RADIUS * 2)
                ])
                # Ensure no overlap at spawn
                if all(np.linalg.norm(pos - p) > self.BALL_RADIUS * 2.5 for p in spawn_positions):
                    spawn_positions.append(pos)
                    break
            
            vel = np.array([random.uniform(-2, 2), random.uniform(-2, 2)])
            self.balls.append({'pos': pos, 'vel': vel, 'color': self.BALL_COLORS[i]})

    def _spawn_targets(self, count=1):
        for _ in range(count):
            while True:
                pos = np.array([
                    random.uniform(self.TARGET_RADIUS, self.WIDTH - self.TARGET_RADIUS),
                    random.uniform(self.TARGET_RADIUS, self.HEIGHT - self.TARGET_RADIUS)
                ])
                # Ensure it doesn't spawn on a ball
                if all(np.linalg.norm(pos - b['pos']) > self.BALL_RADIUS + self.TARGET_RADIUS + 10 for b in self.balls):
                    self.targets.append({'pos': pos})
                    break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0

        self._handle_input(action)
        reward += self._update_physics_and_collisions()
        self._update_particles()

        self.steps += 1
        self.time_remaining -= 1

        terminated = self._check_termination()
        truncated = False
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0  # Win bonus
            else:
                reward -= 10.0 # Time out penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, _, shift_held_action = action
        shift_held = shift_held_action == 1

        # Cycle active ball on SHIFT press (not hold)
        if shift_held and not self.shift_was_held:
            self.active_ball_index = (self.active_ball_index + 1) % len(self.balls)
        self.shift_was_held = shift_held

        # Apply nudge to the active ball
        active_ball = self.balls[self.active_ball_index]
        if movement == 1:  # Up
            active_ball['vel'][1] -= self.NUDGE_STRENGTH
        elif movement == 2:  # Down
            active_ball['vel'][1] += self.NUDGE_STRENGTH
        elif movement == 3:  # Left
            active_ball['vel'][0] -= self.NUDGE_STRENGTH
        elif movement == 4:  # Right
            active_ball['vel'][0] += self.NUDGE_STRENGTH

    def _update_physics_and_collisions(self):
        step_reward = 0.0

        # Update ball positions and handle wall collisions
        for ball in self.balls:
            ball['vel'] *= self.FRICTION
            speed = np.linalg.norm(ball['vel'])
            if speed > self.MAX_SPEED:
                ball['vel'] = ball['vel'] * (self.MAX_SPEED / speed)
            ball['pos'] += ball['vel']

            if ball['pos'][0] < self.BALL_RADIUS:
                ball['pos'][0] = self.BALL_RADIUS
                ball['vel'][0] *= -1
            elif ball['pos'][0] > self.WIDTH - self.BALL_RADIUS:
                ball['pos'][0] = self.WIDTH - self.BALL_RADIUS
                ball['vel'][0] *= -1

            if ball['pos'][1] < self.BALL_RADIUS:
                ball['pos'][1] = self.BALL_RADIUS
                ball['vel'][1] *= -1
            elif ball['pos'][1] > self.HEIGHT - self.BALL_RADIUS:
                ball['pos'][1] = self.HEIGHT - self.BALL_RADIUS
                ball['vel'][1] *= -1

        # Handle ball-target collisions
        hit_targets_indices = []
        for i, target in enumerate(self.targets):
            for ball in self.balls:
                if np.linalg.norm(ball['pos'] - target['pos']) < self.BALL_RADIUS + self.TARGET_RADIUS:
                    if i not in hit_targets_indices:
                        self.score += 10
                        step_reward += 10.0 # Reward for hitting a target
                        hit_targets_indices.append(i)
                        self._create_particle_burst(target['pos'], self.COLOR_PARTICLE_HIT, 20)
        
        if hit_targets_indices:
            self.targets = [t for i, t in enumerate(self.targets) if i not in hit_targets_indices]
            self._spawn_targets(count=len(hit_targets_indices))

        # Handle ball-ball collisions
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                ball1 = self.balls[i]
                ball2 = self.balls[j]
                dist_vec = ball1['pos'] - ball2['pos']
                dist = np.linalg.norm(dist_vec)

                if dist < self.BALL_RADIUS * 2:
                    self.score = max(0, self.score - 5)
                    step_reward -= 5.0 # Penalty for collision
                    
                    # Resolve overlap
                    overlap = self.BALL_RADIUS * 2 - dist
                    correction = (overlap / (dist + 1e-6)) * dist_vec
                    ball1['pos'] += correction / 2
                    ball2['pos'] -= correction / 2
                    
                    # Elastic collision response (simplified)
                    midpoint = (ball1['pos'] + ball2['pos']) / 2
                    self._create_particle_burst(midpoint, self.COLOR_PARTICLE_COLLIDE, 10)
                    
                    v1 = ball1['vel']
                    ball1['vel'] = ball2['vel']
                    ball2['vel'] = v1
        
        return step_reward

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] *= 0.97
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _create_particle_burst(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': random.randint(20, 40),
                'color': color,
                'radius': random.uniform(2, 5)
            })

    def _check_termination(self):
        return self.time_remaining <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render targets
        for target in self.targets:
            pos_int = (int(target['pos'][0]), int(target['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.TARGET_RADIUS, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.TARGET_RADIUS, self.COLOR_TARGET)

        # Render particles
        for p in self.particles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            radius_int = max(1, int(p['radius']))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius_int, p['color'])

        # Render balls
        for i, ball in enumerate(self.balls):
            pos_int = (int(ball['pos'][0]), int(ball['pos'][1]))
            color = ball['color']
            
            # Glow effect
            glow_color = (*color, 60) # RGBA with alpha
            glow_surf = pygame.Surface((self.BALL_RADIUS * 4, self.BALL_RADIUS * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (self.BALL_RADIUS * 2, self.BALL_RADIUS * 2), self.BALL_RADIUS * 1.5)
            self.screen.blit(glow_surf, (pos_int[0] - self.BALL_RADIUS * 2, pos_int[1] - self.BALL_RADIUS * 2))
            
            # Main ball
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, color)

            # Active ball indicator
            if i == self.active_ball_index:
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS + 4, (255, 255, 255))
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS + 5, (255, 255, 255))


    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer display
        time_sec = self.time_remaining / self.FPS
        timer_text = self.font_large.render(f"TIME: {time_sec:.1f}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
            else:
                msg = "TIME'S UP!"
            
            end_text = self.font_large.render(msg, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "active_ball": self.active_ball_index
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Ball Juggler")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    # Mapping keyboard keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while running:
        # Player control
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            for key, move_action in key_to_action.items():
                if keys[key]:
                    movement = move_action
                    break
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_held = 1
            
            # Step the environment
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()