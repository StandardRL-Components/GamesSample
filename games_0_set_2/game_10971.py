import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:24:23.085335
# Source Brief: brief_00971.md
# Brief Index: 971
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
        "Swing a pendulum and launch balls to hit all the targets. "
        "Time your release carefully to clear each level."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to swing the pendulum. Press space to launch a ball."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    COLOR_BG = (15, 15, 25)
    COLOR_PENDULUM_PIVOT = (200, 200, 200)
    COLOR_PENDULUM_ARM = (100, 180, 255)
    COLOR_PENDULUM_BOB = (50, 150, 255)
    COLOR_BALL = (255, 80, 80)
    COLOR_BALL_GLOW = (255, 80, 80, 60)
    COLOR_TARGET = (50, 255, 100)
    COLOR_TARGET_HIT = (255, 255, 50)
    COLOR_TEXT = (240, 240, 240)

    PENDULUM_PIVOT_POS = (SCREEN_WIDTH // 2, 60)
    PENDULUM_LENGTH = 120
    PENDULUM_GRAVITY = 0.15
    PENDULUM_DAMPING = 0.998
    
    BALL_RADIUS = 8
    BALL_GRAVITY = 0.2
    BALL_LAUNCH_SPEED_MULTIPLIER = 10.0
    
    TARGET_WIDTH = 50
    TARGET_HEIGHT = 15
    NUM_TARGETS = 10
    TARGETS_TO_WIN = 8
    
    MAX_BALLS = 30
    MAX_STEPS = 1500 # Increased from 1000 to allow more time for all balls

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_level = pygame.font.SysFont("Consolas", 28, bold=True)

        # Game state that persists across episodes (level progression)
        self.level = 1
        self.max_pendulum_vel = 1.0
        
        # Initialize state variables
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.terminated = False
        
        self.balls_left = self.MAX_BALLS
        self.targets_hit = 0
        
        self.launch_cooldown = 0
        self.pendulum_angle = math.pi # Straight down
        self.pendulum_vel = 0.0
        
        self.balls = []
        self.particles = []
        
        self._generate_targets()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.terminated:
            obs, info = self.reset()
            return obs, 0, self.terminated, False, info
            
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        step_reward = 0

        self._handle_input(movement, space_held)
        self._update_pendulum()
        ball_reward = self._update_balls()
        self._update_particles()
        
        step_reward += ball_reward
        self.score += step_reward
        self.steps += 1
        
        # --- Termination Logic ---
        win_condition = self.targets_hit >= self.TARGETS_TO_WIN
        lose_condition = self.balls_left <= 0 and not self.balls
        timeout = self.steps >= self.MAX_STEPS
        
        self.terminated = win_condition or lose_condition or timeout
        
        if self.terminated:
            if win_condition:
                step_reward += 50  # Goal-oriented reward for winning
                # SFX: Level Complete
                self.level += 1
                self.max_pendulum_vel += 0.2
            else: # Lost or timed out
                step_reward -= 50 # Goal-oriented penalty for losing
                # SFX: Level Fail
                self.level = 1
                self.max_pendulum_vel = 1.0

        return (
            self._get_observation(),
            step_reward,
            self.terminated,
            False,  # truncated always False
            self._get_info()
        )

    # --- Private Update Methods ---

    def _handle_input(self, movement, space_held):
        # Update pendulum velocity based on action
        if movement == 3: # Left
            self.pendulum_vel -= 0.015
        elif movement == 4: # Right
            self.pendulum_vel += 0.015
        
        self.pendulum_vel = np.clip(self.pendulum_vel, -self.max_pendulum_vel, self.max_pendulum_vel)

        # Handle launching a ball
        if self.launch_cooldown > 0:
            self.launch_cooldown -= 1
        
        if space_held and self.launch_cooldown == 0 and self.balls_left > 0:
            # SFX: Launch Ball
            self.balls_left -= 1
            self.launch_cooldown = 15 # 0.5 second cooldown at 30 FPS
            
            bob_pos = self._get_pendulum_bob_pos()
            
            # Velocity is tangential to the pendulum's swing
            launch_angle = self.pendulum_angle + math.pi / 2
            launch_speed = abs(self.pendulum_vel) * self.BALL_LAUNCH_SPEED_MULTIPLIER
            
            vel_x = launch_speed * math.cos(launch_angle)
            vel_y = launch_speed * math.sin(launch_angle)
            
            self.balls.append({'pos': list(bob_pos), 'vel': [vel_x, vel_y]})

    def _update_pendulum(self):
        # Physics: Simple Harmonic Motion with Damping
        angular_accel = (-self.PENDULUM_GRAVITY / self.PENDULUM_LENGTH) * math.sin(self.pendulum_angle - math.pi)
        self.pendulum_vel += angular_accel
        self.pendulum_vel *= self.PENDULUM_DAMPING
        self.pendulum_angle += self.pendulum_vel

    def _update_balls(self):
        step_reward = 0
        for ball in self.balls[:]:
            # Apply gravity
            ball['vel'][1] += self.BALL_GRAVITY
            # Update position
            ball['pos'][0] += ball['vel'][0]
            ball['pos'][1] += ball['vel'][1]
            
            ball_rect = pygame.Rect(ball['pos'][0] - self.BALL_RADIUS, ball['pos'][1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # Check for target collision
            hit_target = False
            for target in self.targets:
                if not target['hit'] and target['rect'].colliderect(ball_rect):
                    # SFX: Target Hit
                    target['hit'] = True
                    self.targets_hit += 1
                    step_reward += 5.0  # Event-based reward for hitting a target
                    self._create_particles(target['rect'].center, self.COLOR_TARGET_HIT)
                    self.balls.remove(ball)
                    hit_target = True
                    break
            if hit_target:
                continue

            # Check for out of bounds (and apply proximity reward)
            if not (0 < ball['pos'][0] < self.SCREEN_WIDTH and 0 < ball['pos'][1] < self.SCREEN_HEIGHT):
                # Find closest non-hit target for proximity reward
                min_dist = float('inf')
                closest_target = None
                for target in self.targets:
                    if not target['hit']:
                        dist = math.hypot(ball['pos'][0] - target['rect'].centerx, ball['pos'][1] - target['rect'].centery)
                        if dist < min_dist:
                            min_dist = dist
                            closest_target = target
                
                if closest_target:
                    # Reward is higher the closer the ball lands to a target center
                    # Max reward of +10 if it lands on the center, decaying over 200px
                    proximity_reward = max(0, 10.0 * (1 - min_dist / 200.0))
                    step_reward += proximity_reward
                
                self.balls.remove(ball)
        return step_reward
        
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    # --- Private Render Methods ---
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render targets
        for target in self.targets:
            color = self.COLOR_TARGET_HIT if target['hit'] else self.COLOR_TARGET
            pygame.draw.rect(self.screen, color, target['rect'], border_radius=3)
            
        # Render pendulum
        bob_pos = self._get_pendulum_bob_pos()
        pygame.draw.aaline(self.screen, self.COLOR_PENDULUM_ARM, self.PENDULUM_PIVOT_POS, bob_pos, 2)
        pygame.gfxdraw.filled_circle(self.screen, int(self.PENDULUM_PIVOT_POS[0]), int(self.PENDULUM_PIVOT_POS[1]), 5, self.COLOR_PENDULUM_PIVOT)
        pygame.gfxdraw.aacircle(self.screen, int(self.PENDULUM_PIVOT_POS[0]), int(self.PENDULUM_PIVOT_POS[1]), 5, self.COLOR_PENDULUM_PIVOT)
        pygame.gfxdraw.filled_circle(self.screen, int(bob_pos[0]), int(bob_pos[1]), 12, self.COLOR_PENDULUM_BOB)
        pygame.gfxdraw.aacircle(self.screen, int(bob_pos[0]), int(bob_pos[1]), 12, self.COLOR_PENDULUM_BOB)
        
        # Render balls
        for ball in self.balls:
            x, y = int(ball['pos'][0]), int(ball['pos'][1])
            # Glow effect
            glow_surf = pygame.Surface((self.BALL_RADIUS * 4, self.BALL_RADIUS * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW, (self.BALL_RADIUS * 2, self.BALL_RADIUS * 2), self.BALL_RADIUS * 2)
            self.screen.blit(glow_surf, (x - self.BALL_RADIUS * 2, y - self.BALL_RADIUS * 2))
            # Ball
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
            
        # Render particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifetime'] / p['start_life']))
            color = (*p['color'], int(alpha))
            # Simple rect particle for performance
            pygame.draw.rect(self.screen, color, (p['pos'][0], p['pos'][1], 2, 2))

    def _render_ui(self):
        # Balls left
        balls_text = self.font_ui.render(f"BALLS: {self.balls_left}/{self.MAX_BALLS}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (10, 10))
        
        # Targets hit
        targets_text = self.font_ui.render(f"HITS: {self.targets_hit}/{self.TARGETS_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(targets_text, (self.SCREEN_WIDTH - targets_text.get_width() - 10, 10))
        
        # Current level
        level_text = self.font_level.render(f"LEVEL {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH // 2 - level_text.get_width() // 2, self.SCREEN_HEIGHT - level_text.get_height() - 10))

    # --- Private Helper Methods ---

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level, "targets_hit": self.targets_hit}

    def _get_pendulum_bob_pos(self):
        x = self.PENDULUM_PIVOT_POS[0] + self.PENDULUM_LENGTH * math.sin(self.pendulum_angle)
        y = self.PENDULUM_PIVOT_POS[1] + self.PENDULUM_LENGTH * math.cos(self.pendulum_angle)
        return x, y

    def _generate_targets(self):
        self.targets = []
        y_positions = [self.SCREEN_HEIGHT - 40, self.SCREEN_HEIGHT - 70, self.SCREEN_HEIGHT - 100]
        
        for i in range(self.NUM_TARGETS):
            placed = False
            while not placed:
                y = random.choice(y_positions)
                x = self.np_random.integers(20, self.SCREEN_WIDTH - self.TARGET_WIDTH - 20)
                new_rect = pygame.Rect(x, y, self.TARGET_WIDTH, self.TARGET_HEIGHT)
                
                # Ensure no overlap
                if not any(new_rect.colliderect(t['rect'].inflate(10,10)) for t in self.targets):
                    self.targets.append({'rect': new_rect, 'hit': False})
                    placed = True
    
    def _create_particles(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos), 
                'vel': vel, 
                'lifetime': lifetime, 
                'start_life': lifetime,
                'color': color
            })
            
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pendulum Launch")
    
    terminated = False
    running = True
    
    while running:
        # --- Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r' key
                    print("--- Resetting Environment ---")
                    obs, info = env.reset()
                    terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

            if terminated:
                print(f"--- Episode Finished ---")
                print(f"Final Score: {info['score']:.2f}, Targets Hit: {info['targets_hit']}, Level: {info['level']}")
                print("Press 'R' to play again or close the window.")

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.metadata["render_fps"])

    pygame.quit()