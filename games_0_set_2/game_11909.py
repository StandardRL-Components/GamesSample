import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:47:19.530525
# Source Brief: brief_01909.md
# Brief Index: 1909
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch a ball to hit targets within a circular arena. Use mid-air control to guide the ball and build momentum for higher scores."
    )
    user_guide = (
        "Use ←→ arrow keys to aim and press space to launch. Use ↑↓←→ arrow keys for mid-air control to guide the ball into the targets."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    CENTER = np.array([WIDTH / 2, HEIGHT / 2])
    ARENA_RADIUS = 180
    FPS = 30 # For time calculations, though step is discrete

    MAX_STEPS = 1000
    WIN_SCORE = 15
    MAX_MISS_STREAK = 3

    BALL_RADIUS = 12
    TARGET_RADIUS = 15
    
    GRAVITY = 0.1
    IMPULSE_STRENGTH = 0.25
    LAUNCH_SPEED = 7.0
    MIN_VEL_FOR_MISS = 0.2
    MOMENTUM_INCREASE = 1.1
    AIM_ROTATION_SPEED = 0.08
    
    # --- COLORS ---
    COLOR_BG = (10, 15, 25)
    COLOR_BOUNDARY = (200, 200, 220)
    COLOR_BALL_NORMAL = (0, 255, 120)
    COLOR_BALL_MOMENTUM = (50, 150, 255)
    COLOR_TARGET = (255, 215, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_AIMER = (255, 255, 255)
    COLOR_FLASH = (255, 50, 50)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_large = pygame.font.SysFont('Consolas', 32)
            self.font_small = pygame.font.SysFont('Consolas', 20)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 28)

        # Initialize non-state attributes
        self.ball_pos = np.zeros(2, dtype=float)
        self.ball_vel = np.zeros(2, dtype=float)
        self.ball_launched = False
        self.aim_angle = 0.0
        self.targets = []
        self.momentum = 1.0
        self.miss_streak = 0
        self.time_remaining = 0
        self.prev_space_held = False
        self.particles = []
        self.ball_trail = deque(maxlen=15)
        self.flash_effect_timer = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_events = []
        
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.ball_pos = self.CENTER + np.array([0, self.ARENA_RADIUS - self.BALL_RADIUS - 5], dtype=float)
        self.ball_vel = np.array([0.0, 0.0], dtype=float)
        self.ball_launched = False
        self.aim_angle = -math.pi / 2  # Straight up
        
        self.momentum = 1.0
        self.miss_streak = 0
        self.time_remaining = self.MAX_STEPS
        
        self.prev_space_held = False
        self.particles = []
        self.ball_trail.clear()
        self.flash_effect_timer = 0
        self.reward_events = []

        self.targets = []
        self._spawn_target()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        self.reward_events = []

        prev_dist_to_target = self._get_dist_to_target(self.ball_pos)

        self._handle_input(action)
        self._update_physics()
        self._check_collisions_and_state()
        self._update_effects()

        new_dist_to_target = self._get_dist_to_target(self.ball_pos)
        
        # Calculate rewards
        reward = 0
        # Shaping reward for getting closer to the target
        if self.ball_launched and prev_dist_to_target is not None and new_dist_to_target is not None:
             reward += (prev_dist_to_target - new_dist_to_target) * 0.01

        for r in self.reward_events:
            reward += r
        
        # Check for termination
        terminated = self._check_termination()
        truncated = False # No truncation condition other than termination
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Win reward
            else:
                reward -= 10.0 # Timeout penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held_int, _ = action
        space_held = space_held_int == 1

        if not self.ball_launched:
            # Aiming phase
            if movement == 3: # Left
                self.aim_angle -= self.AIM_ROTATION_SPEED
            elif movement == 4: # Right
                self.aim_angle += self.AIM_ROTATION_SPEED
            self.aim_angle = max(-math.pi + 0.1, min(-0.1, self.aim_angle))

            # Launch
            if space_held and not self.prev_space_held:
                self.ball_launched = True
                launch_vec = np.array([math.cos(self.aim_angle), math.sin(self.aim_angle)])
                self.ball_vel = launch_vec * self.LAUNCH_SPEED
        else:
            # Mid-air control
            impulse = np.zeros(2)
            if movement == 1: impulse[1] -= 1 # Up
            if movement == 2: impulse[1] += 1 # Down
            if movement == 3: impulse[0] -= 1 # Left
            if movement == 4: impulse[0] += 1 # Right
            
            if np.linalg.norm(impulse) > 0:
                impulse = impulse / np.linalg.norm(impulse)
                self.ball_vel += impulse * self.IMPULSE_STRENGTH

        self.prev_space_held = space_held

    def _update_physics(self):
        if not self.ball_launched:
            self.ball_vel = np.zeros(2)
            return

        # Apply gravity
        self.ball_vel[1] += self.GRAVITY
        
        # Update position with momentum
        self.ball_pos += self.ball_vel * self.momentum
        
        # Boundary collision
        dist_from_center = np.linalg.norm(self.ball_pos - self.CENTER)
        if dist_from_center > self.ARENA_RADIUS - self.BALL_RADIUS:
            normal = (self.ball_pos - self.CENTER) / dist_from_center
            self.ball_vel = self.ball_vel - 2 * np.dot(self.ball_vel, normal) * normal
            
            # Clamp position to be inside the arena to prevent getting stuck
            self.ball_pos = self.CENTER + normal * (self.ARENA_RADIUS - self.BALL_RADIUS)
            
            # Dampen velocity slightly on bounce
            self.ball_vel *= 0.98

    def _check_collisions_and_state(self):
        # Ball-target collision
        if self.targets:
            target_pos = self.targets[0]
            dist_to_target = np.linalg.norm(self.ball_pos - target_pos)
            if dist_to_target < self.BALL_RADIUS + self.TARGET_RADIUS:
                self.score += 1
                self.reward_events.append(1.0)
                self.miss_streak = 0
                self.momentum = min(3.0, self.momentum * self.MOMENTUM_INCREASE) # Cap momentum
                self._create_particles(target_pos, self.COLOR_TARGET, 20)
                self.targets.pop(0)
                if self.score < self.WIN_SCORE:
                    self._spawn_target()

        # Miss condition
        if self.ball_launched and np.linalg.norm(self.ball_vel * self.momentum) < self.MIN_VEL_FOR_MISS:
            self.reward_events.append(-0.5)
            self.miss_streak += 1
            if self.miss_streak >= self.MAX_MISS_STREAK:
                self.momentum = 1.0
                self.miss_streak = 0
                self.flash_effect_timer = 10 # frames
            
            # Reset ball for next launch
            self.ball_launched = False
            self.ball_pos = self.CENTER + np.array([0, self.ARENA_RADIUS - self.BALL_RADIUS - 5])
            self.ball_vel = np.zeros(2)
            
            # Respawn target
            if self.targets: self.targets.pop(0)
            self._spawn_target()

    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE
            or self.time_remaining <= 0
            or self.steps >= self.MAX_STEPS
        )

    def _spawn_target(self):
        while True:
            angle = self.np_random.uniform(0, 2 * math.pi)
            # Spawn away from the edges and center for better gameplay
            radius = self.np_random.uniform(self.TARGET_RADIUS * 2, self.ARENA_RADIUS - self.TARGET_RADIUS * 2)
            pos = self.CENTER + np.array([math.cos(angle), math.sin(angle)]) * radius
            
            # Ensure it's not too close to the player's starting position
            if np.linalg.norm(pos - (self.CENTER + np.array([0, self.ARENA_RADIUS]))) > 50:
                self.targets.append(pos)
                break

    def _update_effects(self):
        # Update ball trail
        if self.ball_launched:
            self.ball_trail.append(self.ball_pos.copy())
        else:
            self.ball_trail.clear()
            
        # Update particles
        new_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles
        
        # Update flash effect
        if self.flash_effect_timer > 0:
            self.flash_effect_timer -= 1

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': color
            })
            
    def _get_dist_to_target(self, pos):
        if not self.targets:
            return None
        return np.linalg.norm(pos - self.targets[0])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Arena Boundary (antialiased)
        pygame.gfxdraw.aacircle(self.screen, int(self.CENTER[0]), int(self.CENTER[1]), self.ARENA_RADIUS, self.COLOR_BOUNDARY)
        
        # Targets
        for target_pos in self.targets:
            # Flashing effect
            flash_alpha = 128 + 127 * math.sin(self.steps * 0.2)
            color = self.COLOR_TARGET
            
            # Glow
            for i in range(5, 0, -1):
                glow_color = (*color, int(flash_alpha / (i*1.5)))
                temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, int(target_pos[0]), int(target_pos[1]), self.TARGET_RADIUS + i*2, glow_color)
                self.screen.blit(temp_surf, (0,0))

            pygame.gfxdraw.filled_circle(self.screen, int(target_pos[0]), int(target_pos[1]), self.TARGET_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, int(target_pos[0]), int(target_pos[1]), self.TARGET_RADIUS, color)

        # Ball Trail
        if len(self.ball_trail) > 1:
            for i, pos in enumerate(self.ball_trail):
                alpha = int(255 * (i / len(self.ball_trail)))
                color = (*self.COLOR_BALL_MOMENTUM, alpha * 0.5)
                radius = int(self.BALL_RADIUS * (i / len(self.ball_trail)))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, color)

        # Ball
        ball_color = self.COLOR_BALL_MOMENTUM if self.momentum > 1.5 else self.COLOR_BALL_NORMAL
        
        # Ball Glow
        for i in range(8, 0, -1):
            glow_color = (*ball_color, int(50 / i))
            temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS + i, glow_color)
            self.screen.blit(temp_surf, (0,0))
            
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, ball_color)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, ball_color)
        
        # Aimer
        if not self.ball_launched:
            aim_len = 40
            end_pos = self.ball_pos + np.array([math.cos(self.aim_angle), math.sin(self.aim_angle)]) * aim_len
            pygame.draw.line(self.screen, self.COLOR_AIMER, self.ball_pos, end_pos, 2)

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 20.0))
            color = (*p['color'], alpha)
            size = int(max(1, 5 * (p['life'] / 20.0)))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, p['pos'] - size)


        # Flash Effect
        if self.flash_effect_timer > 0:
            alpha = 100 * (self.flash_effect_timer / 10)
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_FLASH, alpha))
            self.screen.blit(flash_surface, (0,0))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_text = self.font_large.render(f"{self.time_remaining}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)

        # Momentum & Miss Streak
        momentum_str = f"Momentum: {self.momentum:.1f}x"
        miss_str = f"Misses: {'★' * self.miss_streak}{'☆' * (self.MAX_MISS_STREAK - self.miss_streak)}"
        
        momentum_text = self.font_small.render(momentum_str, True, self.COLOR_TEXT)
        self.screen.blit(momentum_text, (20, self.HEIGHT - 30))

        miss_text = self.font_small.render(miss_str, True, self.COLOR_TEXT)
        miss_rect = miss_text.get_rect(topright=(self.WIDTH - 20, self.HEIGHT - 30))
        self.screen.blit(miss_text, miss_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "momentum": self.momentum,
            "miss_streak": self.miss_streak
        }

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run in a headless environment
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Momentum Ball")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    # --- MANUAL CONTROL MAPPING ---
    # ARROWS: move
    # SPACE: launch
    
    while not terminated:
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

        if terminated:
            print(f"Game Over!")
            print(f"Final Score: {info['score']}")
            print(f"Total Reward: {total_reward:.2f}")
            # Wait for a moment before closing
            pygame.time.wait(2000)
            
    env.close()