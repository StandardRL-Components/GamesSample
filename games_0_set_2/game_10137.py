import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:53:11.506495
# Source Brief: brief_00137.md
# Brief Index: 137
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for effects."""
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = list(pos)
        self.vel = list(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1
        self.radius -= 0.1
        
    def draw(self, surface):
        if self.lifespan > 0 and self.radius > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            current_color = (*self.color, alpha)
            pygame.gfxdraw.filled_circle(
                surface,
                int(self.pos[0]),
                int(self.pos[1]),
                int(self.radius),
                current_color
            )

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball to hit
    numbered targets in descending order. Visual polish and game feel are prioritized.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Launch a bouncing ball to hit numbered targets in descending order. "
        "Aim carefully to create combos and achieve a high score before you run out of misses."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to adjust launch angle and ←→ to adjust power. "
        "Press space to launch the ball."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAY_AREA_INSET = 10
    
    COLOR_BG = (15, 20, 45)
    COLOR_WALL = (200, 200, 255)
    COLOR_BALL = (255, 80, 80)
    COLOR_BALL_GLOW = (255, 100, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_AIMER = (255, 255, 255, 150)
    COLOR_MISS = (255, 50, 50)
    
    TARGET_COLORS = [
        (255, 70, 70), (255, 140, 70), (255, 210, 70),
        (210, 255, 70), (140, 255, 70), (70, 255, 70),
        (70, 255, 140), (70, 255, 210), (70, 210, 255),
        (70, 140, 255)
    ]

    FONT_SIZE_S = 18
    FONT_SIZE_M = 24
    FONT_SIZE_L = 36
    
    BALL_RADIUS = 10
    BALL_START_POS = (SCREEN_WIDTH / 2, SCREEN_HEIGHT - 40)
    TARGET_RADIUS = 15
    NUM_TARGETS = 10
    
    MIN_LAUNCH_POWER = 5
    MAX_LAUNCH_POWER = 15
    POWER_INCREMENT = 0.2
    ANGLE_INCREMENT = 2.0  # degrees
    
    MAX_BOUNCES = 4
    MAX_STEPS = 1500
    WIN_SCORE = 100
    MAX_MISSES = 3
    
    REWARD_CORRECT_HIT = 1.0
    REWARD_WRONG_HIT = 0.1
    REWARD_COMBO = 5.0
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0

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
        
        self.font_s = pygame.font.SysFont("Consolas", self.FONT_SIZE_S, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", self.FONT_SIZE_M, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", self.FONT_SIZE_L, bold=True)
        
        # State variables will be initialized in reset()
        self.steps = None
        self.score = None
        self.misses = None
        self.game_over = None
        self.ball_state = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_bounces = None
        self.launch_angle = None
        self.launch_power = None
        self.targets = None
        self.current_target_value = None
        self.combo_count = None
        self.last_space_held = None
        self.particles = None
        self.starfield = None
        self.combo_fx_timer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        
        self.current_target_value = self.NUM_TARGETS
        self.combo_count = 0
        self.last_space_held = False
        self.particles = []
        self.combo_fx_timer = 0
        
        self._spawn_targets()
        self._reset_ball()
        self._generate_starfield()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0.0

        self._handle_input(action)
        
        if self.ball_state == 'launched':
            hit_event = self._update_ball_and_check_collisions()
            if hit_event:
                # A target was hit or the ball missed
                target_hit = hit_event['target']
                if target_hit:
                    # Sound: Target Hit
                    if target_hit['value'] == self.current_target_value:
                        # Correct Hit
                        step_reward += self.REWARD_CORRECT_HIT
                        self.score += 10
                        self.combo_count += 1
                        if self.combo_count >= 3:
                            step_reward += self.REWARD_COMBO
                            self.score += 5 * (self.combo_count - 2)
                            self.combo_fx_timer = 60 # 2 seconds at 30fps
                            # Sound: Combo Bonus
                        self.current_target_value -= 1
                    else:
                        # Wrong Hit (is a miss)
                        step_reward += self.REWARD_WRONG_HIT
                        self.misses += 1
                        self.combo_count = 0
                        # Sound: Miss
                    
                    target_hit['active'] = False
                    self._spawn_particles(target_hit['pos'], target_hit['color'], 50)
                else: # Ball missed by running out of bounces
                    self.misses += 1
                    self.combo_count = 0
                    # Sound: Miss
                
                self._reset_ball()

        self._update_particles()
        if self.combo_fx_timer > 0:
            self.combo_fx_timer -= 1
            
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                step_reward += self.REWARD_WIN
                # Sound: Win
            elif self.misses >= self.MAX_MISSES:
                step_reward = self.REWARD_LOSS # Overwrite other rewards on loss
                # Sound: Loss
            self.game_over = True

        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_press, _ = action[0], action[1] == 1, action[2] == 1

        if self.ball_state == 'ready':
            if movement == 1: # Up
                self.launch_angle = min(85, self.launch_angle + self.ANGLE_INCREMENT)
            elif movement == 2: # Down
                self.launch_angle = max(-85, self.launch_angle - self.ANGLE_INCREMENT)
            elif movement == 3: # Left
                self.launch_power = max(self.MIN_LAUNCH_POWER, self.launch_power - self.POWER_INCREMENT)
            elif movement == 4: # Right
                self.launch_power = min(self.MAX_LAUNCH_POWER, self.launch_power + self.POWER_INCREMENT)

            if space_press and not self.last_space_held:
                self.ball_state = 'launched'
                angle_rad = math.radians(self.launch_angle)
                self.ball_vel = [
                    self.launch_power * math.cos(angle_rad),
                    -self.launch_power * math.sin(angle_rad) # Pygame y is inverted
                ]
                # Sound: Ball Launch
        
        self.last_space_held = space_press

    def _update_ball_and_check_collisions(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # Wall collisions
        if self.ball_pos[0] <= self.PLAY_AREA_INSET + self.BALL_RADIUS:
            self.ball_pos[0] = self.PLAY_AREA_INSET + self.BALL_RADIUS
            self.ball_vel[0] *= -1
            self.ball_bounces += 1
            # Sound: Wall Bounce
        if self.ball_pos[0] >= self.SCREEN_WIDTH - self.PLAY_AREA_INSET - self.BALL_RADIUS:
            self.ball_pos[0] = self.SCREEN_WIDTH - self.PLAY_AREA_INSET - self.BALL_RADIUS
            self.ball_vel[0] *= -1
            self.ball_bounces += 1
            # Sound: Wall Bounce
        if self.ball_pos[1] <= self.PLAY_AREA_INSET + self.BALL_RADIUS:
            self.ball_pos[1] = self.PLAY_AREA_INSET + self.BALL_RADIUS
            self.ball_vel[1] *= -1
            self.ball_bounces += 1
            # Sound: Wall Bounce
        if self.ball_pos[1] >= self.SCREEN_HEIGHT - self.PLAY_AREA_INSET - self.BALL_RADIUS:
            self.ball_pos[1] = self.SCREEN_HEIGHT - self.PLAY_AREA_INSET - self.BALL_RADIUS
            self.ball_vel[1] *= -1
            self.ball_bounces += 1
            # Sound: Wall Bounce

        # Target collisions
        for target in self.targets:
            if target['active']:
                dist = math.hypot(self.ball_pos[0] - target['pos'][0], self.ball_pos[1] - target['pos'][1])
                if dist < self.BALL_RADIUS + self.TARGET_RADIUS:
                    return {'target': target}
        
        # Miss condition
        if self.ball_bounces > self.MAX_BOUNCES:
            return {'target': None}
        
        return None

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0 and p.radius > 0]
        for p in self.particles:
            p.update()

    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE or
            self.misses >= self.MAX_MISSES or
            self.steps >= self.MAX_STEPS or
            self.current_target_value <= 0
        )

    def _reset_ball(self):
        self.ball_state = 'ready'
        self.ball_pos = list(self.BALL_START_POS)
        self.ball_vel = [0, 0]
        self.ball_bounces = 0
        self.launch_angle = 0
        self.launch_power = (self.MIN_LAUNCH_POWER + self.MAX_LAUNCH_POWER) / 2
        
    def _spawn_targets(self):
        self.targets = []
        for i in range(1, self.NUM_TARGETS + 1):
            placed = False
            while not placed:
                pos = (
                    self.np_random.integers(self.TARGET_RADIUS + self.PLAY_AREA_INSET, self.SCREEN_WIDTH - self.TARGET_RADIUS - self.PLAY_AREA_INSET),
                    self.np_random.integers(self.TARGET_RADIUS + self.PLAY_AREA_INSET, self.SCREEN_HEIGHT - self.TARGET_RADIUS - 80) # Keep away from launch area
                )
                # Check for overlap with existing targets
                overlap = False
                for t in self.targets:
                    if math.hypot(pos[0] - t['pos'][0], pos[1] - t['pos'][1]) < self.TARGET_RADIUS * 2.5:
                        overlap = True
                        break
                if not overlap:
                    self.targets.append({
                        'pos': pos,
                        'value': i,
                        'radius': self.TARGET_RADIUS,
                        'color': self.TARGET_COLORS[i-1],
                        'active': True
                    })
                    placed = True
                    
    def _generate_starfield(self):
        self.starfield = []
        for _ in range(150):
            self.starfield.append(
                (
                    self.np_random.integers(0, self.SCREEN_WIDTH),
                    self.np_random.integers(0, self.SCREEN_HEIGHT),
                    self.np_random.integers(1, 3) # radius
                )
            )

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            radius = self.np_random.uniform(2, 6)
            lifespan = self.np_random.integers(20, 50)
            self.particles.append(Particle(pos, vel, radius, color, lifespan))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_walls()
        self._render_targets()
        self._render_particles()
        self._render_ball_and_aimer()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, r in self.starfield:
            brightness = self.np_random.integers(50, 150)
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, (brightness, brightness, brightness))
            
    def _render_walls(self):
        pygame.draw.rect(self.screen, self.COLOR_WALL, (
            self.PLAY_AREA_INSET, self.PLAY_AREA_INSET,
            self.SCREEN_WIDTH - 2 * self.PLAY_AREA_INSET,
            self.SCREEN_HEIGHT - 2 * self.PLAY_AREA_INSET
        ), 2)
        
    def _render_targets(self):
        for target in self.targets:
            if target['active']:
                pos = (int(target['pos'][0]), int(target['pos'][1]))
                is_current = target['value'] == self.current_target_value
                
                # Glow for current target
                if is_current:
                    glow_radius = int(target['radius'] * (1.4 + 0.2 * math.sin(self.steps * 0.1)))
                    glow_color = (*target['color'], 60)
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, glow_color)
                
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], target['radius'], target['color'])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], target['radius'], (255,255,255))
                
                text_surf = self.font_m.render(str(target['value']), True, (0,0,0))
                text_rect = text_surf.get_rect(center=pos)
                self.screen.blit(text_surf, text_rect)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_color):
        pos_int = (int(pos[0]), int(pos[1]))
        # Draw a few layers of glow
        for i in range(3):
            glow_rad = int(radius * (1.2 + i * 0.2))
            alpha = 80 // (i + 1)
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], glow_rad, (*glow_color, alpha))
        
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(radius), color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(radius), (255,255,255,180))

    def _render_ball_and_aimer(self):
        self._draw_glowing_circle(self.screen, self.ball_pos, self.BALL_RADIUS, self.COLOR_BALL, self.COLOR_BALL_GLOW)
        
        if self.ball_state == 'ready':
            angle_rad = math.radians(self.launch_angle)
            power_ratio = (self.launch_power - self.MIN_LAUNCH_POWER) / (self.MAX_LAUNCH_POWER - self.MIN_LAUNCH_POWER)
            end_x = self.ball_pos[0] + (30 + 40 * power_ratio) * math.cos(angle_rad)
            end_y = self.ball_pos[1] - (30 + 40 * power_ratio) * math.sin(angle_rad)
            
            # Interpolate color based on power
            color_r = int(100 + 155 * (1-power_ratio))
            color_g = int(100 + 155 * power_ratio)
            color_b = 100
            
            pygame.draw.line(self.screen, (color_r, color_g, color_b), 
                             (int(self.ball_pos[0]), int(self.ball_pos[1])), 
                             (int(end_x), int(end_y)), 2)

    def _render_ui(self):
        # Score
        score_surf = self.font_m.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 20))
        
        # Current Target
        if self.current_target_value > 0:
            target_surf = self.font_l.render(f"TARGET: {self.current_target_value}", True, self.COLOR_TEXT)
            target_rect = target_surf.get_rect(centerx=self.SCREEN_WIDTH / 2, y=20)
            self.screen.blit(target_surf, target_rect)
            
        # Misses
        miss_text_surf = self.font_m.render("MISSES:", True, self.COLOR_TEXT)
        miss_text_rect = miss_text_surf.get_rect(topright=(self.SCREEN_WIDTH - 120, 20))
        self.screen.blit(miss_text_surf, miss_text_rect)
        for i in range(self.misses):
            x_surf = self.font_l.render("X", True, self.COLOR_MISS)
            self.screen.blit(x_surf, (miss_text_rect.right + 10 + i * 30, 15))

        # Combo Effect
        if self.combo_fx_timer > 0:
            progress = self.combo_fx_timer / 60
            alpha = int(255 * math.sin(progress * math.pi)) # Fade in and out
            scale = 1.0 + 0.5 * (1 - progress) # Scale down
            
            combo_text = f"COMBO x{self.combo_count}!"
            font = pygame.font.SysFont("Consolas", int(self.FONT_SIZE_M * scale), bold=True)
            combo_surf = font.render(combo_text, True, self.TARGET_COLORS[self.combo_count % len(self.TARGET_COLORS)])
            combo_surf.set_alpha(alpha)
            
            combo_rect = combo_surf.get_rect(midleft=(190, 32))
            self.screen.blit(combo_surf, combo_rect)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            
            text_surf = self.font_l.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
            "current_target": self.current_target_value,
            "combo": self.combo_count,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Manual play example requires a display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bouncing Ball Target Game")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        # Event handling to quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False # To allow restart
                continue # Skip the rest of the loop to start fresh

        if terminated:
            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose for pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        clock.tick(30)
        
    print(f"Game Over. Final Info: {info}, Total Reward: {total_reward:.2f}")
    env.close()
    pygame.quit()