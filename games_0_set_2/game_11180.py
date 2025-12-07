import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:34:22.189885
# Source Brief: brief_01180.md
# Brief Index: 1180
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

# Helper classes for game entities to keep the main class clean
class Ball:
    def __init__(self, pos, radius, color, gravity_multiplier):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(0, 0)
        self.radius = radius
        self.color = color
        self.gravity_multiplier = gravity_multiplier
        self.squash_factor = 1.0
        self.on_ground_frames = 0

class Platform:
    def __init__(self, rect, center_y, amplitude, period, phase_offset):
        self.rect = pygame.Rect(rect)
        self.center_y = center_y
        self.amplitude = amplitude
        self.period = period
        self.phase_offset = phase_offset
        self.vel_y = 0

class Particle:
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
    
    game_description = "Juggle colored balls on moving platforms to score points by bouncing them before time runs out."
    user_guide = "Use ↑↓ to select a ball and ←→ to move it left and right. Each bounce scores a point."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (30, 60, 100)
    COLOR_BALLS = [(255, 80, 80), (80, 255, 80), (80, 120, 255)] # Red, Green, Blue
    COLOR_PLATFORM = (180, 180, 190)
    COLOR_PLATFORM_TOP = (220, 220, 230)
    COLOR_PARTICLE = (255, 255, 255)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_SELECTED_GLOW = (255, 255, 255)

    # Physics
    GRAVITY = 0.15
    BALL_ACCEL = 0.4
    FRICTION = 0.98
    ELASTICITY = 0.9

    # Game Rules
    WIN_SCORE = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_huge = pygame.font.SysFont("Consolas", 64, bold=True)


        # Initialize state variables
        self.balls = []
        self.platforms = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_ball_idx = 0
        self.last_ball_select_action = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_ball_idx = 0
        self.last_ball_select_action = 0 # 0 for none, 1 for up, 2 for down
        self.particles.clear()

        # Initialize Balls
        self.balls.clear()
        grav_multipliers = [1.0, 1.5, 2.0] # Mapped to Red, Green, Blue
        for i in range(3):
            self.balls.append(
                Ball(
                    pos=(self.WIDTH * (i + 1) / 4, self.HEIGHT / 4),
                    radius=15,
                    color=self.COLOR_BALLS[i],
                    gravity_multiplier=grav_multipliers[i],
                )
            )

        # Initialize Platforms
        self.platforms.clear()
        platform_configs = [
            # (x, y_center, width, height, amplitude, period)
            (50, 250, 120, 15, 60, 4.0),
            (260, 300, 120, 15, 80, 5.5),
            (470, 250, 120, 15, 60, 4.0),
        ]
        for x, y_center, w, h, amp, period in platform_configs:
            self.platforms.append(
                Platform(
                    rect=(x, y_center, w, h),
                    center_y=y_center,
                    amplitude=amp,
                    period=period,
                    phase_offset=self.np_random.uniform(0, 2 * math.pi),
                )
            )

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement_action = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        reward = 0
        terminated = False
        truncated = False

        # --- 1. Handle Actions ---
        # Ball selection (on press, not hold)
        if movement_action in [1, 2] and movement_action != self.last_ball_select_action:
            if movement_action == 1: # Up
                self.selected_ball_idx = (self.selected_ball_idx - 1) % len(self.balls)
            elif movement_action == 2: # Down
                self.selected_ball_idx = (self.selected_ball_idx + 1) % len(self.balls)
        self.last_ball_select_action = movement_action

        # Horizontal movement
        selected_ball = self.balls[self.selected_ball_idx]
        if movement_action == 3: # Left
            selected_ball.vel.x -= self.BALL_ACCEL
        elif movement_action == 4: # Right
            selected_ball.vel.x += self.BALL_ACCEL

        # --- 2. Update Game State ---
        self.steps += 1

        # Update platforms
        for p in self.platforms:
            time = (self.steps / self.FPS)
            prev_y = p.rect.y
            p.rect.y = p.center_y + p.amplitude * math.sin(2 * math.pi * time / p.period + p.phase_offset)
            p.vel_y = (p.rect.y - prev_y) * self.FPS

        # Update balls
        for ball in self.balls:
            # Apply gravity
            ball.vel.y += self.GRAVITY * ball.gravity_multiplier
            # Apply friction
            ball.vel.x *= self.FRICTION
            # Update position
            ball.pos += ball.vel
            # Update squash effect
            ball.squash_factor = min(1.0, ball.squash_factor + 0.08)

            # Collision with walls
            if ball.pos.x - ball.radius < 0:
                ball.pos.x = ball.radius
                ball.vel.x *= -self.ELASTICITY
            elif ball.pos.x + ball.radius > self.WIDTH:
                ball.pos.x = self.WIDTH - ball.radius
                ball.vel.x *= -self.ELASTICITY

            # Collision with floor/ceiling
            bounced = False
            if ball.pos.y + ball.radius > self.HEIGHT:
                ball.pos.y = self.HEIGHT - ball.radius
                ball.vel.y *= -self.ELASTICITY
                bounced = True
            elif ball.pos.y - ball.radius < 0:
                ball.pos.y = ball.radius
                ball.vel.y *= -self.ELASTICITY
                bounced = True
            
            if bounced:
                self.score += 1
                reward += 0.1
                ball.squash_factor = 0.7
                self._create_particles(ball.pos)
                # SFX: Wall bounce sound

            # Collision with platforms
            for p in self.platforms:
                # Check if ball is moving down and is within horizontal bounds of platform
                if ball.vel.y > 0 and p.rect.left < ball.pos.x < p.rect.right:
                    # Check for collision between bottom of ball and top of platform
                    if p.rect.top < ball.pos.y + ball.radius < p.rect.top + 20:
                        ball.pos.y = p.rect.top - ball.radius
                        # Bounce with platform's velocity contributing
                        ball.vel.y = -ball.vel.y * self.ELASTICITY + (p.vel_y / self.FPS) * 0.5
                        self.score += 1
                        reward += 0.1
                        ball.squash_factor = 0.7
                        self._create_particles(pygame.math.Vector2(ball.pos.x, p.rect.top))
                        # SFX: Platform bounce sound

        # Update particles
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.pos += p.vel
            p.lifetime -= 1
            p.radius = max(0, p.radius - 0.1)


        # --- 3. Check Termination ---
        if self.score >= self.WIN_SCORE:
            terminated = True
            self.game_over = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Game ends due to time limit
            truncated = True  # But it's a truncation, not a failure state
            self.game_over = True
            reward -= 100
        
        # --- 4. Return Values ---
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _create_particles(self, position):
        for _ in range(10):
            vel = pygame.math.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 0))
            self.particles.append(
                Particle(
                    pos=position,
                    vel=vel,
                    radius=self.np_random.uniform(2, 4),
                    color=self.COLOR_PARTICLE,
                    lifetime=20,
                )
            )

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Background gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Platforms
        for p in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p.rect, border_radius=4)
            top_rect = pygame.Rect(p.rect.x, p.rect.y, p.rect.width, 4)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_TOP, top_rect, border_radius=4)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p.lifetime / p.max_lifetime))
            try:
                # Use a temporary surface for alpha blending
                temp_surf = pygame.Surface((int(p.radius*2), int(p.radius*2)), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(
                    temp_surf, int(p.radius), int(p.radius), int(p.radius), (*p.color, alpha)
                )
                self.screen.blit(temp_surf, (int(p.pos.x - p.radius), int(p.pos.y - p.radius)))
            except (ValueError, pygame.error):
                # Skip drawing if radius is too small or negative
                pass


        # Balls
        for i, ball in enumerate(self.balls):
            pos_x, pos_y = int(ball.pos.x), int(ball.pos.y)
            radius_x = int(ball.radius / ball.squash_factor)
            radius_y = int(ball.radius * ball.squash_factor)

            # Selection indicator
            if i == self.selected_ball_idx:
                glow_radius = int(ball.radius * 1.4)
                # Draw multiple circles for a soft glow effect
                for r in range(glow_radius, int(ball.radius), -2):
                    alpha = 80 * (1 - (r - ball.radius) / (glow_radius - ball.radius))
                    pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, r, (*self.COLOR_SELECTED_GLOW, int(alpha)))

            # Ball ellipse for squash/stretch
            target_rect = pygame.Rect(pos_x - radius_x, pos_y - radius_y, 2 * radius_x, 2 * radius_y)
            pygame.draw.ellipse(self.screen, ball.color, target_rect)
            
            # Add a slight highlight
            highlight_color = tuple(min(255, c + 60) for c in ball.color)
            highlight_offset = ball.vel.normalize() * ball.radius * 0.3 if ball.vel.length() > 0 else pygame.math.Vector2(0, -1) * ball.radius * 0.3
            highlight_pos = (int(ball.pos.x - highlight_offset.x), int(ball.pos.y - highlight_offset.y))
            
            try:
                # Use a temporary surface for alpha blending
                temp_surf = pygame.Surface((int(ball.radius*0.6), int(ball.radius*0.6)), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, int(ball.radius*0.3), int(ball.radius*0.3), int(ball.radius * 0.3), (*highlight_color, 100))
                self.screen.blit(temp_surf, (highlight_pos[0] - int(ball.radius*0.3), highlight_pos[1] - int(ball.radius*0.3)))
            except (ValueError, pygame.error):
                pass


        # UI
        self._render_ui()

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        score_label_text = self.font_small.render("BOUNCES", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_label_text, (20, 45))

        # Timer
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        time_text = self.font_large.render(f"{time_left:.1f}", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)
        
        timer_label_text = self.font_small.render("TIME", True, self.COLOR_UI_TEXT)
        timer_label_rect = timer_label_text.get_rect(topright=(self.WIDTH - 20, 45))
        self.screen.blit(timer_label_text, timer_label_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                end_text = self.font_huge.render("YOU WIN!", True, self.COLOR_BALLS[1])
            else:
                end_text = self.font_huge.render("TIME UP!", True, self.COLOR_BALLS[0])
            
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)


    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually
    # The validation code in the original __init__ is removed for manual play
    class PlayableGameEnv(GameEnv):
        def __init__(self, render_mode="human_playable"):
            super().__init__(render_mode)
            # No validation call
    
    env = PlayableGameEnv(render_mode="human_playable")
    obs, info = env.reset()
    
    # Override screen for direct display
    env.screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Bounce Juggle")

    terminated = False
    truncated = False
    running = True
    total_reward = 0
    
    # Action state
    action = np.array([0, 0, 0]) # [movement, space, shift]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not (terminated or truncated):
            # --- Manual Control Mapping ---
            keys = pygame.key.get_pressed()
            
            # Map WASD/Arrows to MultiDiscrete action[0]
            mov_action = 0 # None
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                mov_action = 1 # Up
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                mov_action = 2 # Down
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                mov_action = 3 # Left
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                mov_action = 4 # Right
            
            action[0] = mov_action
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the game to the display window
        env._render_game()
        pygame.display.flip()

        # Check for reset
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0

        env.clock.tick(env.FPS)

    env.close()