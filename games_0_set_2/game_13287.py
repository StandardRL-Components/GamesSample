import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# --- Helper Classes for Game Entities ---

class Particle:
    """A single particle for visual effects."""
    def __init__(self, pos, color, vel, lifespan):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            size = int(3 * (self.lifespan / self.max_lifespan))
            if size > 0:
                rect = pygame.Rect(int(self.pos[0] - size/2), int(self.pos[1] - size/2), size, size)
                # Create a temporary surface for transparency
                temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                temp_surf.fill((self.color[0], self.color[1], self.color[2], alpha))
                surface.blit(temp_surf, rect.topleft)

class Ball:
    """Represents a bouncing ball."""
    def __init__(self, pos, vel, color, radius):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.radius = radius
        self.trail = []

    def update(self, gravity):
        self.vel[1] += gravity
        self.pos += self.vel

        # Add current position to trail and cap its length
        self.trail.append(tuple(self.pos))
        if len(self.trail) > 10:
            self.trail.pop(0)

    def draw(self, surface):
        # Draw trail for motion blur effect
        for i, pos in enumerate(self.trail):
            alpha = int(100 * (i / len(self.trail)))
            radius = int(self.radius * (i / len(self.trail)))
            if radius > 1:
                pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), radius, (*self.color, alpha))

        # Draw glow effect
        glow_radius = int(self.radius * 1.5)
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), glow_radius, (*self.color, 50))
        
        # Draw main ball
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, self.color)


class Platform:
    """Represents a player-controlled platform."""
    def __init__(self, center_x, y, width, height, color):
        self.base_center_x = center_x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.angle = 0.0
        self.target_angle = 0.0
        self.current_center_x = center_x
        self.vel_x = 0.0

    def update(self, oscillation_phase, oscillation_speed, tilt_lerp_rate):
        # Update horizontal oscillation
        amplitude = 50
        old_x = self.current_center_x
        self.current_center_x = self.base_center_x + amplitude * math.sin(oscillation_phase)
        self.vel_x = (self.current_center_x - old_x) * 0.5 # Momentum transfer factor

        # Smoothly interpolate angle towards target
        self.angle += (self.target_angle - self.angle) * tilt_lerp_rate

    def get_rotated_surface_and_rect(self):
        # Create a surface for the platform
        platform_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        platform_surf.fill(self.color)
        
        # Rotate the surface
        rotated_surf = pygame.transform.rotate(platform_surf, self.angle)
        
        # Get the new rect and center it
        rotated_rect = rotated_surf.get_rect(center=(self.current_center_x, self.y))
        
        return rotated_surf, rotated_rect

    def draw(self, surface):
        rotated_surf, rotated_rect = self.get_rotated_surface_and_rect()
        surface.blit(rotated_surf, rotated_rect)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    game_description = (
        "Keep the balls in the air by tilting the platforms. The longer you last, the higher your score!"
    )
    user_guide = (
        "Controls: Use the ← and → arrow keys to tilt the platforms and keep the balls from falling."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 3600  # 60 seconds at 60 FPS

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (20, 40, 80)
    COLOR_PLATFORM = (180, 180, 190)
    COLOR_LINE = (255, 200, 0, 150)
    COLOR_TEXT = (240, 240, 240)
    BALL_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
    ]

    # Physics
    GRAVITY = 0.15
    BALL_RADIUS = 10
    PLATFORM_WIDTH, PLATFORM_HEIGHT = 120, 15
    PLATFORM_Y = 350
    FAIL_HEIGHT_PIXELS = 380 # Fail if ball center is below this y-coordinate
    MAX_TILT_ANGLE = 35  # degrees
    TILT_LERP_RATE = 0.2
    BOUNCE_ELASTICITY = 0.85
    MIN_BOUNCE_VEL = -8.0
    MOMENTUM_TRANSFER = 0.3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        self.balls = []
        self.platforms = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.oscillation_speed = 0.0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.oscillation_speed = 0.1 # rad/s equivalent

        self.platforms = [
            Platform(self.WIDTH * 0.25, self.PLATFORM_Y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT, self.COLOR_PLATFORM),
            Platform(self.WIDTH * 0.75, self.PLATFORM_Y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT, self.COLOR_PLATFORM)
        ]

        self.balls = []
        
        # Stable starting positions for balls to pass stability test
        ball_xs = [130, 160, 190, 450, 510]
        
        for i in range(5):
            x = ball_xs[i]
            y = self.HEIGHT * 0.2 + self.np_random.uniform(-20, 20)
            self.balls.append(Ball(
                pos=[x, y],
                # Reduce initial horizontal velocity for stability
                vel=[self.np_random.uniform(-0.1, 0.1), self.np_random.uniform(-1, 1)],
                color=self.BALL_COLORS[i],
                radius=self.BALL_RADIUS
            ))
        
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        
        self._handle_input(action)
        self._update_game_state()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        self.game_over = terminated or truncated
        
        if truncated and not terminated:
            # Victory bonus for surviving
            self.score += 100.0
            reward += 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        
        target_angle = 0.0
        if movement == 3: # Left
            target_angle = self.MAX_TILT_ANGLE
        elif movement == 4: # Right
            target_angle = -self.MAX_TILT_ANGLE
        
        for p in self.platforms:
            p.target_angle = target_angle

    def _update_game_state(self):
        # Update difficulty
        if self.steps > 0 and self.steps % 600 == 0:
            self.oscillation_speed += 0.001

        # Update platforms
        oscillation_phase = self.steps * self.oscillation_speed
        for p in self.platforms:
            p.update(oscillation_phase, self.oscillation_speed, self.TILT_LERP_RATE)

        # Update balls and handle collisions
        for ball in self.balls:
            ball.update(self.GRAVITY)
            self._handle_ball_collisions(ball)

        # Update and prune particles
        for particle in self.particles:
            particle.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]

    def _handle_ball_collisions(self, ball):
        # Wall collisions
        if ball.pos[0] - ball.radius < 0:
            ball.pos[0] = ball.radius
            ball.vel[0] *= -1
        elif ball.pos[0] + ball.radius > self.WIDTH:
            ball.pos[0] = self.WIDTH - ball.radius
            ball.vel[0] *= -1

        # Ceiling collision
        if ball.pos[1] - ball.radius < 0:
            ball.pos[1] = ball.radius
            ball.vel[1] *= -1

        # Platform collisions
        for p in self.platforms:
            _, p_rect = p.get_rotated_surface_and_rect()
            if ball.vel[1] > 0 and p_rect.colliderect(ball.pos[0]-ball.radius, ball.pos[1]-ball.radius, ball.radius*2, ball.radius*2):
                
                # Simplified collision check: is ball center above platform rect?
                if ball.pos[1] < p.y:
                    # Calculate surface normal of the platform
                    angle_rad = math.radians(p.angle)
                    normal = np.array([-math.sin(angle_rad), -math.cos(angle_rad)])

                    # Reflect velocity
                    v_in = ball.vel
                    v_out = v_in - (1 + self.BOUNCE_ELASTICITY) * np.dot(v_in, normal) * normal
                    ball.vel = v_out
                    
                    # Add momentum from platform
                    ball.vel[0] += p.vel_x * self.MOMENTUM_TRANSFER

                    # Apply minimum upward velocity for better game feel
                    if ball.vel[1] > self.MIN_BOUNCE_VEL:
                        ball.vel[1] = self.MIN_BOUNCE_VEL

                    # Move ball just outside the platform to prevent sticking
                    ball.pos[1] = p.y - p_rect.height/2 - ball.radius - 1
                    
                    # Create particles on bounce
                    self._create_particles(ball.pos, ball.color)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            self.particles.append(Particle(pos, color, vel, lifespan))

    def _calculate_reward(self):
        balls_aloft = sum(1 for ball in self.balls if ball.pos[1] < self.PLATFORM_Y)
        return 0.01 * balls_aloft
    
    def _check_termination(self):
        for ball in self.balls:
            if ball.pos[1] > self.FAIL_HEIGHT_PIXELS:
                return True
        return False
        
    def _get_observation(self):
        self._render_to_surface()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        # human render mode is handled in the main loop
        return None

    def _render_to_surface(self):
        self._render_background()
        self._render_game()
        self._render_ui()

    def _render_background(self):
        # Draw a gradient background
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw height line
        line_surf = pygame.Surface((self.WIDTH, 2), pygame.SRCALPHA)
        line_surf.fill(self.COLOR_LINE)
        self.screen.blit(line_surf, (0, self.FAIL_HEIGHT_PIXELS - 1))

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Draw platforms
        for p in self.platforms:
            p.draw(self.screen)
            
        # Draw balls
        for b in self.balls:
            b.draw(self.screen)

    def _render_ui(self):
        # Balls aloft count
        balls_aloft = sum(1 for ball in self.balls if ball.pos[1] < self.FAIL_HEIGHT_PIXELS)
        aloft_text = self.font_small.render(f"BALLS ALOFT: {balls_aloft}", True, self.COLOR_TEXT)
        self.screen.blit(aloft_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"{time_left:.2f}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            victory = self.steps >= self.MAX_STEPS
            msg = "VICTORY!" if victory else "GAME OVER"
            color = (100, 255, 100) if victory else (255, 100, 100)
            
            over_text = self.font_large.render(msg, True, color)
            over_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(over_text, over_rect)

            score_text = self.font_small.render(f"Final Score: {self.score:.1f}", True, self.COLOR_TEXT)
            score_rect = score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_aloft": sum(1 for ball in self.balls if ball.pos[1] < self.FAIL_HEIGHT_PIXELS)
        }
        
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Juggling Game")
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # space/shift not used
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        env._render_to_surface()
        surf = pygame.transform.flip(env.screen, False, False)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}")
    
    # Keep the window open for a few seconds to see the final screen
    pygame.time.wait(3000)
    
    env.close()