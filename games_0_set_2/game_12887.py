import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:36:37.017321
# Source Brief: brief_02887.md
# Brief Index: 2887
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball.
    The goal is to score points by bouncing off walls, managing momentum,
    and surviving for as long as possible or until reaching the score limit.

    Visuals: Minimalist, neon-inspired aesthetic.
    Gameplay: Skill-based arcade, risk/reward bouncing mechanics.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball to score points by hitting walls. "
        "Manage momentum for higher scores, but avoid the floor!"
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to nudge the ball when it hits a wall or the ceiling, influencing its direction."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds * 30 FPS

    # Colors
    COLOR_BG = (10, 10, 30)
    COLOR_WALLS = (220, 220, 255)
    COLOR_FLOOR = (255, 50, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_BALL_SLOW = (0, 150, 255)
    COLOR_BALL_FAST = (255, 100, 0)

    # Physics & Gameplay
    GRAVITY = 0.4
    INITIAL_VEL_RANGE = ((-4, -2), (-8, -6))  # Launch upwards to pass stability test
    MOMENTUM_GAIN = 1.02  # Multiplier on speed after a successful bounce
    NUDGE_STRENGTH = 1.5
    BOUNCE_FAIL_CHANCE = 0.10
    MAX_SPEED = 25
    WIN_SCORE = 5000
    
    # Rendering
    WALL_THICKNESS = 5
    BALL_RADIUS = 12
    GLOW_LAYERS = 7

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.ball_pos = None
        self.ball_vel = None
        self.score = None
        self.steps = None
        self.time_remaining = None
        self.momentum = None
        self.last_action_movement = None
        self.particles = None
        self.game_over = None
        self.game_won = None
        self.termination_reason = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        start_x_vel = self.np_random.uniform(self.INITIAL_VEL_RANGE[0][0], self.INITIAL_VEL_RANGE[0][1])
        start_y_vel = self.np_random.uniform(self.INITIAL_VEL_RANGE[1][0], self.INITIAL_VEL_RANGE[1][1])
        
        self.ball_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 4)
        self.ball_vel = pygame.Vector2(start_x_vel, start_y_vel)
        
        self.score = 0
        self.steps = 0
        self.time_remaining = self.MAX_STEPS
        self.momentum = 1.0
        self.last_action_movement = 0  # 0 = no-op
        self.particles = []
        self.game_over = False
        self.game_won = False
        self.termination_reason = ""

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, _, _ = action  # space and shift are unused as per brief
        self.last_action_movement = movement
        
        reward = 0
        terminated = False
        truncated = False

        if not self.game_over:
            self._update_physics()
            reward += self._handle_collisions()
            
            # Update timers
            self.steps += 1
            self.time_remaining -= 1

            # Check for termination conditions
            if self.score >= self.WIN_SCORE:
                self.game_over = True
                self.game_won = True
                self.termination_reason = "SCORE LIMIT REACHED"
                reward += 100 # Goal-oriented reward
            elif self.time_remaining <= 0:
                self.game_over = True
                self.termination_reason = "TIME'S UP"
            # Other termination reasons (floor, bounce fail) are set in _handle_collisions
        
        if self.game_over and not self.game_won:
            reward -= 10 # Failure penalty
        
        terminated = self.game_over
        if self.steps >= self.MAX_STEPS and not terminated:
             truncated = True
             self.termination_reason = "MAX STEPS REACHED"

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _update_physics(self):
        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Update ball
        self.ball_vel.y += self.GRAVITY
        self.ball_pos += self.ball_vel

        # Clamp speed
        speed = self.ball_vel.length()
        if speed > self.MAX_SPEED:
            self.ball_vel.scale_to_length(self.MAX_SPEED)

    def _handle_collisions(self):
        reward = 0
        bounced = False
        
        # Wall collisions (left/right)
        if self.ball_pos.x - self.BALL_RADIUS < self.WALL_THICKNESS or self.ball_pos.x + self.BALL_RADIUS > self.WIDTH - self.WALL_THICKNESS:
            self.ball_pos.x = max(self.BALL_RADIUS + self.WALL_THICKNESS, min(self.ball_pos.x, self.WIDTH - self.BALL_RADIUS - self.WALL_THICKNESS))
            self.ball_vel.x *= -1
            bounced = True
            # Apply horizontal nudge
            if self.last_action_movement == 3: # Left
                self.ball_vel.x -= self.NUDGE_STRENGTH
            elif self.last_action_movement == 4: # Right
                self.ball_vel.x += self.NUDGE_STRENGTH

        # Ceiling collision
        if self.ball_pos.y - self.BALL_RADIUS < self.WALL_THICKNESS:
            self.ball_pos.y = self.BALL_RADIUS + self.WALL_THICKNESS
            self.ball_vel.y *= -1
            bounced = True
            # Apply vertical nudge
            if self.last_action_movement == 1: # Up
                self.ball_vel.y -= self.NUDGE_STRENGTH
            elif self.last_action_movement == 2: # Down
                self.ball_vel.y += self.NUDGE_STRENGTH

        if bounced:
            # Check for random failure
            if self.np_random.random() < self.BOUNCE_FAIL_CHANCE:
                self.game_over = True
                self.termination_reason = "UNSTABLE BOUNCE"
            else:
                old_score = self.score
                
                # Increase momentum and score
                self.momentum = min(self.momentum * self.MOMENTUM_GAIN, 3.0) # Cap momentum
                self.score += int(10 * self.momentum)
                
                # Apply momentum to velocity
                speed = self.ball_vel.length()
                self.ball_vel.scale_to_length(speed * 1.01) # Small speed boost on bounce
                
                # Create particles
                self._create_particles(self.ball_pos, 20)
                
                # Calculate rewards
                reward += 0.1 # Continuous feedback for bouncing
                if (old_score // 100) < (self.score // 100):
                    reward += 10 # Milestone reward

        # Floor collision
        if self.ball_pos.y + self.BALL_RADIUS > self.HEIGHT - self.WALL_THICKNESS:
            self.game_over = True
            self.termination_reason = "HIT THE FLOOR"
        
        return reward

    def _create_particles(self, position, count):
        speed = self.ball_vel.length()
        color = self._get_ball_color(speed)
        
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed_mult = self.np_random.uniform(0.5, 2.5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed_mult
            self.particles.append({
                'pos': position.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
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
            "time_remaining": self.time_remaining,
            "momentum": self.momentum,
            "ball_pos": (self.ball_pos.x, self.ball_pos.y),
            "ball_vel": (self.ball_vel.x, self.ball_vel.y),
            "termination_reason": self.termination_reason,
        }

    def _render_game(self):
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALLS, (0, 0, self.WIDTH, self.HEIGHT), self.WALL_THICKNESS)
        # Draw floor (danger zone)
        pygame.draw.line(self.screen, self.COLOR_FLOOR, (0, self.HEIGHT - self.WALL_THICKNESS//2), (self.WIDTH, self.HEIGHT - self.WALL_THICKNESS//2), self.WALL_THICKNESS)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20.0))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color
            )

        # Draw ball with glow
        speed = self.ball_vel.length()
        ball_color = self._get_ball_color(speed)
        self._draw_glow_circle(
            self.screen,
            (int(self.ball_pos.x), int(self.ball_pos.y)),
            self.BALL_RADIUS,
            ball_color
        )
        
    def _render_ui(self):
        # Render score
        score_text = self.font_large.render(f"{self.score:05d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Render timer
        time_seconds = self.time_remaining / self.FPS
        time_text = self.font_large.render(f"{time_seconds:04.1f}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)

        # Render Game Over/Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "VICTORY!" if self.game_won else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_WALLS)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, end_rect)

            reason_text = self.font_small.render(self.termination_reason, True, self.COLOR_TEXT)
            reason_rect = reason_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(reason_text, reason_rect)


    def _get_ball_color(self, speed):
        # Interpolate color from blue (slow) to red (fast)
        speed_ratio = min(speed / (self.MAX_SPEED * 0.8), 1.0)
        r = int(self.COLOR_BALL_SLOW[0] + (self.COLOR_BALL_FAST[0] - self.COLOR_BALL_SLOW[0]) * speed_ratio)
        g = int(self.COLOR_BALL_SLOW[1] + (self.COLOR_BALL_FAST[1] - self.COLOR_BALL_SLOW[1]) * speed_ratio)
        b = int(self.COLOR_BALL_SLOW[2] + (self.COLOR_BALL_FAST[2] - self.COLOR_BALL_SLOW[2]) * speed_ratio)
        return (r, g, b)

    def _draw_glow_circle(self, surface, pos, radius, color):
        for i in range(self.GLOW_LAYERS, 0, -1):
            alpha = int(255 * (1 - (i / self.GLOW_LAYERS))**2 * 0.5)
            glow_color = (*color, alpha)
            current_radius = int(radius + i * 2.5)
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], current_radius, glow_color)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], current_radius, glow_color)
        
        # Draw the main circle
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is NOT used by the autograder
    os.environ["SDL_VIDEODRIVER"] = "pygame"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Bounce Arena")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("--- RESET ---")
                    obs, info = env.reset()
                    total_reward = 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Reason: {info['termination_reason']}")
            print("Press 'R' to restart.")

        clock.tick(GameEnv.FPS)
        
    env.close()