import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:31:53.708980
# Source Brief: brief_01784.md
# Brief Index: 1784
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for managing ball state and properties
class Ball:
    """A class to represent a single ball in the game."""
    def __init__(self, pos, vel, radius, color, friction):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.radius = int(radius)
        self.color = color
        self.friction = friction
        self.trail = []

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Keep a group of bouncing balls inside a shrinking circle. Nudge the currently "
        "selected ball to avoid elimination and survive as long as possible."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to nudge the currently selected ball. The selected "
        "ball changes after each nudge. Keep all balls inside the shrinking circle."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # Visuals
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_BOUNDARY = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.BALL_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
        ]
        self.BALL_COLOR_NAMES = ["RED", "GREEN", "BLUE", "YELLOW", "MAGENTA"]
        self.TRAIL_LENGTH = 20

        # Physics & Gameplay
        self.BALL_FRICTIONS = [0.001, 0.003, 0.005, 0.007, 0.009]
        self.BALL_RADIUS = 12
        self.PLAYER_FORCE_MAGNITUDE = 0.4
        self.INITIAL_BOUNDARY_RADIUS = 180
        self.BOUNDARY_SHRINK_RATE = 1.0 / self.FPS # Shrinks by 1 unit per second

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
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 30)
            self.font_small = pygame.font.SysFont(None, 24)

        # --- Game State (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.boundary_radius = None
        self.boundary_center = None
        self.balls = None
        self.selected_ball_index = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.boundary_radius = float(self.INITIAL_BOUNDARY_RADIUS)
        self.boundary_center = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        self.selected_ball_index = 0
        self.balls = []
        for i in range(5):
            # Spawn balls in a safe central area to avoid instant game over
            while True:
                angle = self.np_random.uniform(0, 2 * math.pi)
                dist = self.np_random.uniform(0, self.INITIAL_BOUNDARY_RADIUS * 0.6)
                pos = self.boundary_center + pygame.math.Vector2(dist, 0).rotate_rad(angle)
                
                is_overlapping = False
                for other_ball in self.balls:
                    if pos.distance_to(other_ball.pos) < self.BALL_RADIUS * 2.2:
                        is_overlapping = True
                        break
                if not is_overlapping:
                    break
            
            vel_angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(0.5, 1.5)
            vel = pygame.math.Vector2(vel_mag, 0).rotate_rad(vel_angle)

            ball = Ball(
                pos=pos,
                vel=vel,
                radius=self.BALL_RADIUS,
                color=self.BALL_COLORS[i],
                friction=self.BALL_FRICTIONS[i]
            )
            self.balls.append(ball)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        
        movement = action[0]
        if movement != 0:
            force_direction = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            force_vec = pygame.math.Vector2(force_direction) * self.PLAYER_FORCE_MAGNITUDE
            self.balls[self.selected_ball_index].vel += force_vec
            
            self.selected_ball_index = (self.selected_ball_index + 1) % len(self.balls)

        self._update_physics()
        self.boundary_radius = max(0, self.boundary_radius - self.BOUNDARY_SHRINK_RATE)

        terminated, win = self._check_termination()
        reward = self._calculate_reward(terminated, win)
        self.score += reward

        truncated = self.steps >= self.MAX_STEPS
        terminated = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated is handled by terminated logic in this game
            self._get_info()
        )

    def _update_physics(self):
        # Ball-Ball collisions
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                self._handle_ball_collision(self.balls[i], self.balls[j])

        # Update position and handle screen boundary collisions
        for ball in self.balls:
            ball.vel *= (1.0 - ball.friction)
            ball.pos += ball.vel
            
            ball.trail.append(ball.pos.copy())
            if len(ball.trail) > self.TRAIL_LENGTH:
                ball.trail.pop(0)

            # Screen boundary bounce
            if ball.pos.x - ball.radius < 0 or ball.pos.x + ball.radius > self.WIDTH:
                ball.pos.x = np.clip(ball.pos.x, ball.radius, self.WIDTH - ball.radius)
                ball.vel.x *= -1
            if ball.pos.y - ball.radius < 0 or ball.pos.y + ball.radius > self.HEIGHT:
                ball.pos.y = np.clip(ball.pos.y, ball.radius, self.HEIGHT - ball.radius)
                ball.vel.y *= -1

    def _handle_ball_collision(self, b1, b2):
        delta = b1.pos - b2.pos
        distance = delta.length()
        
        if 0 < distance < b1.radius + b2.radius:
            overlap = (b1.radius + b2.radius) - distance
            push_vec = delta.normalize() * overlap
            b1.pos += push_vec / 2
            b2.pos -= push_vec / 2

            n = (b2.pos - b1.pos).normalize()
            t = pygame.math.Vector2(-n.y, n.x)

            v1n, v1t = b1.vel.dot(n), b1.vel.dot(t)
            v2n, v2t = b2.vel.dot(n), b2.vel.dot(t)
            
            v1n_new, v2n_new = v2n, v1n
            
            b1.vel = (n * v1n_new) + (t * v1t)
            b2.vel = (n * v2n_new) + (t * v2t)

    def _check_termination(self):
        for ball in self.balls:
            dist_from_center = ball.pos.distance_to(self.boundary_center)
            if dist_from_center > self.boundary_radius - ball.radius:
                self.game_over = True
                return True, False  # terminated=True, win=False

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True, True  # terminated=True, win=True
        
        return False, False

    def _calculate_reward(self, terminated, win):
        if terminated:
            return 100.0 if win else -100.0
        return 0.1 * len(self.balls)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.gfxdraw.aacircle(
            self.screen, int(self.boundary_center.x), int(self.boundary_center.y), 
            int(self.boundary_radius), self.COLOR_BOUNDARY
        )
        
        for i, ball in enumerate(self.balls):
            pos_int = (int(ball.pos.x), int(ball.pos.y))
            
            # Render trail
            if len(ball.trail) > 1:
                for k in range(len(ball.trail) - 1):
                    alpha = (k / self.TRAIL_LENGTH) ** 1.5
                    color = (
                        int(ball.color[0] * alpha * 0.7),
                        int(ball.color[1] * alpha * 0.7),
                        int(ball.color[2] * alpha * 0.7)
                    )
                    p1 = (int(ball.trail[k].x), int(ball.trail[k].y))
                    p2 = (int(ball.trail[k+1].x), int(ball.trail[k+1].y))
                    pygame.draw.line(self.screen, color, p1, p2, 2)
            
            # Render ball
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], ball.radius, ball.color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], ball.radius, ball.color)

            # Render selection highlight
            if i == self.selected_ball_index and not self.game_over:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                glow_radius = ball.radius + 3 + int(pulse * 3)
                glow_color = (255, 255, 150, int(100 + 50 * pulse))
                
                temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, glow_color)
                self.screen.blit(temp_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        elapsed_time = self.steps / self.FPS
        time_left = max(0, (self.MAX_STEPS / self.FPS) - elapsed_time)
        timer_surf = self.font_main.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (10, 10))

        radius_surf = self.font_main.render(f"RADIUS: {self.boundary_radius:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(radius_surf, (self.WIDTH - radius_surf.get_width() - 10, 10))
        
        selected_color_name = self.BALL_COLOR_NAMES[self.selected_ball_index]
        selected_color_value = self.BALL_COLORS[self.selected_ball_index]
        selected_surf = self.font_small.render(f"NUDGE TARGET: {selected_color_name}", True, selected_color_value)
        self.screen.blit(selected_surf, (10, self.HEIGHT - selected_surf.get_height() - 10))

        score_surf = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, self.HEIGHT - score_surf.get_height() - 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "boundary_radius": self.boundary_radius,
            "selected_ball": self.selected_ball_index
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Ball Survival Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        movement_action = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement_action = 1
                elif event.key == pygame.K_DOWN:
                    movement_action = 2
                elif event.key == pygame.K_LEFT:
                    movement_action = 3
                elif event.key == pygame.K_RIGHT:
                    movement_action = 4
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    total_reward = 0.0
                    print("--- Environment Reset ---")
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # In human mode, we only apply one action per frame.
        # The space/shift keys are not used in this brief's gameplay.
        action = [movement_action, 0, 0] 
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0.0
            pygame.time.wait(2000) # Pause for 2 seconds before auto-reset

        clock.tick(env.FPS)

    env.close()