import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→ to move the paddle."

    # Must be a short, user-facing description of the game:
    game_description = "Survive the onslaught of bouncing balls for 60 seconds in this isometric-2D arcade game."

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_WALL = (180, 180, 200)
        self.COLOR_PADDLE = (0, 150, 255)
        self.COLOR_PADDLE_GLOW = (0, 150, 255, 50)
        self.COLOR_BALL_SLOW = (255, 100, 180)
        self.COLOR_BALL_FAST = (255, 0, 100)
        self.COLOR_TEXT = (50, 255, 50)
        self.COLOR_PARTICLE_LOSE = (255, 50, 50)

        # Play area definition (for isometric look)
        self.play_area_rect = pygame.Rect(50, 50, self.WIDTH - 100, self.HEIGHT - 100)

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
        self.font = pygame.font.Font(None, 36)
        
        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.paddle = None
        self.balls = []
        self.particles = []
        self.base_ball_speed = 0.0
        self.last_move_was_unnecessary = False
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Paddle
        paddle_width = 100
        paddle_height = 10
        self.paddle = pygame.Rect(
            self.play_area_rect.centerx - paddle_width / 2,
            self.play_area_rect.bottom - paddle_height,
            paddle_width,
            paddle_height
        )
        self.paddle_speed = 8

        # Balls
        self.balls = []
        self.base_ball_speed = 3.0
        self._spawn_ball()

        # Particles
        self.particles = []

        # Reward tracking
        self.last_move_was_unnecessary = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        # --- Update Game Logic ---
        reward, terminated = self._update_game_state(movement)
        
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated:
            self.game_over = True
            reward = -10 # Loss penalty
        elif truncated:
            self.game_over = True
            reward += 100 # Win bonus
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_game_state(self, movement):
        # --- Action Handling ---
        paddle_moved = False
        if movement == 3: # Left
            self.paddle.x -= self.paddle_speed
            paddle_moved = True
        elif movement == 4: # Right
            self.paddle.x += self.paddle_speed
            paddle_moved = True
        
        self.paddle.left = max(self.play_area_rect.left, self.paddle.left)
        self.paddle.right = min(self.play_area_rect.right, self.paddle.right)

        # --- Reward for unnecessary movement ---
        self.last_move_was_unnecessary = False
        if paddle_moved:
            threat_zone = self.paddle.inflate(100, 200)
            threat_zone.top -= 200
            # FIX: Pygame's collidepoint expects a tuple or two numbers, not a numpy array.
            is_ball_threatening = any(threat_zone.collidepoint(tuple(ball['pos'])) for ball in self.balls)
            if not is_ball_threatening:
                self.last_move_was_unnecessary = True

        # --- Core Updates ---
        self.steps += 1
        
        self._update_particles()
        bounce_reward, terminated = self._update_balls()

        # --- Difficulty Scaling ---
        self._scale_difficulty()

        # --- Final Reward Calculation ---
        reward = bounce_reward + 0.1 # Survival reward
        if self.last_move_was_unnecessary:
            reward -= 0.2
        
        return reward, terminated

    def _update_particles(self):
        # Efficiently remove dead particles using a list comprehension
        self.particles = [p for p in self.particles if p['lifespan'] > 1]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _update_balls(self):
        bounce_reward = 0
        terminated = False
        for ball in self.balls:
            ball['pos'] += ball['vel']

            # Wall collisions
            if ball['pos'][0] - ball['radius'] <= self.play_area_rect.left:
                ball['pos'][0] = self.play_area_rect.left + ball['radius']
                ball['vel'][0] *= -1
                self._create_particles(ball['pos'], 10)
            elif ball['pos'][0] + ball['radius'] >= self.play_area_rect.right:
                ball['pos'][0] = self.play_area_rect.right - ball['radius']
                ball['vel'][0] *= -1
                self._create_particles(ball['pos'], 10)

            if ball['pos'][1] - ball['radius'] <= self.play_area_rect.top:
                ball['pos'][1] = self.play_area_rect.top + ball['radius']
                ball['vel'][1] *= -1
                self._create_particles(ball['pos'], 10)

            # Paddle collision
            ball_rect = pygame.Rect(ball['pos'][0] - ball['radius'], ball['pos'][1] - ball['radius'], ball['radius']*2, ball['radius']*2)
            if ball['vel'][1] > 0 and self.paddle.colliderect(ball_rect):
                ball['pos'][1] = self.paddle.top - ball['radius']
                ball['vel'][1] *= -1
                hit_offset = (ball['pos'][0] - self.paddle.centerx) / (self.paddle.width / 2)
                ball['vel'][0] += hit_offset * 2.0
                
                current_speed = np.linalg.norm(ball['vel'])
                if current_speed > 0:
                    ball['vel'] = (ball['vel'] / current_speed) * ball['speed']

                self.score += 1
                bounce_reward += 1
                self._create_particles(ball['pos'], 20, self.COLOR_PADDLE)
                
            # Bottom edge (Game Over)
            if ball['pos'][1] + ball['radius'] > self.play_area_rect.bottom:
                terminated = True
                self._create_particles(ball['pos'], 50, self.COLOR_PARTICLE_LOSE)
        
        return bounce_reward, terminated

    def _scale_difficulty(self):
        if self.steps > 0:
            if self.steps % 100 == 0:
                self.base_ball_speed += 0.05
                for ball in self.balls:
                    current_speed = np.linalg.norm(ball['vel'])
                    if current_speed > 0:
                       ball['vel'] = (ball['vel'] / current_speed) * self.base_ball_speed
                    ball['speed'] = self.base_ball_speed
            if self.steps % 500 == 0:
                self._spawn_ball()

    def _spawn_ball(self):
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        speed = self.base_ball_speed
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        
        ball = {
            'pos': np.array([self.play_area_rect.centerx, self.play_area_rect.top + 20], dtype=np.float64),
            'vel': np.array(vel, dtype=np.float64),
            'radius': 8,
            'speed': speed
        }
        self.balls.append(ball)

    def _create_particles(self, pos, count, color=None):
        if color is None: color = self.COLOR_WALL
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({
                'pos': np.array(pos, dtype=np.float64),
                'vel': np.array(vel, dtype=np.float64),
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_isometric_grid()
        self._render_walls()
        self._render_particles()
        self._render_balls()
        self._render_paddle()

    def _render_isometric_grid(self):
        spacing = 40
        for i in range(-self.WIDTH // spacing, self.WIDTH // spacing * 2):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i * spacing, -self.HEIGHT), (i * spacing + self.HEIGHT, self.HEIGHT), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID, (i * spacing, self.HEIGHT * 2), (i * spacing - self.HEIGHT, 0), 1)
            
    def _render_walls(self):
        pr = self.play_area_rect
        depth = 20
        top_left, top_right, bottom_left = pr.topleft, pr.topright, pr.bottomleft
        
        pygame.draw.polygon(self.screen, self.COLOR_WALL, [
            top_left, top_right, (top_right[0] - depth, top_right[1] - depth), (top_left[0] - depth, top_left[1] - depth)
        ])
        # FIX: Pygame colors must be integers. The generator was creating floats.
        dark_wall_color = tuple(int(c * 0.8) for c in self.COLOR_WALL)
        pygame.draw.polygon(self.screen, dark_wall_color, [
            top_left, bottom_left, (bottom_left[0] - depth, bottom_left[1] - depth), (top_left[0] - depth, top_left[1] - depth)
        ])
        
        pygame.draw.line(self.screen, self.COLOR_BG, pr.topleft, pr.topright, 2)
        pygame.draw.line(self.screen, self.COLOR_BG, pr.topleft, pr.bottomleft, 2)
        pygame.draw.line(self.screen, self.COLOR_BG, pr.bottomleft, pr.bottomright, 2)
        pygame.draw.line(self.screen, self.COLOR_BG, pr.topright, pr.bottomright, 2)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color_with_alpha = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, color_with_alpha
            )

    def _render_balls(self):
        max_speed_for_color = 8.0
        for ball in self.balls:
            speed_ratio = min(1.0, max(0.0, (ball['speed'] - 3.0) / (max_speed_for_color - 3.0)))
            color = [int(c1 + (c2 - c1) * speed_ratio) for c1, c2 in zip(self.COLOR_BALL_SLOW, self.COLOR_BALL_FAST)]
            pos_int = (int(ball['pos'][0]), int(ball['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], ball['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], ball['radius'], color)

    def _render_paddle(self):
        glow_rect = self.paddle.inflate(10, 10)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, self.COLOR_PADDLE_GLOW, glow_surface.get_rect(), border_radius=5)
        self.screen.blit(glow_surface, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        remaining_seconds = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = self.font.render(f"TIME: {max(0, remaining_seconds):.1f}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(timer_text, timer_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls": len(self.balls),
            "ball_speed": self.base_ball_speed,
        }

    def close(self):
        pygame.quit()