import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Bounce a ball to break bricks and clear 5 progressively harder levels in this visually vibrant, side-view puzzle game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 8
        self.MAX_STEPS = 10000
        self.INITIAL_BALLS = 3
        self.MAX_LEVEL = 5

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # Colors
        self.COLOR_BG_TOP = (10, 5, 30)
        self.COLOR_BG_BOTTOM = (30, 10, 50)
        self.COLOR_PADDLE = (0, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BALL_GLOW = (0, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.BRICK_COLORS = [
            (255, 0, 255),  # 1 pt - Magenta
            (0, 128, 255),  # 2 pt - Blue
            (0, 255, 0),    # 3 pt - Green
            (255, 255, 0),  # 4 pt - Yellow
            (255, 128, 0),  # 5 pt - Orange
        ]

        # Initialize state variables to be defined in reset()
        self.rng = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 0
        self.current_level = 0
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_stuck = True
        self.ball_speed = 0
        self.bricks = []
        self.particles = []
        self.ball_trail = deque(maxlen=10)
        self.stuck_check_pos = 0
        self.stuck_check_counter = 0

        # Initialize state
        # self.reset() is called by the wrapper, no need to call it here.
        
        # Run validation check - this is good practice but can cause issues with some wrappers
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = self.INITIAL_BALLS
        self.current_level = 1
        
        self.paddle_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self.particles.clear()
        self.ball_trail.clear()

        self._setup_level(self.current_level)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = -0.02  # Small penalty for time passing

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1

        # Handle player input
        self._handle_input(movement, space_held)

        # Update game logic
        step_reward = self._update_game_state()
        reward += step_reward

        # Check for termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        # Apply terminal rewards
        if terminated and not self.game_over: # Max steps reached
            self.game_over = True
        
        if self.game_over:
            if self.current_level > self.MAX_LEVEL: # Win condition
                reward += 100.0
            elif self.balls_left <= 0: # Lose condition
                reward -= 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Paddle movement
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle_rect.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle_rect.x))

        # Launch ball
        if self.ball_stuck and space_held:
            # sfx: launch_ball.wav
            self.ball_stuck = False
            launch_angle = self.rng.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball_vel = pygame.Vector2(math.cos(launch_angle), math.sin(launch_angle)) * self.ball_speed

    def _update_game_state(self):
        reward = 0
        
        if self.ball_stuck:
            self.ball_pos.x = self.paddle_rect.centerx
            self.ball_pos.y = self.paddle_rect.top - self.BALL_RADIUS
        else:
            self.ball_trail.append(pygame.Vector2(self.ball_pos))
            self.ball_pos += self.ball_vel

            # Collisions
            reward += self._handle_collisions()

        # Level clear
        if not self.bricks:
            # sfx: level_clear.wav
            reward += 10.0
            self.current_level += 1
            if self.current_level > self.MAX_LEVEL:
                self.game_over = True # Game won!
            else:
                self._setup_level(self.current_level)
                
        # Update particles
        self._update_particles()
        
        # Anti-softlock mechanism
        self._check_softlock()

        return reward

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.SCREEN_WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.SCREEN_WIDTH - self.BALL_RADIUS, self.ball_pos.x))
            # sfx: bounce_wall.wav
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = max(self.BALL_RADIUS, self.ball_pos.y)
            # sfx: bounce_wall.wav

        # Bottom "wall" (lose ball)
        if self.ball_pos.y + self.BALL_RADIUS >= self.SCREEN_HEIGHT:
            # sfx: lose_ball.wav
            self.balls_left -= 1
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
            return 0 # Terminal reward handled in step()

        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel.y > 0:
            # sfx: bounce_paddle.wav
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle_rect.top - self.BALL_RADIUS
            
            # Influence horizontal velocity based on hit location
            offset = (self.ball_pos.x - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * 2.0
            self.ball_vel.normalize_ip()
            self.ball_vel *= self.ball_speed

        # Brick collisions
        for brick, value in self.bricks[:]:
            if ball_rect.colliderect(brick):
                # sfx: break_brick.wav
                self.bricks.remove((brick, value))
                reward += 1.0 + value
                self.score += value
                self._spawn_particles(brick.center, self.BRICK_COLORS[value-1])
                
                # Simple bounce logic: reverse vertical velocity
                self.ball_vel.y *= -1
                break # Only break one brick per frame
        
        return reward

    def _setup_level(self, level_num):
        self.bricks.clear()
        self._reset_ball()
        
        # Increase ball speed per level
        self.ball_speed = 6.0 + (level_num - 1) * 0.5
        
        brick_width, brick_height = 50, 20
        gap = 5
        rows, cols = 0, 0
        
        layout = self._get_level_layout(level_num)
        rows = len(layout)
        cols = len(layout[0]) if rows > 0 else 0
        
        total_width = cols * (brick_width + gap) - gap
        start_x = (self.SCREEN_WIDTH - total_width) / 2
        start_y = 50

        for r, row_data in enumerate(layout):
            for c, brick_val in enumerate(row_data):
                if brick_val > 0:
                    x = start_x + c * (brick_width + gap)
                    y = start_y + r * (brick_height + gap)
                    self.bricks.append((pygame.Rect(x, y, brick_width, brick_height), brick_val))

    def _get_level_layout(self, level_num):
        if level_num == 1:
            return [[1, 1, 1, 1, 1, 1, 1, 1, 1]]
        elif level_num == 2:
            return [[2, 2, 2, 2, 0, 2, 2, 2, 2],
                    [1, 1, 1, 1, 0, 1, 1, 1, 1]]
        elif level_num == 3:
            return [[0, 0, 0, 3, 3, 3, 0, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0]]
        elif level_num == 4:
            return [[4, 0, 3, 0, 2, 0, 3, 0, 4],
                    [4, 0, 3, 0, 2, 0, 3, 0, 4],
                    [4, 0, 3, 0, 2, 0, 3, 0, 4]]
        elif level_num == 5:
            return [[5, 4, 3, 2, 1, 2, 3, 4, 5],
                    [0, 5, 4, 3, 2, 3, 4, 5, 0],
                    [0, 0, 5, 4, 3, 4, 5, 0, 0],
                    [0, 0, 0, 5, 4, 5, 0, 0, 0]]
        return []

    def _reset_ball(self):
        self.ball_stuck = True
        self.ball_pos = pygame.Vector2(
            self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS
        )
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_trail.clear()
        self.stuck_check_counter = 0

    def _check_softlock(self):
        if not self.ball_stuck:
            if abs(self.ball_pos.y - self.stuck_check_pos) < 0.1:
                self.stuck_check_counter += 1
            else:
                self.stuck_check_counter = 0
                self.stuck_check_pos = self.ball_pos.y
            
            if self.stuck_check_counter > 150: # After 5 seconds of being stuck horizontally
                self.ball_vel.y += self.rng.choice([-0.5, 0.5])
                self.ball_vel.normalize_ip()
                self.ball_vel *= self.ball_speed
                self.stuck_check_counter = 0
                
    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "level": self.current_level,
        }

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game_elements(self):
        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['life']))

        # Ball trail
        for i, pos in enumerate(self.ball_trail):
            alpha = int(255 * (i / len(self.ball_trail)))
            trail_surf = pygame.Surface((self.BALL_RADIUS*2, self.BALL_RADIUS*2), pygame.SRCALPHA)
            pygame.draw.circle(trail_surf, (*self.COLOR_BALL_GLOW, alpha), (self.BALL_RADIUS, self.BALL_RADIUS), self.BALL_RADIUS)
            self.screen.blit(trail_surf, (int(pos.x - self.BALL_RADIUS), int(pos.y - self.BALL_RADIUS)))

        # Ball glow and ball
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_BALL_GLOW, 50))
        pygame.gfxdraw.aacircle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_BALL_GLOW, 70))
        self.screen.blit(glow_surf, (int(self.ball_pos.x-glow_radius), int(self.ball_pos.y-glow_radius)))
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=5)

        # Bricks
        for brick, value in self.bricks:
            color = self.BRICK_COLORS[value - 1]
            pygame.draw.rect(self.screen, color, brick, border_radius=3)
            # Add an inner highlight for depth
            highlight_color = tuple(min(255, c + 50) for c in color)
            inner_rect = brick.inflate(-6, -6)
            pygame.draw.rect(self.screen, highlight_color, inner_rect, border_radius=2)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        level_text = self.font_large.render(f"LEVEL: {self.current_level}/{self.MAX_LEVEL}", True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(centerx=self.SCREEN_WIDTH / 2, y=10)
        self.screen.blit(level_text, level_rect)

        balls_text = self.font_large.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        balls_rect = balls_text.get_rect(right=self.SCREEN_WIDTH - 10, y=10)
        self.screen.blit(balls_text, balls_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "LEVELS CLEARED" if self.current_level > self.MAX_LEVEL else "GAME OVER"
            end_text_surf = self.font_large.render(win_text, True, self.COLOR_BALL)
            end_text_rect = end_text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(end_text_surf, end_text_rect)

            final_score_surf = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20))
            self.screen.blit(final_score_surf, final_score_rect)


    def _spawn_particles(self, pos, color):
        for _ in range(10):
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(self.rng.uniform(-2, 2), self.rng.uniform(-2, 2)),
                'life': self.rng.uniform(3, 6),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 0.3
            if p['life'] <= 0:
                self.particles.remove(p)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")