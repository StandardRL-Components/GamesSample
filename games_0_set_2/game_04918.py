import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑/↓ to move paddle. Hold Shift for a sharp-angle hit. Press Space for a temporary speed boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pixel Pong: A fast-paced, grid-based pong game. Hit the ball to score, but miss 5 times and you're out. Score 7 points to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.CELL_WIDTH = self.SCREEN_WIDTH // self.GRID_COLS
        self.CELL_HEIGHT = self.SCREEN_HEIGHT // self.GRID_ROWS

        self.COLOR_BG = (15, 15, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PADDLE = (60, 160, 255)
        self.COLOR_PADDLE_GLOW = (30, 80, 130)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BALL_GLOW = (200, 200, 255)
        self.COLOR_SCORE = (100, 255, 100)
        self.COLOR_LIVES = (255, 80, 80)
        self.COLOR_TEXT = (220, 220, 220)

        self.PADDLE_HEIGHT_CELLS = 3
        self.PADDLE_WIDTH = 12
        self.PADDLE_SPEED = self.CELL_HEIGHT / 2.5
        self.PADDLE_X = self.CELL_WIDTH - self.PADDLE_WIDTH - 10

        self.BALL_SIZE = 16
        self.BASE_BALL_SPEED = self.CELL_WIDTH / 6.0

        self.MAX_SCORE = 7
        self.MAX_LIVES = 5
        self.MAX_STEPS = 1500  # Increased for longer games

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("monospace", 50, bold=True)

        # --- Game State ---
        self.paddle_y = 0
        self.ball_pos = np.zeros(2, dtype=float)
        self.ball_vel = np.zeros(2, dtype=float)
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        self.win = False

        self.particles = []
        self.speed_boost_timer = 0
        self.sharp_angle_armed = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.paddle_y = self.SCREEN_HEIGHT / 2 - (self.PADDLE_HEIGHT_CELLS * self.CELL_HEIGHT) / 2

        # Reset ball
        self._reset_ball()

        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.win = False

        self.particles = []
        self.speed_boost_timer = 0
        self.sharp_angle_armed = False

        return self._get_observation(), self._get_info()

    def _reset_ball(self, to_player=True):
        self.ball_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        if to_player:
            angle += math.pi

        speed = self.BASE_BALL_SPEED
        self.ball_vel = np.array([speed * math.cos(angle), speed * math.sin(angle)])

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)  # Maintain 30 FPS

        if self.game_over:
            # If game is over, no state should change.
            # Return the final state.
            reward = 0
            terminated = True
            return (
                self._get_observation(), reward, terminated, False, self._get_info()
            )

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Paddle movement
        if movement == 1:  # Up
            self.paddle_y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle_y += self.PADDLE_SPEED

        paddle_max_h = self.PADDLE_HEIGHT_CELLS * self.CELL_HEIGHT
        self.paddle_y = np.clip(self.paddle_y, 0, self.SCREEN_HEIGHT - paddle_max_h)

        # Special moves
        if space_held and self.speed_boost_timer <= 0:
            self.speed_boost_timer = 15  # Boost for 0.5s
            # sfx: Powerup activate

        if shift_held:
            self.sharp_angle_armed = True

        # --- Game Logic & Physics ---
        reward = 0

        # Update ball position
        speed_multiplier = 2.0 if self.speed_boost_timer > 0 else 1.0
        self.ball_pos += self.ball_vel * speed_multiplier
        if self.speed_boost_timer > 0:
            self.speed_boost_timer -= 1

        # --- Collision Detection ---
        ball_rect = pygame.Rect(self.ball_pos[0], self.ball_pos[1], self.BALL_SIZE, self.BALL_SIZE)
        paddle_rect = pygame.Rect(self.PADDLE_X, self.paddle_y, self.PADDLE_WIDTH,
                                  self.PADDLE_HEIGHT_CELLS * self.CELL_HEIGHT)

        # Top/Bottom wall collision
        if ball_rect.top < 0 or ball_rect.bottom > self.SCREEN_HEIGHT:
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            ball_rect.bottom = min(self.SCREEN_HEIGHT, ball_rect.bottom)
            self.ball_pos[1] = ball_rect.y
            self._create_particles(self.ball_pos + self.BALL_SIZE / 2, 5, self.COLOR_BALL)
            # sfx: Wall bounce

        # Right wall collision
        if ball_rect.right > self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.right = self.SCREEN_WIDTH
            self.ball_pos[0] = ball_rect.x
            self._create_particles(self.ball_pos + self.BALL_SIZE / 2, 5, self.COLOR_BALL)
            # sfx: Wall bounce

        # Paddle collision
        if ball_rect.colliderect(paddle_rect) and self.ball_vel[0] < 0:
            self.score += 1
            reward += 1  # Reward for scoring a point

            # sfx: Paddle hit
            # FIX: Convert pygame.Rect.midleft tuple to a numpy array
            self._create_particles(np.array(ball_rect.midleft, dtype=float), 20, self.COLOR_PADDLE)

            # Reverse horizontal velocity
            self.ball_vel[0] *= -1.05  # Increase speed slightly on each hit
            self.ball_vel[0] = min(self.ball_vel[0], self.CELL_WIDTH / 2)  # Cap speed

            # Vertical velocity change based on hit position
            hit_center = paddle_rect.centery
            ball_center = ball_rect.centery
            relative_impact = (hit_center - ball_center) / (paddle_rect.height / 2)

            angle_mod = relative_impact * 0.8  # Base angle change
            if self.sharp_angle_armed:
                angle_mod *= 2.5  # Sharp angle effect
                self.sharp_angle_armed = False
                # sfx: Sharp angle hit

            self.ball_vel[1] -= angle_mod * abs(self.ball_vel[0] * 0.5)

            # Normalize velocity to prevent runaway speeds but maintain new direction
            current_speed = np.linalg.norm(self.ball_vel)
            base_speed = self.BASE_BALL_SPEED + (self.score * 0.2)  # Speed scales with score
            if current_speed > 0:
                self.ball_vel = (self.ball_vel / current_speed) * base_speed

            self.ball_pos[0] = paddle_rect.right  # Prevent sticking

        # Miss (Left wall)
        elif ball_rect.left < 0:
            self.lives -= 1
            reward -= 1  # Penalty for losing a life
            self._reset_ball(to_player=False)  # Serve to opponent side
            # sfx: Life lost

        # --- Particle Update ---
        self._update_particles()

        # --- Termination Check ---
        self.steps += 1
        terminated = False
        truncated = False
        if self.score >= self.MAX_SCORE:
            terminated = True
            self.win = True
            reward += 100  # Win bonus
            # sfx: Win jingle
        elif self.lives <= 0:
            terminated = True
            self.win = False
            reward -= 100  # Lose penalty
            # sfx: Lose sound
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for timeout
            # No bonus/penalty for timeout

        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(1, self.GRID_COLS):
            x = i * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for i in range(1, self.GRID_ROWS):
            y = i * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

        # Draw particles
        for p in self.particles:
            p_size = int(p['size'] * (p['life'] / p['max_life']))
            if p_size > 0:
                pygame.draw.rect(self.screen, p['color'],
                                 (int(p['pos'][0] - p_size / 2), int(p['pos'][1] - p_size / 2), p_size, p_size))

        # Draw paddle
        paddle_h = self.PADDLE_HEIGHT_CELLS * self.CELL_HEIGHT
        paddle_rect = pygame.Rect(int(self.PADDLE_X), int(self.paddle_y), self.PADDLE_WIDTH, int(paddle_h))

        # Glow effect for shift
        if self.sharp_angle_armed:
            glow_rect = paddle_rect.inflate(10, 10)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE, 80), (0, 0, *glow_rect.size), border_radius=8)
            self.screen.blit(glow_surf, glow_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)

        # Draw ball
        ball_rect = pygame.Rect(int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_SIZE, self.BALL_SIZE)

        # Glow effect for speed boost
        glow_color = self.COLOR_PADDLE if self.speed_boost_timer > 0 else self.COLOR_BALL_GLOW
        alpha = 150 if self.speed_boost_timer > 0 else 100
        glow_size = self.BALL_SIZE * 2.5
        pygame.gfxdraw.filled_circle(
            self.screen,
            int(ball_rect.centerx), int(ball_rect.centery),
            int(glow_size / 2),
            (*glow_color, alpha)
        )
        pygame.draw.rect(self.screen, self.COLOR_BALL, ball_rect, border_radius=4)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 20))

        # Lives
        for i in range(self.lives):
            self._draw_heart(self.screen, 30 + i * 35, 35, 12, self.COLOR_LIVES)

        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_SCORE if self.win else self.COLOR_LIVES
            text_surf = self.font_gameover.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))

            # Background for text
            bg_rect = text_rect.inflate(40, 40)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((*self.COLOR_BG, 200))
            self.screen.blit(bg_surf, bg_rect)

            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color,
                'size': self.np_random.uniform(3, 8)
            })

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95  # friction
            p['life'] -= 1
            if p['life'] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _draw_heart(self, surface, x, y, size, color):
        # Simple procedural heart shape
        points = [
            (x, y + size * 0.75),
            (x - size, y - size * 0.25),
            (x - size * 0.5, y - size * 0.75),
            (x, y - size * 0.25),
            (x + size * 0.5, y - size * 0.75),
            (x + size, y - size * 0.25),
        ]
        pygame.draw.polygon(surface, color, points)


    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game with keyboard controls
    # It requires pygame to be installed with a display driver.
    # To run headlessly, comment out this block.
    
    # Unset the dummy video driver if we want to render
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    pygame.display.set_caption("Pixel Pong")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    terminated = False
    truncated = False

    while not terminated and not truncated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0  # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is already the rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()  # Reset on 'R' key
                terminated = False
                truncated = False

    # Keep the final screen for 2 seconds
    if env.game_over:
        pygame.time.wait(2000)

    env.close()