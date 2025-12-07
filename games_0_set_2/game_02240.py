
# Generated: 2025-08-27T19:43:28.507724
# Source Brief: brief_02240.md
# Brief Index: 2240

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class Particle:
    """A simple class for a cosmetic particle effect."""
    def __init__(self, x, y, vx, vy, radius, color, lifetime):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime

    def update(self):
        """Update particle position and lifetime."""
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # Gravity effect
        self.lifetime -= 1

    def draw(self, surface):
        """Draw the particle on the screen, fading it out."""
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            current_color = (*self.color, alpha)
            # Use gfxdraw for anti-aliased circles
            pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), int(self.radius), current_color)
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), current_color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the paddle. Keep the ball in play. "
        "Bouncing the ball on the edges of the paddle gives more points."
    )

    game_description = (
        "A minimalist, grid-based arcade game. Control a paddle to keep a "
        "bouncing ball in play, aiming for a high score by taking risky shots."
    )

    auto_advance = True

    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_CELLS_X = 10
    GRID_CELLS_Y = 10
    GRID_WIDTH = 600
    GRID_HEIGHT = 360
    MARGIN_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    MARGIN_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 40)
    COLOR_PADDLE = (0, 200, 200)
    COLOR_BALL = (220, 0, 150)
    COLOR_TEXT = (240, 240, 240)
    COLOR_DANGER = (255, 50, 50)
    COLOR_WALL = (100, 100, 110)

    # Game Parameters
    PADDLE_WIDTH = 120  # 2 grid cells
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 20
    BALL_RADIUS = 10
    BALL_SPEED = 6
    MAX_LIVES = 3
    WIN_SCORE = 100
    MAX_STEPS = 1800 # 60 seconds at 30fps

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

        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        self.paddle_pos = 0
        self.ball_pos = [0, 0]
        self.ball_vel = [0, 0]
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle_pos = self.SCREEN_WIDTH / 2
        
        self.ball_pos = [self.SCREEN_WIDTH / 2, self.MARGIN_Y + self.BALL_RADIUS + 50]
        # Give the ball a random downward direction
        angle = self.np_random.uniform(math.pi * 0.35, math.pi * 0.65)
        self.ball_vel = [
            math.cos(angle) * self.BALL_SPEED * self.np_random.choice([-1, 1]),
            math.sin(angle) * self.BALL_SPEED
        ]

        self.score = 0
        self.lives = self.MAX_LIVES
        self.steps = 0
        self.game_over = False
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = self._update_game_logic(movement)

        self.steps += 1
        terminated = self._check_termination()
        
        # Terminal rewards
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100
            elif self.lives <= 0:
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_logic(self, movement):
        reward = 0

        # 1. Handle player input
        if movement == 3:  # Left
            self.paddle_pos -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_pos += self.PADDLE_SPEED

        # Clamp paddle position within grid boundaries
        self.paddle_pos = np.clip(
            self.paddle_pos,
            self.MARGIN_X + self.PADDLE_WIDTH / 2,
            self.SCREEN_WIDTH - self.MARGIN_X - self.PADDLE_WIDTH / 2
        )

        # 2. Update ball position
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # 3. Collision detection
        # Walls (left/right)
        if self.ball_pos[0] <= self.MARGIN_X + self.BALL_RADIUS or self.ball_pos[0] >= self.SCREEN_WIDTH - self.MARGIN_X - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.MARGIN_X + self.BALL_RADIUS, self.SCREEN_WIDTH - self.MARGIN_X - self.BALL_RADIUS)
            self._create_particles(self.ball_pos, self.COLOR_WALL, 5)
            # sfx: wall_bounce.wav

        # Wall (top)
        if self.ball_pos[1] <= self.MARGIN_Y + self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.MARGIN_Y + self.BALL_RADIUS
            self._create_particles(self.ball_pos, self.COLOR_WALL, 5)
            # sfx: wall_bounce.wav

        # Paddle collision
        paddle_y = self.SCREEN_HEIGHT - self.MARGIN_Y - self.PADDLE_HEIGHT
        if self.ball_vel[1] > 0 and (paddle_y - self.BALL_RADIUS) < self.ball_pos[1] < (paddle_y + self.BALL_RADIUS):
            paddle_left = self.paddle_pos - self.PADDLE_WIDTH / 2
            paddle_right = self.paddle_pos + self.PADDLE_WIDTH / 2
            if paddle_left < self.ball_pos[0] < paddle_right:
                # Collision occurred
                self.ball_vel[1] *= -1
                self.ball_pos[1] = paddle_y - self.BALL_RADIUS # prevent sticking

                hit_offset = (self.ball_pos[0] - self.paddle_pos) / (self.PADDLE_WIDTH / 2)
                hit_offset = np.clip(hit_offset, -1.0, 1.0)
                
                # Modify horizontal velocity based on where it hit
                self.ball_vel[0] += hit_offset * 4 
                # Normalize speed to prevent it from getting too fast/slow
                speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
                if speed > 0:
                    self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_SPEED
                    self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_SPEED

                # Calculate reward and score
                hit_offset_abs = abs(hit_offset)
                if hit_offset_abs >= 0.9: # Risky bounce (outer 10% on each side)
                    reward += 1.0
                    self.score += 2
                    self._create_particles(self.ball_pos, (255, 255, 100), 20)
                    # sfx: risky_bounce.wav
                else: # Normal bounce
                    self.score += 1
                    reward += 0.1
                    if hit_offset_abs <= 0.2: # Safe bounce (center 20%)
                        reward -= 0.02
                    self._create_particles(self.ball_pos, self.COLOR_PADDLE, 10)
                    # sfx: paddle_bounce.wav

        # Miss (ball hits bottom)
        if self.ball_pos[1] > self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 10 # Small penalty for a miss in addition to terminal
            # sfx: miss.wav
            if self.lives > 0:
                # Reset ball
                self.ball_pos = [self.SCREEN_WIDTH / 2, self.MARGIN_Y + self.BALL_RADIUS + 50]
                angle = self.np_random.uniform(math.pi * 0.35, math.pi * 0.65)
                self.ball_vel = [
                    math.cos(angle) * self.BALL_SPEED * self.np_random.choice([-1, 1]),
                    math.sin(angle) * self.BALL_SPEED
                ]
            else:
                self.game_over = True
        
        # 4. Update particles
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            return True
        if self.lives <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_CELLS_X + 1):
            x = self.MARGIN_X + i * (self.GRID_WIDTH / self.GRID_CELLS_X)
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.MARGIN_Y), (x, self.SCREEN_HEIGHT - self.MARGIN_Y))
        for i in range(self.GRID_CELLS_Y + 1):
            y = self.MARGIN_Y + i * (self.GRID_HEIGHT / self.GRID_CELLS_Y)
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.MARGIN_X, y), (self.SCREEN_WIDTH - self.MARGIN_X, y))

        # Draw walls
        wall_rect = pygame.Rect(self.MARGIN_X, self.MARGIN_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_WALL, wall_rect, 2)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw ball with a glow effect
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        glow_color = (*self.COLOR_BALL, 50)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 4, glow_color)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 4, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

        # Draw paddle
        paddle_rect = pygame.Rect(
            self.paddle_pos - self.PADDLE_WIDTH / 2,
            self.SCREEN_HEIGHT - self.MARGIN_Y - self.PADDLE_HEIGHT,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)

    def _render_ui(self):
        # Render score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Render lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 180, 10))
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH - 80 + i * 30, 22, 10, self.COLOR_DANGER)
            pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH - 80 + i * 30, 22, 10, self.COLOR_DANGER)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = self.COLOR_PADDLE
            else:
                msg = "GAME OVER"
                color = self.COLOR_DANGER
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            radius = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append(Particle(pos[0], pos[1], vx, vy, radius, color, lifetime))

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use 'windows' or 'x11' or 'dummy'

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()

    terminated = False
    while not terminated:
        # Map keyboard keys to the MultiDiscrete action space
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()

        if terminated:
            print(f"Game Over. Final Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds before closing
            
        clock.tick(30) # Run at 30 FPS

    env.close()