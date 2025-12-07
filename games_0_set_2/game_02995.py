
# Generated: 2025-08-28T06:39:18.566721
# Source Brief: brief_02995.md
# Brief Index: 2995

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
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
        "A fast-paced, neon-drenched block breaker. Clear all the blocks by "
        "bouncing the ball. Chain hits for a combo score multiplier!"
    )

    # Frames auto-advance at 30fps for smooth gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_PADDLE = (0, 150, 255)
        self.COLOR_PADDLE_GLOW = (0, 75, 128)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BALL_GLOW = (128, 128, 128)
        self.COLOR_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = [
            (255, 0, 128), (0, 255, 255), (255, 255, 0),
            (0, 255, 0), (255, 128, 0),
        ]

        # Game parameters
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 8
        self.PADDLE_INFLUENCE = 3.0
        self.INITIAL_LIVES = 3

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_in_play = False
        self.blocks = []
        self.particles = []
        self.combo = 0
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.combo = 0

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        self.ball_in_play = False
        self._reset_ball()

        self.particles = []
        self._create_blocks()
        
        # Assertions for state validation
        assert len(self.blocks) == 100, "Should start with 100 blocks"
        assert self.lives == self.INITIAL_LIVES, "Lives should reset correctly"
        assert self.score == 0, "Score should reset to zero"

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        reward = -0.02  # Small penalty per step to encourage speed

        self._handle_input(action)
        reward += self._update_game_state()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if not self.blocks: # Win condition
                reward += 100
            elif self.lives <= 0: # Lose condition
                reward -= 10

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED

        # Clamp paddle position
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        if space_held and not self.ball_in_play:
            self.ball_in_play = True
            # Sound effect placeholder: # sfx_launch
            self.ball_vel = [self.np_random.uniform(-0.5, 0.5), -self.BALL_SPEED]

    def _update_game_state(self):
        # Update ball position if not attached to paddle
        if not self.ball_in_play:
            self._reset_ball()
            return 0
        
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS,
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
        )

        return self._handle_collisions(ball_rect)

    def _handle_collisions(self, ball_rect):
        reward = 0

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(1, ball_rect.left)
            ball_rect.right = min(self.WIDTH - 1, ball_rect.right)
            # Sound effect placeholder: # sfx_wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(1, ball_rect.top)
            # Sound effect placeholder: # sfx_wall_bounce

        # Bottom wall (lose life)
        if ball_rect.top >= self.HEIGHT:
            self.lives -= 1
            self.ball_in_play = False
            self.combo = 0
            self._reset_ball()
            reward -= 2.0
            # Sound effect placeholder: # sfx_lose_life
            return reward

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # Sound effect placeholder: # sfx_paddle_bounce
            ball_rect.bottom = self.paddle.top
            self.ball_pos[1] = ball_rect.centery
            
            offset = ball_rect.centerx - self.paddle.centerx
            norm_offset = offset / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += norm_offset * self.PADDLE_INFLUENCE
            # Clamp horizontal velocity to prevent extreme angles
            self.ball_vel[0] = max(-self.BALL_SPEED * 0.9, min(self.BALL_SPEED * 0.9, self.ball_vel[0]))
            
            self.ball_vel[1] *= -1
            self.combo = 0 # Combo resets on paddle hit
            reward += 0.1 # Reward for keeping ball in play

        # Block collisions
        hit_block_idx = ball_rect.collidelist([b[0] for b in self.blocks])
        if hit_block_idx != -1:
            block_rect, block_color = self.blocks.pop(hit_block_idx)
            # Sound effect placeholder: # sfx_block_break

            # Determine collision side to correctly reverse velocity
            # A simple approximation: check overlap
            overlap = ball_rect.clip(block_rect)
            if overlap.width < overlap.height:
                self.ball_vel[0] *= -1
            else:
                self.ball_vel[1] *= -1

            self.score += 10
            self.combo += 1
            reward += 1.0 + (0.5 * self.combo)
            self._create_particles(block_rect.center, block_color)

        # Ensure ball velocity doesn't decay
        speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
        if speed > 0 and abs(speed - self.BALL_SPEED) > 0.1:
            self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_SPEED
            self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_SPEED

        self._update_particles()
        return reward

    def _check_termination(self):
        if self.lives <= 0 or not self.blocks or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _create_blocks(self):
        self.blocks = []
        block_width, block_height = 58, 20
        gap = 6
        rows, cols = 10, 10
        start_x = (self.WIDTH - (cols * (block_width + gap) - gap)) / 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                color_index = (r // 2) % len(self.BLOCK_COLORS)
                color = self.BLOCK_COLORS[color_index]
                rect = pygame.Rect(
                    start_x + c * (block_width + gap),
                    start_y + r * (block_height + gap),
                    block_width,
                    block_height,
                )
                self.blocks.append((rect, color))

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Blocks
        for rect, color in self.blocks:
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            darker_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.rect(self.screen, darker_color, rect, width=2, border_radius=3)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color = p['color']
            size = max(1, int(p['lifespan'] / 5))
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, alpha), s.get_rect())
            self.screen.blit(s, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

        # Paddle Glow
        glow_rect = self.paddle.inflate(10, 10)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PADDLE_GLOW, 100), glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, glow_rect.topleft)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Ball Glow
        glow_radius = int(self.BALL_RADIUS * 2.5)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), glow_radius, (*self.COLOR_BALL_GLOW, 50))
        
        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        life_text = self.font_large.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(life_text, (self.WIDTH - 180, 10))
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 80 + i * 30, 25, self.BALL_RADIUS, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 80 + i * 30, 25, self.BALL_RADIUS, self.COLOR_PADDLE)
        
        # Combo
        if self.combo > 1:
            combo_text = self.font_medium.render(f"COMBO x{self.combo}", True, self.BLOCK_COLORS[2])
            text_rect = combo_text.get_rect(center=(self.WIDTH / 2, 20))
            self.screen.blit(combo_text, text_rect)
            
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_lose_msg = "YOU WIN!" if not self.blocks else "GAME OVER"
            end_text = self.font_large.render(win_lose_msg, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
            "combo": self.combo,
        }
    
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")