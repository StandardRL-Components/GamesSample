
# Generated: 2025-08-27T16:12:38.059973
# Source Brief: brief_01154.md
# Brief Index: 1154

        
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
        "A fast-paced, top-down block breaker. Destroy all blocks before you run out of time or balls!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (15, 15, 35)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 255, 150)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TIMER_BAR = (0, 255, 0)
    COLOR_TIMER_BAR_WARN = (255, 255, 0)
    COLOR_TIMER_BAR_DANGER = (255, 0, 0)

    BLOCK_COLORS = {
        1: (0, 200, 0),    # Green
        2: (0, 150, 255),  # Blue
        3: (220, 50, 50),  # Red
    }
    BLOCK_OUTLINE_DARKEN_FACTOR = 0.6

    # Game element properties
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    BALL_SPEED_INITIAL = 5
    BALL_SPEED_MAX_X = 6

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_game_over = pygame.font.SysFont('Consolas', 50, bold=True)

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = None
        self.blocks = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.balls_left = 0
        self.game_over = False
        self.win = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_attached = True
        self._reset_ball()

        self._create_blocks()
        self.particles = []

        self.steps = 0
        self.score = 0
        self.balls_left = 3
        self.game_over = False
        self.win = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1

        if not self.game_over:
            self._handle_input(action)
            self._update_physics()

            block_collision_reward = self._handle_block_collisions()
            reward += block_collision_reward

            if self._check_ball_loss():
                reward -= 50 # Large penalty for losing a ball
                self.balls_left -= 1
                if self.balls_left > 0:
                    self._reset_ball()
                else:
                    self.game_over = True

            self._update_particles()

        # Check termination conditions
        if not self.game_over:
            if not self.blocks:
                self.game_over = True
                self.win = True
                reward += 50 # Large reward for winning
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True

        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _reset_ball(self):
        self.ball_attached = True
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=float)
        self.ball_vel = np.array([0.0, 0.0])

    def _create_blocks(self):
        self.blocks = []
        block_width = 50
        block_height = 20
        rows = 5
        cols = self.SCREEN_WIDTH // (block_width + 4)
        start_x = (self.SCREEN_WIDTH - cols * (block_width + 4)) // 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                points = 1 if r >= 3 else (2 if r >= 1 else 3)
                color = self.BLOCK_COLORS[points]
                rect = pygame.Rect(
                    start_x + c * (block_width + 4),
                    start_y + r * (block_height + 4),
                    block_width,
                    block_height
                )
                self.blocks.append({'rect': rect, 'points': points, 'color': color})

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Paddle movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        # Ball launch
        if self.ball_attached and space_held:
            # SFX: Ball launch
            self.ball_attached = False
            angle = (self.np_random.uniform(-math.pi/4, math.pi/4))
            self.ball_vel = np.array([math.sin(angle), -math.cos(angle)]) * self.BALL_SPEED_INITIAL

    def _update_physics(self):
        if self.ball_attached:
            self.ball_pos[0] = self.paddle.centerx
        else:
            self.ball_pos += self.ball_vel

            # Wall collisions
            if self.ball_pos[0] - self.BALL_RADIUS <= 0 or self.ball_pos[0] + self.BALL_RADIUS >= self.SCREEN_WIDTH:
                self.ball_vel[0] *= -1
                self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
                # SFX: Wall bounce
            if self.ball_pos[1] - self.BALL_RADIUS <= 0:
                self.ball_vel[1] *= -1
                self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.SCREEN_HEIGHT - self.BALL_RADIUS)
                # SFX: Wall bounce

            # Paddle collision
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
                # SFX: Paddle bounce
                self.ball_vel[1] *= -1
                offset = (self.paddle.centerx - self.ball_pos[0]) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] = -offset * self.BALL_SPEED_MAX_X
                self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS - 1

    def _handle_block_collisions(self):
        reward = 0
        if self.ball_attached:
            return 0

        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                # SFX: Block break
                self.score += block['points']
                reward += block['points']
                self._create_particles(block['rect'].center, block['color'])
                self.blocks.remove(block)

                # Determine collision side to correctly reflect the ball
                # A simple but effective method: check overlap amount
                overlap_x = (ball_rect.width / 2 + block['rect'].width / 2) - abs(ball_rect.centerx - block['rect'].centerx)
                overlap_y = (ball_rect.height / 2 + block['rect'].height / 2) - abs(ball_rect.centery - block['rect'].centery)

                if overlap_x < overlap_y:
                    self.ball_vel[0] *= -1 # Horizontal collision
                else:
                    self.ball_vel[1] *= -1 # Vertical collision
                break # Only handle one block collision per frame
        
        return reward

    def _check_ball_loss(self):
        return self.ball_pos[1] - self.BALL_RADIUS > self.SCREEN_HEIGHT

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw blocks with outline
        for block in self.blocks:
            darker_color = tuple(c * self.BLOCK_OUTLINE_DARKEN_FACTOR for c in block['color'])
            pygame.draw.rect(self.screen, darker_color, block['rect'].inflate(4, 4))
            pygame.draw.rect(self.screen, block['color'], block['rect'])

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifespan'] / 40))
            size = max(1, int(p['lifespan'] / 10))
            pygame.draw.circle(self.screen, p['color'] + (alpha,), [int(p['pos'][0]), int(p['pos'][1])], size)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Draw ball with glow
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 2, self.COLOR_BALL_GLOW + (50,))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)


    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 5))

        # Balls left
        for i in range(self.balls_left - (1 if self.ball_attached else 0)):
             pygame.draw.circle(self.screen, self.COLOR_BALL, (self.SCREEN_WIDTH - 20 - i * 20, 18), 6)

        # Timer bar
        time_ratio = 1.0 - (self.steps / self.MAX_STEPS)
        bar_width = int((self.SCREEN_WIDTH / 2) * time_ratio)
        bar_x = self.SCREEN_WIDTH / 4
        
        timer_color = self.COLOR_TIMER_BAR
        if time_ratio < 0.25:
            timer_color = self.COLOR_TIMER_BAR_DANGER
        elif time_ratio < 0.5:
            timer_color = self.COLOR_TIMER_BAR_WARN

        pygame.draw.rect(self.screen, (50,50,50), (bar_x, 10, self.SCREEN_WIDTH/2, 15))
        pygame.draw.rect(self.screen, timer_color, (bar_x, 10, bar_width, 15))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play ---
    # This part is for human testing and visualization
    # It requires a window, so we'll re-init pygame for display
    pygame.display.init()
    pygame.display.set_caption(GameEnv.game_description)
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Main game loop
    running = True
    while running:
        # Action defaults
        movement = 0 # no-op
        space = 0 # released
        shift = 0 # released

        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        if keys[pygame.K_r]: # Press R to reset
             obs, info = env.reset()
             terminated = False

        # Step the environment
        if not terminated:
            action = np.array([movement, space, shift])
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control frame rate
        env.clock.tick(GameEnv.FPS)

    env.close()