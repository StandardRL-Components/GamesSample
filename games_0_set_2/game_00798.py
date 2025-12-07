import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


# Set the SDL_VIDEODRIVER to "dummy" for headless operation.
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    """
    A fast-paced, procedurally generated block-breaking game with a neon-arcade aesthetic.
    The player controls a paddle to bounce a ball, clearing stages of blocks.
    Risk-taking (hitting the ball with the edge of the paddle) is rewarded.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro neon block-breaker. Clear all the blocks to advance through 3 stages. "
        "Hitting the ball with the edge of the paddle gives you more control and bonus points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    MAX_STAGES = 3
    INITIAL_LIVES = 5
    MAX_STEPS = 10000

    # --- Colors (Neon Palette) ---
    COLOR_BG_DARK = (10, 0, 20)
    COLOR_BG_LIGHT = (40, 0, 60)
    COLOR_PADDLE = (0, 255, 255)  # Cyan
    COLOR_BALL = (255, 255, 255)  # White
    COLOR_BLOCKS = [(255, 0, 255), (255, 255, 0), (0, 255, 0)]  # Magenta, Yellow, Green
    COLOR_TEXT = (220, 220, 220)
    COLOR_GLOW = (200, 200, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 60)

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.ball_trail = deque(maxlen=10)

        self.steps = 0
        self.score = 0
        self.lives = 0
        self.stage = 0
        self.game_over = False
        self.game_won = False
        self.ball_attached = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.stage = 1
        self.game_over = False
        self.game_won = False

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        self.particles.clear()
        self.ball_trail.clear()

        self._setup_stage()

        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Initializes the ball and blocks for the current stage."""
        self.ball_attached = True
        ball_base_speed = 5 + (self.stage - 1) * 0.2
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=np.float64)
        self.ball_vel = np.array([0.0, -ball_base_speed], dtype=np.float64)
        self.ball_trail.clear()

        # Procedurally generate blocks
        self.blocks = []
        block_width, block_height = 40, 15
        rows = 4 + self.stage
        cols = 12
        x_padding = (self.WIDTH - cols * (block_width + 5)) / 2

        for r in range(rows):
            for c in range(cols):
                # Higher chance of block appearing in later stages
                if self.np_random.uniform() < (0.5 + self.stage * 0.15):
                    block_x = x_padding + c * (block_width + 5)
                    block_y = 50 + r * (block_height + 5)
                    color_index = (r + c) % len(self.COLOR_BLOCKS)
                    block_rect = pygame.Rect(block_x, block_y, block_width, block_height)
                    self.blocks.append({"rect": block_rect, "color": self.COLOR_BLOCKS[color_index]})

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if not self.game_over and not self.game_won:
            self._handle_actions(action)
            step_reward = self._update_game_state()
            reward += step_reward

        self.steps += 1
        
        if self.game_over or self.game_won:
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        """Processes player input from the action array."""
        movement = action[0]
        space_held = action[1] == 1

        # Paddle Movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED

        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        # Launch Ball
        if space_held and self.ball_attached:
            self.ball_attached = False
            # Sound: Ball Launch
            initial_vel_x = self.np_random.uniform(-1, 1)
            # Ensure argument to sqrt is non-negative
            sqrt_arg = self.ball_vel[1] ** 2 - initial_vel_x ** 2
            if sqrt_arg < 0:
                sqrt_arg = 0
            initial_vel_y = -math.sqrt(sqrt_arg)
            self.ball_vel = np.array([initial_vel_x, initial_vel_y], dtype=np.float64)

    def _update_game_state(self):
        """Updates positions, handles collisions, and manages game logic."""
        step_reward = 0.0

        if self.ball_attached:
            self.ball_pos[0] = self.paddle.centerx
            return 0.0

        # --- Ball Movement ---
        self.ball_pos += self.ball_vel
        self.ball_trail.append(self.ball_pos.copy())

        # --- Collisions ---
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS,
                                self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            self.ball_vel[1] += self.np_random.uniform(-0.1, 0.1)  # Anti-softlock
            # Sound: Wall Bounce
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
            self.ball_vel[0] += self.np_random.uniform(-0.1, 0.1)  # Anti-softlock
            # Sound: Wall Bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle):
            # Sound: Paddle Hit
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

            offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = 4 * offset  # More control

            # Normalize velocity to maintain speed
            speed = np.linalg.norm(self.ball_vel)
            if speed > 0:
                base_speed = 5 + (self.stage - 1) * 0.2
                self.ball_vel = self.ball_vel / speed * base_speed

            # Reward for risky vs safe play
            if abs(offset) > 0.7:
                step_reward += 0.1  # Risky hit
            else:
                step_reward -= 0.02  # Safe hit

        # Block collisions
        hit_block = ball_rect.collidelistall([b['rect'] for b in self.blocks])
        if hit_block:
            # Sort by index to remove from the end to avoid shifting indices
            for i in sorted(hit_block, reverse=True):
                block = self.blocks.pop(i)
                self.score += 1
                step_reward += 1
                self._create_particles(block["rect"].center, block["color"])
            
            # Simple bounce, only one bounce direction change per frame
            self.ball_vel[1] *= -1
            # Sound: Block Break

        # Ball out of bounds
        if self.ball_pos[1] > self.HEIGHT:
            self.lives -= 1
            # Sound: Life Lost
            if self.lives <= 0:
                self.game_over = True
            else:
                self._setup_stage()  # Resets ball

        # --- Particles ---
        self.particles = [p for p in self.particles if p["life"] > 1]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

        # --- Stage Clear ---
        if not self.blocks and not self.game_won:
            self.stage += 1
            self.score += 10
            step_reward += 10
            # Sound: Stage Clear
            if self.stage > self.MAX_STAGES:
                self.game_won = True
                self.score += 100
                step_reward += 100
            else:
                self._setup_stage()

        return step_reward

    def _create_particles(self, pos, color):
        """Spawns explosion particles."""
        for _ in range(20):
            vel = self.np_random.uniform(-2, 2, size=2)
            self.particles.append({
                "pos": np.array(pos, dtype=np.float64),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

    def _get_observation(self):
        # --- Draw Background ---
        self.screen.fill(self.COLOR_BG_DARK)
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_DARK[0] * (1 - interp) + self.COLOR_BG_LIGHT[0] * interp),
                int(self.COLOR_BG_DARK[1] * (1 - interp) + self.COLOR_BG_LIGHT[1] * interp),
                int(self.COLOR_BG_DARK[2] * (1 - interp) + self.COLOR_BG_LIGHT[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # --- Draw Game Elements ---
        self._render_game_elements()

        # --- Draw UI ---
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game_elements(self):
        # Particles
        for p in self.particles:
            # FIX: Ensure alpha is an integer, as Pygame color tuples require integers.
            alpha = int(max(0, 255 * (p['life'] / 30)))
            p_color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            temp_surf = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, p_color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Ball Trail
        if len(self.ball_trail) > 1:
            for i, pos in enumerate(self.ball_trail):
                alpha = int(200 * (i / len(self.ball_trail)))
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.BALL_RADIUS, (*self.COLOR_BALL, alpha))

        # Blocks
        for block in self.blocks:
            self._draw_neon_rect(self.screen, block["rect"], block["color"], 2)

        # Paddle
        self._draw_neon_rect(self.screen, self.paddle, self.COLOR_PADDLE, 3)

        # Ball
        self._draw_neon_circle(self.screen, (int(self.ball_pos[0]), int(self.ball_pos[1])), self.BALL_RADIUS, self.COLOR_BALL, 4)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Stage
        stage_text = self.font_ui.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, 10))

        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 20, 10))

        # Game Over / Win Message
        if self.game_over:
            msg = self.font_msg.render("GAME OVER", True, self.COLOR_BLOCKS[0])
            self.screen.blit(msg, (self.WIDTH // 2 - msg.get_width() // 2, self.HEIGHT // 2 - msg.get_height() // 2))
        elif self.game_won:
            msg = self.font_msg.render("YOU WIN!", True, self.COLOR_PADDLE)
            self.screen.blit(msg, (self.WIDTH // 2 - msg.get_width() // 2, self.HEIGHT // 2 - msg.get_height() // 2))

    def _draw_neon_rect(self, surface, rect, color, glow_size):
        """Draws a rectangle with a glowing effect."""
        # Using gfxdraw for alpha blending on the main surface
        for i in range(glow_size, 0, -1):
            alpha = int(80 / (i + 1))
            glow_rect = rect.inflate(i * 2, i * 2)
            pygame.gfxdraw.box(surface, glow_rect, (*color, alpha))
            
        pygame.draw.rect(surface, color, rect, border_radius=3)
        pygame.draw.rect(surface, (255, 255, 255), rect.inflate(-4, -4), 1, border_radius=2)

    def _draw_neon_circle(self, surface, pos, radius, color, glow_size):
        """Draws a circle with a glowing effect."""
        for i in range(glow_size, 0, -1):
            alpha = int(100 / (i + 1))
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius + i, (*color, alpha))
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
        }

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # Set Pygame to use a visible display for testing
    os.environ['SDL_VIDEODRIVER'] = 'x11'  # Use 'windows' on Windows, 'x11' on Linux, 'quartz' on Mac

    env = GameEnv(render_mode="rgb_array")

    # Pygame setup for display
    pygame.display.init()
    screen_display = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Neon Breakout")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False

    print(env.user_guide)

    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0  # no-op
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0  # Not used in this game

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Display ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & FPS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(30)  # Run at 30 FPS

    env.close()
    print(f"Game Over! Final Score: {info['score']}")