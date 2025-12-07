
# Generated: 2025-08-27T18:37:09.680339
# Source Brief: brief_01894.md
# Brief Index: 1894

        
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
    """
    A fast-paced arcade game based on the classic 'Breakout'.
    The player controls a paddle at the bottom of the screen to bounce a ball
    and destroy a wall of blocks at the top. The game ends if the ball
    falls past the paddle or if all blocks are destroyed.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → arrow keys to move the paddle horizontally."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Bounce a pixelated ball off your paddle to break blocks in this fast-paced arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment, including Pygame, spaces, and game-specific variables.
        """
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.rng = np.random.default_rng()

        # Colors (Dark background, bright interactive elements)
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (200, 200, 220)
        self.COLOR_PADDLE_HILIGHT = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.BLOCK_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 255, 80), (80, 255, 255), (255, 80, 255)
        ]

        # Fonts
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 64, bold=True)

        # Game constants
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_INITIAL_SPEED = 6
        self.BALL_MAX_HORIZONTAL_VEL = 7
        self.MAX_STEPS = 1000

        # Game state variables (initialized in reset)
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.active_blocks = 0
        self.particles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        """
        Resets the game to its initial state.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Paddle state
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 10
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle_rect = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball state
        self.ball_pos = np.array([self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS - 5], dtype=float)
        angle = self.rng.uniform(-math.pi * 0.6, -math.pi * 0.4) # Upwards angle
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=float) * self.BALL_INITIAL_SPEED

        # Blocks state
        self.blocks = []
        num_cols = 12
        num_rows = 5
        block_width = 50
        block_height = 20
        total_block_width = num_cols * block_width
        start_x = (self.WIDTH - total_block_width) / 2
        start_y = 50
        for i in range(num_rows):
            for j in range(num_cols):
                color = self.rng.choice(self.BLOCK_COLORS)
                rect = pygame.Rect(
                    start_x + j * block_width,
                    start_y + i * block_height,
                    block_width - 2,
                    block_height - 2
                )
                self.blocks.append({"rect": rect, "color": color})
        self.active_blocks = len(self.blocks)

        # Other state
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        """
        Advances the game state by one frame based on the given action.
        """
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0.0
        
        if not self.game_over:
            # 1. Handle player input
            movement = action[0]
            if movement == 3:  # Left
                self.paddle_rect.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle_rect.x += self.PADDLE_SPEED
            self.paddle_rect.clamp_ip(self.screen.get_rect())

            # 2. Update ball position
            self.ball_pos += self.ball_vel

            # 3. Handle collisions
            # Walls
            if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
                self.ball_vel[0] *= -1
                self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
                # sfx: wall_bounce.wav
            if self.ball_pos[1] <= self.BALL_RADIUS:
                self.ball_vel[1] *= -1
                self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
                # sfx: wall_bounce.wav
            
            # Paddle
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if self.paddle_rect.colliderect(ball_rect) and self.ball_vel[1] > 0:
                self.ball_vel[1] *= -1
                # Influence horizontal velocity based on where the ball hit the paddle
                offset = (self.ball_pos[0] - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] = np.clip(offset * self.BALL_MAX_HORIZONTAL_VEL, -self.BALL_MAX_HORIZONTAL_VEL, self.BALL_MAX_HORIZONTAL_VEL)
                # Normalize speed to keep it constant
                speed = np.linalg.norm(self.ball_vel)
                if speed > 0:
                    self.ball_vel = (self.ball_vel / speed) * self.BALL_INITIAL_SPEED
                self.ball_pos[1] = self.paddle_rect.top - self.BALL_RADIUS
                # sfx: paddle_bounce.wav

            # Blocks
            for i in range(len(self.blocks) - 1, -1, -1):
                block = self.blocks[i]
                if block["rect"].colliderect(ball_rect):
                    self._create_particles(block["rect"].center, block["color"])
                    self.blocks.pop(i)
                    self.active_blocks -= 1
                    self.ball_vel[1] *= -1 # Simple vertical bounce
                    self.score += 10
                    reward += 1.0
                    # sfx: block_break.wav
                    break # Only break one block per frame
            
            # 4. Check for win/loss conditions
            # Loss condition
            if self.ball_pos[1] > self.HEIGHT + self.BALL_RADIUS:
                self.game_over = True
                reward = -100.0
                # sfx: game_over.wav
            
            # Win condition
            if self.active_blocks == 0 and not self.win:
                self.win = True
                self.game_over = True
                self.score += 500
                reward += 105.0 # +100 for win, +5 for last block
                # sfx: win_game.wav

            if not self.game_over:
                reward += 0.1 # Survival reward

        # 5. Update particles
        self._update_particles()

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
             reward = -100.0 # Penalty for running out of time
             self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, pos, color):
        """Spawns a burst of particles at a given position."""
        for _ in range(15):
            vel = np.array([self.rng.uniform(-3, 3), self.rng.uniform(-3, 3)], dtype=float)
            lifespan = self.rng.integers(10, 20)
            self.particles.append({"pos": np.array(pos, dtype=float), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        """Updates position and lifespan of all active particles."""
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.pop(i)

    def _get_observation(self):
        """
        Renders the current game state to a Pygame surface and returns it as a NumPy array.
        """
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
        """Renders all primary game objects."""
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 20))
            color = (p["color"][0], p["color"][1], p["color"][2], alpha)
            size = max(1, int(p["lifespan"] / 4))
            rect = pygame.Rect(p["pos"][0] - size // 2, p["pos"][1] - size // 2, size, size)
            
            # Create a temporary surface for transparency
            temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, temp_surf.get_rect())
            self.screen.blit(temp_surf, rect.topleft)

        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
        
        # Render paddle with a highlight for a 3D effect
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=5)
        highlight_rect = self.paddle_rect.copy()
        highlight_rect.height = 3
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_HILIGHT, highlight_rect, border_radius=5)
        
        # Render ball (anti-aliased for smoothness)
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        """Renders UI elements like score and game over text."""
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            if self.win:
                end_text = self.font_game_over.render("YOU WIN!", True, self.BLOCK_COLORS[1])
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.BLOCK_COLORS[0])
            
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        """
        Returns a dictionary with auxiliary diagnostic information.
        """
        return {
            "score": self.score,
            "steps": self.steps,
            "ball_pos": self.ball_pos.tolist(),
            "ball_vel": self.ball_vel.tolist(),
            "active_blocks": self.active_blocks,
        }
        
    def close(self):
        """Closes the Pygame window."""
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Breakout")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    running = True
    
    # Action state
    action = np.array([0, 0, 0]) # [movement, space, shift]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Reset actions
            action[0] = 0 # No movement
            
            # Map keys to actions
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Not used in this game, but part of the standard action space
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            # Wait for a moment to show the game over screen, then allow reset
            pygame.time.wait(500)

    env.close()