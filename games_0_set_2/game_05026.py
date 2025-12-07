
# Generated: 2025-08-28T03:45:05.507386
# Source Brief: brief_05026.md
# Brief Index: 5026

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    A fast-paced, top-down block breaker where strategic paddle positioning and
    risk-taking are key to achieving a high score. The player controls a paddle
    at the bottom of the screen to bounce a ball upwards, destroying a field of
    blocks. The game is won by clearing all blocks and lost if the player runs
    out of lives. The ball's speed increases as more blocks are destroyed,
    adding to the challenge.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, arcade-style block breaker. Destroy all the blocks to win, but don't lose the ball!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_PADDLE = (240, 240, 240)
    COLOR_BALL = (255, 255, 0)
    COLOR_WALL = (60, 60, 90)
    COLOR_GRID = (25, 25, 45)
    BLOCK_COLORS = [
        (255, 60, 60), (60, 255, 60), (60, 60, 255),
        (255, 255, 60), (60, 255, 255), (255, 60, 255)
    ]
    COLOR_TEXT = (220, 220, 220)
    COLOR_GLOW = (255, 255, 150)

    # Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_Y_POS = SCREEN_HEIGHT - 40
    BALL_RADIUS = 7
    BLOCK_WIDTH = 58
    BLOCK_HEIGHT = 20
    WALL_THICKNESS = 10

    # Physics
    PADDLE_SPEED = 10
    INITIAL_BALL_SPEED = 5.0
    BALL_SPEED_INCREMENT = 0.5
    MAX_EPISODE_STEPS = 10000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables are initialized in reset()
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        self.particles = None
        self.blocks_destroyed_count = None
        self.current_ball_speed = None

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize paddle
        self.paddle = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.PADDLE_Y_POS,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Initialize ball
        self.ball_launched = False
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1], dtype=np.float64)
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float64)
        
        # Initialize blocks (50 total)
        self.blocks = []
        num_rows = 5
        num_cols = 10
        block_area_top = 60
        for i in range(num_rows):
            for j in range(num_cols):
                block_x = self.WALL_THICKNESS + j * (self.BLOCK_WIDTH + 2) + 20
                block_y = block_area_top + i * (self.BLOCK_HEIGHT + 2)
                block_rect = pygame.Rect(block_x, block_y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                color = self.BLOCK_COLORS[(i + j) % len(self.BLOCK_COLORS)]
                self.blocks.append({"rect": block_rect, "color": color})

        # Initialize game state
        self.lives = 3
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.blocks_destroyed_count = 0
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        
        # Particle system
        self.particles = []

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Handle Player Input ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED

        # Clamp paddle to screen
        self.paddle.x = max(self.WALL_THICKNESS, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH - self.WALL_THICKNESS))

        # Launch ball
        if space_held and not self.ball_launched:
            # sfx: launch_ball.wav
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25) # Upwards cone
            self.ball_vel = np.array([math.cos(angle), math.sin(angle)]) * self.current_ball_speed

        # --- Update Game Logic ---
        if self.ball_launched:
            reward -= 0.02 # Continuous penalty to encourage action
            self.ball_pos += self.ball_vel
        else:
            # Ball follows paddle before launch
            self.ball_pos[0] = self.paddle.centerx

        # --- Collision Detection ---
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= self.WALL_THICKNESS and self.ball_vel[0] < 0:
            self.ball_vel[0] *= -1
            ball_rect.left = self.WALL_THICKNESS
            # sfx: wall_bounce.wav
        if ball_rect.right >= self.SCREEN_WIDTH - self.WALL_THICKNESS and self.ball_vel[0] > 0:
            self.ball_vel[0] *= -1
            ball_rect.right = self.SCREEN_WIDTH - self.WALL_THICKNESS
            # sfx: wall_bounce.wav
        if ball_rect.top <= self.WALL_THICKNESS and self.ball_vel[1] < 0:
            self.ball_vel[1] *= -1
            ball_rect.top = self.WALL_THICKNESS
            # sfx: wall_bounce.wav

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle_bounce.wav
            self.ball_vel[1] *= -1
            ball_rect.bottom = self.paddle.top
            # Add "spin" based on hit location for more control
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.0
            # Re-normalize speed
            norm = np.linalg.norm(self.ball_vel)
            if norm > 0:
                self.ball_vel = self.ball_vel / norm * self.current_ball_speed
            
        # Block collisions
        hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            # sfx: block_break.wav
            hit_block = self.blocks.pop(hit_block_idx)
            reward += 1.0
            self.score += 10
            
            # Create particles for visual feedback
            self._create_particles(hit_block['rect'].center, hit_block['color'])
            
            # Determine bounce direction (simple AABB)
            prev_ball_center = self.ball_pos - self.ball_vel
            if (prev_ball_center[1] < hit_block['rect'].top or prev_ball_center[1] > hit_block['rect'].bottom):
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1
                
            # Speed up ball every 10 blocks
            self.blocks_destroyed_count += 1
            if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 10 == 0:
                self.current_ball_speed += self.BALL_SPEED_INCREMENT
                norm = np.linalg.norm(self.ball_vel)
                if norm > 0:
                    self.ball_vel = self.ball_vel / norm * self.current_ball_speed

        # Ball out of bounds (lose life)
        if ball_rect.top >= self.SCREEN_HEIGHT:
            # sfx: lose_life.wav
            self.lives -= 1
            reward -= 10.0
            self.ball_launched = False
            self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1], dtype=np.float64)
            self.ball_vel = np.array([0.0, 0.0])
            # Reset speed based on progress
            self.current_ball_speed = self.INITIAL_BALL_SPEED + (self.blocks_destroyed_count // 10) * self.BALL_SPEED_INCREMENT

        # Update particles
        self._update_particles()
        
        # --- Termination Check ---
        self.steps += 1
        terminated = False
        if self.lives <= 0:
            self.game_over = True
            terminated = True
            reward -= 100.0
            self.win = False
            # sfx: game_over.wav
        elif not self.blocks:
            self.game_over = True
            terminated = True
            reward += 100.0
            self.win = True
            # sfx: win_game.wav
        elif self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over = True

        # Update ball position from rect after collisions
        self.ball_pos[0] = ball_rect.centerx
        self.ball_pos[1] = ball_rect.centery

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=2)
            pygame.draw.rect(self.screen, tuple(min(255, c*0.7) for c in block['color']), block['rect'], 2)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        pygame.draw.rect(self.screen, (255,255,255), self.paddle.inflate(-4, -4), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle.inflate(-8, -8), border_radius=3)

        # Draw ball with glow
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        # Glow effect using alpha blending
        for i in range(self.BALL_RADIUS, 0, -2):
            alpha = 100 - (i / self.BALL_RADIUS * 100)
            pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + i, (*self.COLOR_GLOW, int(alpha)))
        # Main ball with anti-aliasing
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            alpha = p['alpha'] * (p['life'] / p['max_life'])
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 15, self.SCREEN_HEIGHT - 35))

        # Lives
        lives_text = self.font_main.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - self.WALL_THICKNESS - 15, self.SCREEN_HEIGHT - 35))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
        }

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': np.array(pos, dtype=np.float64),
                'vel': vel,
                'life': life,
                'max_life': life,
                'size': self.np_random.uniform(2, 5),
                'color': color,
                'alpha': self.np_random.uniform(150, 255)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping effect
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # This part is for human testing and requires a display.
    # It will not run in a headless environment without a virtual display.
    try:
        import os
        # Try to set a display driver, may not be necessary on all systems
        if os.name == 'posix' and "DISPLAY" not in os.environ:
             os.environ["SDL_VIDEODRIVER"] = "dummy"
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption(env.game_description)
    except pygame.error:
        print("Pygame display could not be initialized. Running in headless mode is supported.")
        print("To play manually, ensure you have a display environment (e.g., a desktop).")
        screen = None

    if screen:
        obs, info = env.reset()
        terminated = False
        clock = pygame.time.Clock()
        print(env.user_guide)

        while not terminated:
            # Action mapping for keyboard
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 0 # Not used in this game

            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)

            # Render to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting game.")
                    obs, info = env.reset()

            clock.tick(60) # Limit to 60 FPS for smooth play

        print(f"Game Over. Final Score: {info['score']}")
        env.close()