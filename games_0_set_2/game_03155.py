import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use ← and → to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Clear all the blocks by bouncing the ball off your paddle. "
        "Some blocks are worth more points than others!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and game dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.WALL_THICKNESS = 10
        self.MAX_STEPS = 2000

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_WALL = (180, 180, 200)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = {
            10: (50, 205, 50),   # Green
            20: (65, 105, 225),  # Blue
            30: (220, 20, 60),   # Red
        }

        # Game entity properties
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 7.0

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.balls_left = None
        self.steps = None
        self.score = None
        self.initial_score = 0
        self.game_over = None
        
        # This will be populated by reset()
        self.np_random = None
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        
        # Paddle
        paddle_y = self.HEIGHT - 40
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball
        self._reset_ball()

        # Blocks
        self._generate_blocks()
        self.initial_score = sum(b['points'] for b in self.blocks)

        # Particles
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = 0.0
        paddle_moved = False

        # 1. Handle player input
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            paddle_moved = True
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            paddle_moved = True
        
        # Keep paddle within walls
        self.paddle.x = max(self.WALL_THICKNESS, self.paddle.x)
        self.paddle.x = min(self.WIDTH - self.WALL_THICKNESS - self.PADDLE_WIDTH, self.paddle.x)
        
        if space_held and not self.ball_launched:
            self._launch_ball()
            # sfx: launch_ball.wav
        
        # 2. Update game state
        if self.ball_launched:
            if not paddle_moved:
                reward -= 0.02 # Penalty for inaction

            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
            
            # Collision detection
            hit_reward, block_destroyed = self._handle_collisions()
            reward += hit_reward
            if block_destroyed:
                reward += 0.1 # Continuous feedback for any block hit
        
        self._update_particles()
        
        # 3. Check for termination
        self.steps += 1
        terminated = self._check_termination()
        
        if self.game_over and self.balls_left > 0: # Win condition
            reward += 50
        
        # 4. Return 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _reset_ball(self):
        """Resets the ball to be on the paddle, unlaunched."""
        self.ball_launched = False
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _launch_ball(self):
        """Launches the ball from the paddle."""
        self.ball_launched = True
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25) # Upward cone
        self.ball_vel = [
            self.BALL_SPEED * math.cos(angle),
            self.BALL_SPEED * math.sin(angle)
        ]

    def _generate_blocks(self):
        """Procedurally generates the grid of blocks."""
        self.blocks = []
        block_width, block_height = 40, 20
        gap = 5
        rows = 6
        cols = 14
        start_x = (self.WIDTH - (cols * (block_width + gap))) / 2
        start_y = 50
        
        for r in range(rows):
            for c in range(cols):
                # Randomly skip some blocks to create patterns
                if self.np_random.random() > 0.15:
                    points = self.np_random.choice(list(self.BLOCK_COLORS.keys()))
                    color = self.BLOCK_COLORS[points]
                    rect = pygame.Rect(
                        start_x + c * (block_width + gap),
                        start_y + r * (block_height + gap),
                        block_width,
                        block_height
                    )
                    self.blocks.append({'rect': rect, 'points': points, 'color': color})

    def _handle_collisions(self):
        """Handles all ball collisions and returns reward from hits."""
        reward = 0
        block_destroyed = False
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        
        # Wall collisions
        if ball_rect.left <= self.WALL_THICKNESS:
            self.ball_vel[0] *= -1
            ball_rect.left = self.WALL_THICKNESS
            # sfx: bounce_wall.wav
        if ball_rect.right >= self.WIDTH - self.WALL_THICKNESS:
            self.ball_vel[0] *= -1
            ball_rect.right = self.WIDTH - self.WALL_THICKNESS
            # sfx: bounce_wall.wav
        if ball_rect.top <= self.WALL_THICKNESS:
            self.ball_vel[1] *= -1
            ball_rect.top = self.WALL_THICKNESS
            # sfx: bounce_wall.wav
        
        # Bottom wall (lose a ball)
        if ball_rect.top >= self.HEIGHT:
            self.balls_left -= 1
            reward -= 1.0
            # sfx: lose_ball.wav
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
            return reward, block_destroyed
        
        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            angle = math.pi/2 - offset * (math.pi/3) # Reflect based on hit location
            
            self.ball_vel[0] = self.BALL_SPEED * -math.cos(angle)
            self.ball_vel[1] = -self.BALL_SPEED * math.sin(angle)
            
            ball_rect.bottom = self.paddle.top - 1
            # sfx: bounce_paddle.wav

            # Anti-stuck mechanism: if ball gets too horizontal, give it some vertical speed
            if abs(self.ball_vel[1]) < 0.1 * self.BALL_SPEED:
                self.ball_vel[1] = -0.1 * self.BALL_SPEED * np.sign(self.ball_vel[1] or -1)


        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                reward += block['points'] / 10.0
                self._create_particles(block['rect'].center, block['color'])
                self.blocks.remove(block)
                block_destroyed = True
                # sfx: break_block.wav

                # Determine bounce direction
                # A simple vertical bounce is most common in these games
                self.ball_vel[1] *= -1
                break # Only handle one block collision per frame

        self.ball_pos = [ball_rect.centerx, ball_rect.centery]
        return reward, block_destroyed

    def _create_particles(self, pos, color):
        """Create a burst of particles."""
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(10, 20),
                'color': color
            })
    
    def _update_particles(self):
        """Update position and life of all particles."""
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        """Check for game termination conditions."""
        if not self.blocks: # All blocks destroyed
            self.game_over = True
        if self.balls_left <= 0:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

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
        """Renders all the game elements."""
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            
        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Particles
        for p in self.particles:
            radius = int(p['life'] / 4)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])

    def _render_ui(self):
        """Renders the UI elements like score and balls left."""
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.HEIGHT - 35))

        balls_text = self.font_large.render(f"Balls: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - balls_text.get_width() - self.WALL_THICKNESS - 10, self.HEIGHT - 35))

        if not self.ball_launched and self.balls_left > 0:
            launch_text = self.font_small.render("Press SPACE to launch", True, self.COLOR_TEXT)
            self.screen.blit(launch_text, (self.paddle.centerx - launch_text.get_width() / 2, self.paddle.y - 30))

        if self.game_over:
            status_text = "YOU WIN!" if not self.blocks else "GAME OVER"
            end_text = self.font_large.render(status_text, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        # The total score is the points from the blocks that have been destroyed.
        current_block_points = sum(b['points'] for b in self.blocks)
        self.score = self.initial_score - current_block_points

        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks)
        }

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
        # Need to reset to initialize everything before getting an observation
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test reset
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)