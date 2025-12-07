
# Generated: 2025-08-27T15:47:15.273756
# Source Brief: brief_01076.md
# Brief Index: 1076

        
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

    user_guide = (
        "Controls: ←→ to move the paddle. Press Space to launch the ball."
    )

    game_description = (
        "A retro block-breaking game. Clear all the colored blocks by bouncing "
        "the ball with your paddle. You have three balls per game."
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = {
        1: (0, 200, 100),  # Green
        2: (100, 150, 255), # Blue
        3: (255, 100, 100), # Red
    }

    # Game parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 7
    BALL_SPEED_MAGNITUDE = 7
    MAX_STEPS = 3000 # Increased from 1000 to allow more time for clearing levels
    INITIAL_LIVES = 3

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Particle system
        self.particles = []

        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.game_won = False
        self.reward_this_step = 0

        # Paddle state
        self.paddle = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT * 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Ball state
        self.ball_attached = True
        self._reset_ball()

        # Block state
        self._generate_blocks()
        self.total_blocks = len(self.blocks)
        self.blocks_cleared = 0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0.01 # Small reward for surviving a step

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        self._handle_input(movement, space_pressed)
        self._update_game_logic()
        
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.game_won:
                self.reward_this_step += 50
            else: # Lost all lives or timed out
                self.reward_this_step -= 50

        # Enforce reward scale
        reward = np.clip(self.reward_this_step, -100, 100)
        
        if self.auto_advance:
            self.clock.tick(30)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        # Paddle Movement (adapted from MultiDiscrete)
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))

        # Launch Ball
        if space_pressed and self.ball_attached:
            self.ball_attached = False
            # sfx: launch_ball
            # Give it an initial upward velocity, with a slight random horizontal component
            self.ball_vel = [self.np_random.uniform(-1, 1), -1]
            self._normalize_ball_velocity()

    def _update_game_logic(self):
        if self.ball_attached:
            # Ball follows the paddle
            self.ball_pos[0] = self.paddle.centerx
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
        else:
            # Move ball
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
            self._handle_collisions()
        
        # Update particles
        self._update_particles()

    def _handle_collisions(self):
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            # sfx: wall_bounce

        # Bottom wall (lose life)
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            self.reward_this_step -= 5 # Penalty for losing a ball
            # sfx: ball_loss
            if self.lives > 0:
                self.ball_attached = True
                self._reset_ball()
            else:
                self.game_over = True

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle_hit
            self.ball_vel[1] *= -1
            
            # Change horizontal velocity based on where it hit the paddle
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = offset * 2.5 # Give more horizontal influence
            self._normalize_ball_velocity()
            # Move ball slightly out of paddle to prevent sticking
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS 

        # Block collisions
        for block in self.blocks:
            if block['active'] and ball_rect.colliderect(block['rect']):
                # sfx: block_hit
                block['active'] = False
                self.blocks_cleared += 1
                
                # Add reward based on block value
                points = block['points']
                self.score += points
                self.reward_this_step += points

                self._create_particles(block['rect'].center, self.BLOCK_COLORS[points])

                # Determine bounce direction
                prev_ball_rect = pygame.Rect(ball_rect.x - self.ball_vel[0], ball_rect.y - self.ball_vel[1], ball_rect.width, ball_rect.height)
                
                if prev_ball_rect.bottom <= block['rect'].top or prev_ball_rect.top >= block['rect'].bottom:
                    self.ball_vel[1] *= -1
                else:
                    self.ball_vel[0] *= -1
                
                # Exit loop after one block collision per frame
                break

    def _generate_blocks(self):
        self.blocks = []
        block_width = 50
        block_height = 20
        gap = 4
        rows = 5
        cols = self.SCREEN_WIDTH // (block_width + gap)
        
        start_x = (self.SCREEN_WIDTH - cols * (block_width + gap) + gap) // 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                # Randomly skip some blocks for variety
                if self.np_random.random() > 0.15:
                    points = self.np_random.choice([1, 1, 1, 2, 2, 3])
                    rect = pygame.Rect(
                        start_x + c * (block_width + gap),
                        start_y + r * (block_height + gap),
                        block_width,
                        block_height
                    )
                    self.blocks.append({'rect': rect, 'points': points, 'active': True})

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _normalize_ball_velocity(self):
        mag = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
        if mag > 0:
            self.ball_vel[0] = (self.ball_vel[0] / mag) * self.BALL_SPEED_MAGNITUDE
            self.ball_vel[1] = (self.ball_vel[1] / mag) * self.BALL_SPEED_MAGNITUDE

    def _check_termination(self):
        if self.lives <= 0:
            return True
        if self.blocks_cleared >= self.total_blocks and self.total_blocks > 0:
            self.game_won = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Render blocks
        for block in self.blocks:
            if block['active']:
                pygame.draw.rect(self.screen, self.BLOCK_COLORS[block['points']], block['rect'])
                pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.BLOCK_COLORS[block['points']]), block['rect'], 2)

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Render ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)

        # Render particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], (p['pos'][0], p['pos'][1], p['size'], p['size']))

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Lives
        for i in range(self.lives):
            pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH - 20 - i * 25, 22, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH - 20 - i * 25, 22, self.BALL_RADIUS, self.COLOR_BALL)
    
    def _render_game_over(self):
        text = "YOU WIN!" if self.game_won else "GAME OVER"
        text_surf = self.font_game_over.render(text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        # Add a semi-transparent overlay
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_cleared": self.blocks_cleared,
        }

    # --- Particle System ---
    def _create_particles(self, pos, color):
        for _ in range(15):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)],
                'lifespan': self.np_random.integers(10, 20),
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            # Fade color
            p['color'] = tuple(max(0, c - 15) for c in p['color'])
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Override Pygame screen for display
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    
    terminated = False
    total_reward = 0

    # Game loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            # --- Get human input ---
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space, shift]

            # --- Step the environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # --- Render the observation to the display window ---
            # The observation is (H, W, C), but pygame wants (W, H) surface
            # and surfarray.blit_array expects (W, H, C)
            obs_transposed = np.transpose(obs, (1, 0, 2))
            pygame.surfarray.blit_array(env.screen, obs_transposed)
            pygame.display.flip()

        else:
            # If terminated, wait for a key press to reset
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    print(f"Episode finished. Total Reward: {total_reward:.2f}")
                    obs, info = env.reset()
                    terminated = False
                    total_reward = 0

    pygame.quit()
    print(f"Final Score: {total_reward:.2f}")