
# Generated: 2025-08-27T14:37:54.653322
# Source Brief: brief_00742.md
# Brief Index: 742

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Bounce a ball off a paddle to break all the blocks. Clear all three stages to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 12
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    WALL_THICKNESS = 10
    MAX_STEPS = 10000
    TOTAL_STAGES = 3

    # --- Colors ---
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_WALL = (150, 150, 150)
    COLOR_TEXT = (255, 255, 255)
    BLOCK_COLORS = [(255, 50, 50), (50, 255, 50), (50, 50, 255)] # Red, Green, Blue
    
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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # Etc...        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self.balls_left = 3
        self.base_ball_speed = 3.0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = True
        self.blocks = []
        self.particles = []
        self.steps_since_last_hit = 0
        self.rng = None
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        self.balls_left = 3
        self.base_ball_speed = 3.0
        self.particles = []
        
        self._setup_stage()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _setup_stage(self):
        # Reset paddle
        paddle_y = self.SCREEN_HEIGHT - 40
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Reset ball
        self._reset_ball()

        # Generate blocks
        self.blocks = []
        num_cols = 10
        num_rows = 5
        block_width = (self.SCREEN_WIDTH - 2 * self.WALL_THICKNESS) / num_cols
        block_height = 20
        for i in range(num_rows):
            for j in range(num_cols):
                block_x = self.WALL_THICKNESS + j * block_width
                block_y = 50 + i * block_height
                color = self.rng.choice(self.BLOCK_COLORS, axis=0)
                self.blocks.append({"rect": pygame.Rect(block_x, block_y, block_width, block_height), "color": color})

    def _reset_ball(self):
        self.ball_attached = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]
        self.steps_since_last_hit = 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Continuous penalty per step

        # --- 1. Handle Player Input ---
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen boundaries
        self.paddle.x = max(self.WALL_THICKNESS, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH - self.WALL_THICKNESS))

        # Launch ball if attached and space is pressed
        if self.ball_attached and space_held:
            self.ball_attached = False
            # sfx: ball_launch
            speed = self.base_ball_speed + (self.current_stage - 1) * 0.2
            launch_angle = self.rng.uniform(-math.pi * 0.75, -math.pi * 0.25) # Upward cone
            self.ball_vel = [speed * math.cos(launch_angle), speed * math.sin(launch_angle)]
        
        # --- 2. Update Game State ---
        if self.ball_attached:
            self.ball_pos[0] = self.paddle.centerx
        else:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
            self.steps_since_last_hit += 1

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

        # --- 3. Handle Collisions ---
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= self.WALL_THICKNESS or ball_rect.right >= self.SCREEN_WIDTH - self.WALL_THICKNESS:
            self.ball_vel[0] *= -1
            ball_rect.left = max(ball_rect.left, self.WALL_THICKNESS + 1)
            ball_rect.right = min(ball_rect.right, self.SCREEN_WIDTH - self.WALL_THICKNESS - 1)
            self.ball_pos[0] = ball_rect.centerx
            self.steps_since_last_hit = 0
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = 1
            self.ball_pos[1] = ball_rect.centery
            self.steps_since_last_hit = 0
            # sfx: wall_bounce

        # Paddle collision
        if self.paddle.colliderect(ball_rect) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2
            ball_rect.bottom = self.paddle.top
            self.ball_pos[1] = ball_rect.centery
            reward += 0.1
            self.steps_since_last_hit = 0
            # sfx: paddle_bounce

        # Block collisions
        block_rects = [b['rect'] for b in self.blocks]
        hit_block_idx = ball_rect.collidelist(block_rects)
        if hit_block_idx != -1:
            block_data = self.blocks.pop(hit_block_idx)
            # sfx: block_break
            self.score += 1
            reward += 1

            # Create particle effects
            for _ in range(10):
                self.particles.append({
                    'pos': [block_data['rect'].centerx, block_data['rect'].centery],
                    'vel': [self.rng.uniform(-2, 2), self.rng.uniform(-2, 2)],
                    'life': self.rng.integers(10, 20),
                    'color': block_data['color']
                })

            # Determine bounce direction
            prev_ball_rect = ball_rect.move(-self.ball_vel[0], -self.ball_vel[1])
            if prev_ball_rect.centery < block_data['rect'].top or prev_ball_rect.centery > block_data['rect'].bottom:
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1
            self.steps_since_last_hit = 0

        # --- 4. Check Win/Loss/Progression ---
        # Lose a ball if it goes off the bottom
        if ball_rect.top > self.SCREEN_HEIGHT:
            self.balls_left -= 1
            # sfx: lose_life
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
                reward = -100

        # Stage clear
        if not self.blocks and not self.game_over:
            self.current_stage += 1
            reward += 5
            if self.current_stage > self.TOTAL_STAGES:
                self.game_over = True
                reward += 100 # Win bonus
                # sfx: win_game
            else:
                self.base_ball_speed += 0.2
                self._setup_stage()
                # sfx: stage_clear

        # Anti-softlock mechanism
        if self.steps_since_last_hit > 300: # Generous timeout
            self._reset_ball()

        # Termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            reward = -100 # Penalize for timeout
            terminated = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Draw background gradient
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            border_color = tuple(c * 0.7 for c in block['color'])
            pygame.draw.rect(self.screen, border_color, block['rect'], 2)

        # Particles
        for p in self.particles:
            size = max(0, int(p['life'] / 4))
            pygame.draw.rect(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1]), size, size))

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball with glow effect
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        glow_color = (128, 128, 0, 100)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 2, glow_color)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 2, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, 10))

        # Balls
        balls_text = self.font_small.render(f"Balls: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.SCREEN_WIDTH - balls_text.get_width() - self.WALL_THICKNESS - 10, 10))

        # Stage
        stage_text = self.font_small.render(f"Stage: {self.current_stage}/{self.TOTAL_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, ((self.SCREEN_WIDTH - stage_text.get_width()) / 2, 10))

        # Game Over / Win message
        if self.game_over:
            win = self.current_stage > self.TOTAL_STAGES
            msg = "YOU WIN!" if win else "GAME OVER"
            color = (50, 255, 50) if win else (255, 50, 50)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "balls_left": self.balls_left,
        }

    def close(self):
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

# Example usage for visualization and human play
if __name__ == '__main__':
    import os
    # This allows the game to run in environments without a display
    # If you want to see the game, comment this line out.
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    
    # Create a real screen if not in a headless environment
    try:
        real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption(env.game_description)
        is_headless = False
    except pygame.error:
        is_headless = True
        print("Running in headless mode. No display will be shown.")

    obs, info = env.reset()
    done = False
    
    # Manual Control Loop for human play
    action = np.array([0, 0, 0]) # [movement, space, shift]
    
    while not done:
        if not is_headless:
            # Get keyboard input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            
            # Reset action array
            action.fill(0)

            # Map keys to actions
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            else:
                action[0] = 0

            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        else:
            # In headless mode, use random actions
            action = env.action_space.sample()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if not is_headless:
            # Render the observation to the real screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            real_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        # Cap the frame rate
        env.clock.tick(30)

    env.close()
    print(f"Game Over. Final Score: {info['score']}")