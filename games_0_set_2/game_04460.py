
# Generated: 2025-08-28T02:28:29.772446
# Source Brief: brief_04460.md
# Brief Index: 4460

        
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
        "Controls: ←→ to move the paddle. Clear all the blocks to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A vibrant, retro-inspired block breaker. Use the paddle to bounce the ball, "
        "destroying all the blocks on the screen. Different colored blocks award different points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For auto-advance clock
        self.MAX_STEPS = 4500 # 2.5 minutes at 30fps

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 55)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.BLOCK_DEFINITIONS = {
            1: {'color': (0, 200, 100), 'points': 1},  # Green
            3: {'color': (0, 150, 255), 'points': 3},  # Blue
            5: {'color': (255, 80, 80), 'points': 5},    # Red
        }

        # Game parameters
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.INITIAL_LIVES = 3
        
        self.GRID_ROWS = 10
        self.GRID_COLS = 10
        self.BLOCK_AREA_TOP = 50
        self.BLOCK_WIDTH = self.WIDTH // self.GRID_COLS
        self.BLOCK_HEIGHT = 18

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # --- Game State Attributes ---
        self.paddle = None
        self.ball = None
        self.ball_velocity = None
        self.blocks = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        self.ball_launch_timer = 0
        self.blocks_destroyed_count = 0
        self.total_blocks = 0
        
        # Initialize state variables
        self.reset()

        # Validate implementation after setup
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.INITIAL_LIVES
        self.blocks_destroyed_count = 0
        self.particles = []

        # Paddle
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 10
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        # Blocks
        self.blocks = []
        block_types = list(self.BLOCK_DEFINITIONS.keys())
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Simple pattern for visual appeal and predictability
                if r < 2: points = 5 # Red
                elif r < 6: points = 3 # Blue
                else: points = 1 # Green

                block_rect = pygame.Rect(
                    c * self.BLOCK_WIDTH,
                    self.BLOCK_AREA_TOP + r * self.BLOCK_HEIGHT,
                    self.BLOCK_WIDTH,
                    self.BLOCK_HEIGHT
                )
                self.blocks.append({
                    'rect': block_rect,
                    'points': points,
                    'color': self.BLOCK_DEFINITIONS[points]['color']
                })
        self.total_blocks = len(self.blocks)
        
        # Ball
        self._spawn_ball()
        
        return self._get_observation(), self._get_info()
    
    def _spawn_ball(self):
        """Resets the ball's position and state, attaching it to the paddle."""
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top
        self.ball_velocity = [0, 0]
        self.ball_launch_timer = 90  # Wait 3 seconds (90 frames) before auto-launch

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        # --- Action Handling ---
        movement = action[0]
        
        if not self.game_over:
            # Move paddle
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED

            # Clamp paddle to screen
            self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH))

            # If ball is waiting to launch, keep it on the paddle
            if self.ball_launch_timer > 0:
                self.ball.centerx = self.paddle.centerx

        # --- Update Game Logic ---
        reward = self._update_game_state()
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_game_state(self):
        """Main logic update function, called every step."""
        if self.game_over:
            return 0.0

        step_reward = -0.01  # Small penalty for each step to encourage speed

        # --- Ball Launch Timer ---
        if self.ball_launch_timer > 0:
            self.ball_launch_timer -= 1
            if self.ball_launch_timer == 0:
                # Launch the ball
                initial_angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
                speed = self._get_current_ball_speed()
                self.ball_velocity = [
                    speed * math.cos(initial_angle),
                    speed * math.sin(initial_angle)
                ]
                # sfx: ball_launch

        # --- Ball Movement ---
        self.ball.x += self.ball_velocity[0]
        self.ball.y += self.ball_velocity[1]

        # --- Collisions ---
        # Walls
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_velocity[0] *= -1
            self.ball.x = max(0, min(self.ball.x, self.WIDTH - self.ball.width))
            # sfx: wall_bounce
        if self.ball.top <= 0:
            self.ball_velocity[1] *= -1
            self.ball.y = max(0, self.ball.y)
            # sfx: wall_bounce

        # Bottom of screen (lose life)
        if self.ball.top >= self.HEIGHT:
            self.lives -= 1
            # sfx: lose_life
            if self.lives <= 0:
                self.game_over = True
                return -100.0  # Large penalty for losing
            else:
                self._spawn_ball()
                return -10.0 # Penalty for losing a life

        # Paddle
        if self.ball.colliderect(self.paddle) and self.ball_velocity[1] > 0:
            self.ball.bottom = self.paddle.top - 1
            
            # Calculate bounce angle based on impact point
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            # Max angle is ~75 degrees (1.3 radians)
            bounce_angle = offset * 1.3
            
            speed = self._get_current_ball_speed()
            self.ball_velocity[0] = speed * math.sin(bounce_angle)
            self.ball_velocity[1] = -speed * math.cos(bounce_angle)
            # sfx: paddle_bounce
        
        # Blocks
        hit_block_this_step = False
        for block in self.blocks[:]:
            if self.ball.colliderect(block['rect']):
                # sfx: block_break
                self.blocks.remove(block)
                self.blocks_destroyed_count += 1
                
                # Add score and reward
                step_reward += block['points']
                self.score += block['points'] * 10 # Scale score for display
                hit_block_this_step = True

                # Create particles
                self._create_particles(block['rect'].center, block['color'])
                
                # Determine bounce direction
                # A simple approximation: check which side was hit most
                overlap = self.ball.clip(block['rect'])
                if overlap.width < overlap.height:
                    self.ball_velocity[0] *= -1
                else:
                    self.ball_velocity[1] *= -1
                
                # Check for win condition
                if not self.blocks:
                    self.game_over = True
                    return 100.0 + step_reward # Large reward for winning
                
                break # Only handle one block collision per frame

        if hit_block_this_step:
            step_reward += 0.1

        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        return step_reward

    def _get_current_ball_speed(self):
        """Calculates ball speed based on blocks destroyed."""
        base_speed = 5.0
        speed_increase = (self.blocks_destroyed_count // 20) * 0.5
        return min(base_speed + speed_increase, 10.0)

    def _create_particles(self, pos, color):
        """Spawns explosion particles."""
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'color': color,
                'size': self.np_random.integers(2, 5)
            })
    
    def _get_observation(self):
        # --- Main Rendering ---
        # Background
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, self.BLOCK_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.BLOCK_AREA_TOP), (x, self.HEIGHT))
        for y in range(self.BLOCK_AREA_TOP, self.HEIGHT, self.BLOCK_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, self.COLOR_BG, block['rect'], 1)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_center = self.ball.center
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_BALL, 50))
        self.screen.blit(glow_surf, (glow_center[0] - glow_radius, glow_center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size'], p['size']), pygame.SRCALPHA)
            temp_surf.fill(color)
            self.screen.blit(temp_surf, (int(p['pos'][0]), int(p['pos'][1])))
        
        # UI Overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_main.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 150, 10))
        for i in range(self.lives):
            heart_pos = (self.WIDTH - 60 + (i * 20), 18)
            pygame.gfxdraw.filled_circle(self.screen, heart_pos[0], heart_pos[1], 7, (255, 50, 50))
            pygame.gfxdraw.aacircle(self.screen, heart_pos[0], heart_pos[1], 7, (255, 50, 50))

        # Blocks remaining
        blocks_left = len(self.blocks)
        blocks_text = self.font_small.render(f"{blocks_left}/{self.total_blocks} BLOCKS", True, self.COLOR_TEXT)
        text_rect = blocks_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 25))
        self.screen.blit(blocks_text, text_rect)

        # Launch timer
        if self.ball_launch_timer > 0:
            timer_val = math.ceil(self.ball_launch_timer / self.FPS)
            timer_text = self.font_main.render(str(timer_val), True, self.COLOR_TEXT)
            text_rect = timer_text.get_rect(center=self.ball.center)
            self.screen.blit(timer_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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


# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        import os
        # Set a non-dummy video driver for human rendering
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode=render_mode)

    if render_mode == "human":
        # For human play, we need a display
        pygame.display.set_caption("Block Breaker")
        human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    done = False
    
    # Simple human player controller
    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
        else: # Random agent for "rgb_array" mode
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if render_mode == "human":
            # Blit the observation from the environment's buffer to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()

        if done:
            print(f"Game Over! Final Info: {info}")

    env.close()