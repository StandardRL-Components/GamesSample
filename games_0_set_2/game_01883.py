
# Generated: 2025-08-28T03:00:28.199264
# Source Brief: brief_01883.md
# Brief Index: 1883

        
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
    A fast-paced, top-down block breaker where risk-taking is rewarded.
    The player controls a paddle to bounce a ball and break a field of blocks.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ←→ to move the paddle. Break all the blocks to win!"
    )

    # User-facing description of the game
    game_description = (
        "A retro arcade block-breaker. Break blocks to score points and build combos. "
        "The ball speeds up as you clear the screen. Don't lose all your balls!"
    )

    # Frames auto-advance at a fixed rate for smooth, real-time gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For auto_advance=True, this is the target step rate.
        
        # Colors (Neon Arcade Style)
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_GRID = (20, 30, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_GAMEOVER = (255, 0, 60)
        self.COLOR_WIN = (60, 255, 120)

        # Game Mechanics
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 5.0
        self.MAX_STEPS = 10000
        self.COMBO_WINDOW = 60 # steps (2 seconds at 30fps)

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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_remaining = 0
        self.paddle = None
        self.ball = None
        self.ball_vel = [0, 0]
        self.current_ball_speed = 0
        self.blocks = []
        self.block_colors = []
        self.total_blocks = 0
        self.particles = []
        self.combo_timer = 0
        self.combo_count = 0
        
        # Initialize state variables for the first time
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset all game state ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_remaining = 3
        
        # Paddle
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 10
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        # Blocks
        self._create_blocks()
        
        # Ball
        self._reset_ball()
        
        # Particles & Combos
        self.particles = []
        self.combo_timer = 0
        self.combo_count = 0

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        """Resets the ball's position and velocity."""
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2 - 5,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        # Start ball in a random upward direction
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = [
            math.cos(angle) * self.current_ball_speed,
            -abs(math.sin(angle) * self.current_ball_speed)
        ]

    def _create_blocks(self):
        """Creates the grid of blocks with a rainbow gradient."""
        self.blocks = []
        self.block_colors = []
        block_rows = 5
        block_cols = 10
        self.total_blocks = block_rows * block_cols
        
        block_width = self.WIDTH // block_cols
        block_height = 20
        top_offset = 40
        
        for r in range(block_rows):
            for c in range(block_cols):
                hue = int((r * block_cols + c) * 360 / self.total_blocks)
                color = pygame.Color(0)
                color.hsla = (hue, 100, 50, 100) # Vibrant rainbow
                
                block_rect = pygame.Rect(
                    c * block_width,
                    top_offset + r * block_height,
                    block_width - 1, # Gaps between blocks
                    block_height - 1
                )
                self.blocks.append(block_rect)
                self.block_colors.append(tuple(color)[:3])

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01  # Small penalty per step to encourage speed
        self.steps += 1
        
        # --- 1. Handle Input ---
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)
        
        # --- 2. Update Game Logic ---
        self._update_ball()
        self._update_particles()
        
        # Combo timer
        if self.combo_timer > 0:
            self.combo_timer -= 1
        else:
            self.combo_count = 0
            
        # --- 3. Handle Collisions and Rewards ---
        reward += self._handle_collisions()
        
        # --- 4. Check Termination ---
        win = not self.blocks
        lose = self.balls_remaining <= 0
        timeout = self.steps >= self.MAX_STEPS
        
        terminated = win or lose or timeout
        if terminated:
            self.game_over = True
            if win:
                reward += 100
            if lose:
                reward += -100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        """Updates ball position based on velocity."""
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

    def _handle_collisions(self):
        """Manages all ball collisions and returns resulting rewards."""
        reward = 0

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.x = np.clip(self.ball.x, 0, self.WIDTH - self.ball.width)
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.top = 0

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # Prevent sticking
            self.ball.bottom = self.paddle.top
            
            # Change vertical velocity
            self.ball_vel[1] *= -1
            
            # Change horizontal velocity based on impact point
            dist = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += dist * 2.5 # More control
            
            # Normalize velocity to maintain consistent speed
            self._normalize_ball_velocity()
            
            reward += 0.1
            # # Sound effect placeholder
            # pygame.mixer.Sound("paddle_hit.wav").play()
            
            # Penalty for "safe play" (paddle directly under ball)
            if abs(dist) < 0.1:
                reward -= max(0, self.score * 0.2)


        # Block collisions
        collided_idx = self.ball.collidelist(self.blocks)
        if collided_idx != -1:
            block = self.blocks.pop(collided_idx)
            color = self.block_colors.pop(collided_idx)
            
            self._spawn_particles(block.center, color)
            
            # Determine if hit was horizontal or vertical
            self.ball_vel[1] *= -1 # Simple vertical reflection is most common
            
            # Rewards
            reward += 1
            self.score += 10
            
            # Combo
            if self.combo_timer > 0:
                self.combo_count += 1
                combo_bonus = 5 * self.combo_count
                reward += combo_bonus
                self.score += combo_bonus * 10
            else:
                self.combo_count = 1
            self.combo_timer = self.COMBO_WINDOW
            
            # Speed up
            blocks_broken = self.total_blocks - len(self.blocks)
            self.current_ball_speed += 0.05
            if blocks_broken % 5 == 0:
                self.current_ball_speed += 0.25
            self._normalize_ball_velocity()

            # # Sound effect placeholder
            # pygame.mixer.Sound("block_break.wav").play()

        # Ball loss
        if self.ball.top >= self.HEIGHT:
            self.balls_remaining -= 1
            self.combo_timer = 0
            self.combo_count = 0
            if self.balls_remaining > 0:
                self._reset_ball()
            # # Sound effect placeholder
            # pygame.mixer.Sound("ball_loss.wav").play()

        return reward

    def _normalize_ball_velocity(self):
        """Ensures the ball's speed is constant."""
        norm = math.hypot(*self.ball_vel)
        if norm > 0:
            scale = self.current_ball_speed / norm
            self.ball_vel[0] *= scale
            self.ball_vel[1] *= scale

    def _spawn_particles(self, pos, color):
        """Creates a particle explosion."""
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifetime, 'color': color})

    def _update_particles(self):
        """Updates position and lifetime of all particles."""
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # --- Render all game elements to the screen surface ---
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array (required format)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the main game elements."""
        # Background
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Blocks
        for i, block in enumerate(self.blocks):
            color = self.block_colors[i]
            border_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, color, block)
            pygame.draw.rect(self.screen, border_color, block, 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Ball (with anti-aliasing)
        if self.balls_remaining > 0:
            center = (int(self.ball.centerx), int(self.ball.centery))
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.BALL_RADIUS, self.COLOR_BALL)
            
        # Particles
        for p in self.particles:
            alpha = p['life'] / 30.0
            color = (int(p['color'][0]*alpha), int(p['color'][1]*alpha), int(p['color'][2]*alpha))
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['life'] / 5)
            if radius > 0:
                pygame.draw.circle(self.screen, p['color'], pos, radius)

    def _render_ui(self):
        """Renders UI text on top of the game."""
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        balls_text = self.font_main.render(f"BALLS: {self.balls_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - balls_text.get_width() - 10, 5))

        if self.combo_count > 1:
            combo_text = self.font_main.render(f"x{self.combo_count} COMBO!", True, self.COLOR_PADDLE)
            pos = (self.WIDTH // 2 - combo_text.get_width() // 2, self.HEIGHT - 80)
            self.screen.blit(combo_text, pos)
        
        if self.game_over:
            if not self.blocks: # Win
                end_text = self.font_large.render("YOU WIN!", True, self.COLOR_WIN)
            else: # Lose
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_GAMEOVER)
            
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_remaining": self.balls_remaining,
            "blocks_remaining": len(self.blocks),
            "combo": self.combo_count,
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
    env = GameEnv(render_mode="rgb_array")
    
    # --- For human play ---
    # This setup allows playing the game with keyboard controls.
    # Note: Gymnasium's auto_advance=True means we must call step continuously.
    
    obs, info = env.reset()
    terminated = False
    
    # Set up a window to display the game
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    # Game loop for human play
    running = True
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Unused actions for this game
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Reset if the episode is over
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()

        # Control the frame rate
        env.clock.tick(env.FPS)

    env.close()