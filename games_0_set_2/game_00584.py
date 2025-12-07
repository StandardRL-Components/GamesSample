import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro-arcade block breaker. Clear all the blocks to win, but lose a life if the ball hits the bottom. You have 3 lives."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_MAX_SPEED = 8
        self.INITIAL_LIVES = 3
        self.MAX_STEPS = 10000

        # --- Colors ---
        self.COLOR_BG = (10, 10, 25)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = {
            30: (255, 50, 50),   # Red
            20: (0, 128, 255),  # Blue
            10: (0, 255, 128)   # Green
        }
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- Game State Variables ---
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_on_paddle = None
        self.blocks = None
        self.particles = None
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.steps = 0
        self.game_over = False
        self.particles = []

        # Player paddle
        paddle_y = self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Blocks
        self._create_blocks()

        # Ball
        self._reset_ball()
        
        return self._get_observation(), self._get_info()
    
    def _create_blocks(self):
        self.blocks = []
        block_scores_layout = [30, 30, 20, 20, 10]
        num_rows = 5
        num_cols = 10
        block_width = (self.SCREEN_WIDTH - (num_cols + 1) * 4) / num_cols
        block_height = 20
        top_margin = 50
        
        for i in range(num_rows):
            for j in range(num_cols):
                score = block_scores_layout[i]
                color = self.BLOCK_COLORS[score]
                x = j * (block_width + 4) + 4
                y = i * (block_height + 4) + top_margin
                block_rect = pygame.Rect(x, y, block_width, block_height)
                self.blocks.append({'rect': block_rect, 'score': score, 'color': color})

    def _reset_ball(self):
        self.ball_on_paddle = True
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0
        
        # Move paddle
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            reward -= 0.01 # Small penalty for movement
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            reward -= 0.01

        # Keep paddle in bounds
        self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))

        # Launch ball
        if self.ball_on_paddle and space_held:
            self.ball_on_paddle = False
            # FIX: The low argument must be less than the high argument for uniform.
            # -3*pi/4 is smaller than -pi/4.
            angle = self.np_random.uniform(-3*math.pi/4, -math.pi/4) # Upward cone
            self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * (self.BALL_MAX_SPEED * 0.8)
            # sfx: launch_ball

        # --- Game Logic ---
        if self.ball_on_paddle:
            self.ball_pos.x = self.paddle.centerx
        else:
            self.ball_pos += self.ball_vel

        # Ball collision
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Walls
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.SCREEN_WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.ball_pos.x, self.SCREEN_WIDTH - self.BALL_RADIUS))
            # sfx: wall_bounce
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # sfx: wall_bounce

        # Bottom (lose life)
        if self.ball_pos.y + self.BALL_RADIUS >= self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 10 # Penalty for losing a life
            if self.lives <= 0:
                self.game_over = True
                reward -= 100 # Terminal penalty
            else:
                self._reset_ball()
                # sfx: lose_life

        # Paddle
        if not self.ball_on_paddle and ball_rect.colliderect(self.paddle):
            # Ensure ball is above paddle to prevent multiple collisions
            if self.ball_vel.y > 0:
                self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
                
                # "Game feel" bounce physics
                offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                bounce_angle = offset * (math.pi / 2.5) # Max 72 degrees
                
                new_vel_x = self.BALL_MAX_SPEED * math.sin(bounce_angle)
                new_vel_y = -self.BALL_MAX_SPEED * math.cos(bounce_angle)
                
                self.ball_vel = pygame.Vector2(new_vel_x, new_vel_y)
                # sfx: paddle_hit

        # Blocks
        hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            hit_block = self.blocks[hit_block_idx]
            
            # Determine collision side
            # A simple approach is to check overlap and reverse velocity accordingly
            # This prevents ball getting stuck inside blocks
            overlap_x = (ball_rect.width / 2 + hit_block['rect'].width / 2) - abs(ball_rect.centerx - hit_block['rect'].centerx)
            overlap_y = (ball_rect.height / 2 + hit_block['rect'].height / 2) - abs(ball_rect.centery - hit_block['rect'].centery)

            if overlap_x < overlap_y:
                self.ball_vel.x *= -1
            else:
                self.ball_vel.y *= -1

            self.score += hit_block['score']
            reward += hit_block['score'] / 10.0 # Reward based on block value
            
            # Create particles
            self._create_particles(hit_block['rect'].center, hit_block['color'])
            
            # Check for row clear bonus
            hit_y = hit_block['rect'].y
            self.blocks.pop(hit_block_idx)
            row_cleared = not any(b['rect'].y == hit_y for b in self.blocks)
            if row_cleared:
                reward += 10 # Row clear bonus
                # sfx: row_clear_bonus

            # sfx: block_hit
        
        # Update particles
        self._update_particles()
        
        # --- Termination Conditions ---
        self.steps += 1
        terminated = self.game_over
        
        if not self.blocks: # Win condition
            terminated = True
            reward += 100 # Win bonus
        
        truncated = self.steps >= self.MAX_STEPS
        
        self.game_over = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        if not self.particles:
            return
        
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
        
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        
        # --- Game Elements ---
        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
            
        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Ball (with antialiasing)
        x, y = int(self.ball_pos.x), int(self.ball_pos.y)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 20.0))))
            color_with_alpha = (*p['color'], alpha)
            particle_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(particle_surf, color_with_alpha, particle_surf.get_rect())
            self.screen.blit(particle_surf, (int(p['pos'].x), int(p['pos'].y)))

        # --- UI ---
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        if self.game_over:
            win_text = "YOU WIN!" if not self.blocks else "GAME OVER"
            text_surface = self.font_game_over.render(win_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surface, text_rect)
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks)
        }

    def render(self):
        return self._get_observation()
    
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will create a window and render the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a different screen for display to avoid conflicts with headless rendering
    pygame.display.init() # Re-init display for non-dummy driver
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    
    terminated = False
    truncated = False
    clock = pygame.time.Clock()
    
    while not terminated and not truncated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The environment's internal screen is used for the observation.
        # We need to draw it to the display screen.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Frame Rate ---
        clock.tick(60) # Run at 60 FPS
        
    env.close()
    pygame.quit()