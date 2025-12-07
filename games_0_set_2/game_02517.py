
# Generated: 2025-08-28T05:08:00.544154
# Source Brief: brief_02517.md
# Brief Index: 2517

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A retro block-breaking game. Clear all 50 blocks to win. You have 3 balls."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 15, 35)
    COLOR_GRID = (30, 30, 50)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (255, 87, 34), (139, 195, 74), (3, 169, 244), (233, 30, 99),
        (156, 39, 176), (0, 150, 136), (255, 235, 59)
    ]

    # Dimensions
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    BALL_RADIUS = 7
    BLOCK_WIDTH, BLOCK_HEIGHT = 40, 20

    # Physics
    PADDLE_SPEED = 12
    BALL_SPEED = 8
    MAX_BOUNCE_ANGLE_FACTOR = 0.8

    # Game Rules
    TOTAL_BLOCKS = 50
    INITIAL_BALLS = 3
    MAX_STEPS = 30 * 120 # 2 minutes at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # State variables are initialized in reset()
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.block_colors = None
        self.particles = None
        self.ball_trail = None
        self.balls_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.balls_left = self.INITIAL_BALLS
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.ball_trail = []

        self._reset_ball()
        self._generate_blocks()

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_launched = False
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )
        self.ball_vel = [0, 0]
        self.ball_trail = []

    def _generate_blocks(self):
        self.blocks = []
        self.block_colors = {}
        
        grid_w, grid_h = 12, 8
        cell_w = self.WIDTH / grid_w
        cell_h = 30
        
        available_positions = []
        for row in range(grid_h):
            for col in range(grid_w):
                if 1 <= col < grid_w - 1: # Avoid side walls
                    available_positions.append((col, row))
        
        chosen_indices = self.np_random.choice(len(available_positions), self.TOTAL_BLOCKS, replace=False)

        for i in chosen_indices:
            col, row = available_positions[i]
            block_rect = pygame.Rect(
                col * cell_w + (cell_w - self.BLOCK_WIDTH) / 2,
                row * cell_h + 50,
                self.BLOCK_WIDTH,
                self.BLOCK_HEIGHT
            )
            self.blocks.append(block_rect)
            self.block_colors[id(block_rect)] = random.choice(self.BLOCK_COLORS)

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = 0
        
        if self.game_over:
            self.steps += 1
            terminated = self._check_termination()
            return self._get_observation(), 0, terminated, False, self._get_info()

        # Unpack actions
        movement = action[0]
        space_held = action[1] == 1

        # Handle input
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.WIDTH, self.paddle.right)

        if space_held and not self.ball_launched:
            # SFX: Ball Launch
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi/4, math.pi/4) # Random upward angle
            self.ball_vel = [self.BALL_SPEED * math.sin(angle), -self.BALL_SPEED * math.cos(angle)]

        # Update game state
        if not self.ball_launched:
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
        else:
            self.ball_trail.append(self.ball.center)
            if len(self.ball_trail) > 10:
                self.ball_trail.pop(0)
            
            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]

            # Collisions
            # Walls
            if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
                self.ball_vel[0] *= -1
                self.ball.left = max(0, self.ball.left)
                self.ball.right = min(self.WIDTH, self.ball.right)
                # SFX: Wall Bounce
            if self.ball.top <= 0:
                self.ball_vel[1] *= -1
                # SFX: Wall Bounce

            # Paddle
            if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
                # SFX: Paddle Hit
                self.ball.bottom = self.paddle.top
                self.ball_vel[1] *= -1

                offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] += offset * self.MAX_BOUNCE_ANGLE_FACTOR * self.BALL_SPEED
                
                # Normalize speed
                speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
                if speed > 0:
                    self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_SPEED
                    self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_SPEED

                # Reward for paddle hit style
                if abs(offset) > 0.7:
                    reward += 0.1 # Risky play
                else:
                    reward -= 0.02 # Safe play

            # Blocks
            hit_block_idx = self.ball.collidelist(self.blocks)
            if hit_block_idx != -1:
                # SFX: Block Break
                block = self.blocks.pop(hit_block_idx)
                reward += 1.0

                # Particle effect
                color = self.block_colors.pop(id(block), self.COLOR_BALL)
                for _ in range(15):
                    p_vel = [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)]
                    p_pos = list(block.center)
                    p_life = self.np_random.uniform(10, 20)
                    self.particles.append({'pos': p_pos, 'vel': p_vel, 'life': p_life, 'color': color})

                # Bounce logic
                self.ball_vel[1] *= -1
                
                if not self.blocks:
                    self.win = True
                    self.game_over = True
                    reward += 100 # Win bonus
                    # SFX: Game Win

            # Out of bounds
            if self.ball.top > self.HEIGHT:
                # SFX: Lose Ball
                self.balls_left -= 1
                self._reset_ball()
                if self.balls_left <= 0:
                    self.game_over = True
                    reward -= 100 # Lose penalty
                    # SFX: Game Over

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        self.steps += 1
        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
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
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _render_game(self):
        # Blocks
        for block in self.blocks:
            color = self.block_colors.get(id(block), self.COLOR_BALL)
            pygame.draw.rect(self.screen, color, block, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), block, width=2, border_radius=3)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Ball trail
        for i, pos in enumerate(self.ball_trail):
            alpha = int(255 * (i / len(self.ball_trail)) * 0.5)
            color = (self.COLOR_BALL[0], self.COLOR_BALL[1], self.COLOR_BALL[2], alpha)
            temp_surf = pygame.Surface((self.BALL_RADIUS*2, self.BALL_RADIUS*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, self.BALL_RADIUS, self.BALL_RADIUS, self.BALL_RADIUS - i//2, color)
            self.screen.blit(temp_surf, (pos[0]-self.BALL_RADIUS, pos[1]-self.BALL_RADIUS))

        # Ball
        if self.balls_left > 0:
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Particles
        for p in self.particles:
            size = int(max(0, p['life'] / 4))
            pygame.draw.rect(self.screen, p['color'], (int(p['pos'][0]-size/2), int(p['pos'][1]-size/2), size, size))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls
        for i in range(self.balls_left - (1 if self.ball_launched else 0)):
            pos_x = self.WIDTH - 20 - i * (self.BALL_RADIUS * 2 + 5)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text = "YOU WIN!" if self.win else "GAME OVER"
        color = (100, 255, 100) if self.win else (255, 100, 100)
        
        game_over_surf = self.font_game_over.render(text, True, color)
        game_over_rect = game_over_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(game_over_surf, game_over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a separate pygame window for rendering
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
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
            # Keyboard controls
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 0 # Unused
            
            action = [movement, space_held, shift_held]
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            # Optional: Add a delay or wait for a key press to reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False
            print(f"Episode finished. Total Reward: {total_reward}")
            total_reward = 0

    env.close()