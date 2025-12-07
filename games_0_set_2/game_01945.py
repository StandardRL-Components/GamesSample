
# Generated: 2025-08-28T03:10:23.191281
# Source Brief: brief_01945.md
# Brief Index: 1945

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the paddle. Break all the blocks to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Control the paddle to bounce the ball, "
        "destroy all the blocks, and achieve a high score. The ball gets faster as you progress!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (35, 35, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (255, 80, 80), (255, 160, 80), (255, 255, 80), (80, 255, 80),
            (80, 255, 255), (80, 80, 255), (160, 80, 255), (255, 80, 255)
        ]

        # Game constants
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 4.0
        self.MAX_BALL_SPEED = 8.0
        self.MAX_STEPS = 10000
        self.NUM_BLOCK_ROWS = 8
        self.NUM_BLOCK_COLS = 10
        self.TOTAL_BLOCKS = self.NUM_BLOCK_ROWS * self.NUM_BLOCK_COLS
        
        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.blocks = None
        self.particles = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.blocks_destroyed_count = None
        
        # Initialize state
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize paddle
        paddle_y = self.SCREEN_HEIGHT - 40
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2, paddle_y,
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        
        # Initialize ball
        self.ball = pygame.Rect(
            self.paddle.centerx - self.BALL_RADIUS,
            self.paddle.top - self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2, self.BALL_RADIUS * 2
        )
        # Random initial horizontal direction
        initial_vx = self.INITIAL_BALL_SPEED * random.choice([-0.707, 0.707])
        initial_vy = -self.INITIAL_BALL_SPEED * 0.707
        self.ball_vel = [initial_vx, initial_vy]

        # Generate blocks
        self._generate_blocks()
        
        # Initialize game state
        self.lives = 3
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.blocks_destroyed_count = 0
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _generate_blocks(self):
        self.blocks = []
        self.block_rows = [self.NUM_BLOCK_COLS] * self.NUM_BLOCK_ROWS
        block_width = 58
        block_height = 18
        gap = 6
        start_x = (self.SCREEN_WIDTH - (self.NUM_BLOCK_COLS * (block_width + gap) - gap)) / 2
        start_y = 50
        
        for i in range(self.NUM_BLOCK_ROWS):
            for j in range(self.NUM_BLOCK_COLS):
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                x = start_x + j * (block_width + gap)
                y = start_y + i * (block_height + gap)
                block_rect = pygame.Rect(x, y, block_width, block_height)
                self.blocks.append({'rect': block_rect, 'color': color, 'row': i})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # Unpack action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # 1. Handle player input
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            reward -= 0.02
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            reward -= 0.02
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle.x))
        
        # 2. Update ball physics
        prev_ball_center = self.ball.center
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]
        
        # 3. Handle collisions
        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.SCREEN_WIDTH, self.ball.right)
            # sfx: wall_bounce
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.top = max(0, self.ball.top)
            # sfx: wall_bounce

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            reward += 0.1
            self.ball.bottom = self.paddle.top
            
            # Change angle based on impact point
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            self.ball_vel[0] = speed * offset
            self.ball_vel[1] *= -1
            # Normalize to maintain speed
            current_speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if current_speed > 0:
                scale = speed / current_speed
                self.ball_vel[0] *= scale
                self.ball_vel[1] *= scale
            # sfx: paddle_hit

        # Block collisions
        hit_index = self.ball.collidelist([b['rect'] for b in self.blocks])
        if hit_index != -1:
            hit_block = self.blocks.pop(hit_index)
            reward += 1.0
            self.blocks_destroyed_count += 1
            
            # Check for row clear
            self.block_rows[hit_block['row']] -= 1
            if self.block_rows[hit_block['row']] == 0:
                reward += 5.0
                # sfx: row_clear_bonus

            # Determine bounce direction
            # A simple method: check overlap delta
            dx = self.ball.centerx - hit_block['rect'].centerx
            dy = self.ball.centery - hit_block['rect'].centery
            w = (self.ball.width + hit_block['rect'].width) / 2
            h = (self.ball.height + hit_block['rect'].height) / 2
            
            if abs(dx) / w > abs(dy) / h:
                self.ball_vel[0] *= -1
                self.ball.x = prev_ball_center[0] - self.BALL_RADIUS
            else:
                self.ball_vel[1] *= -1
                self.ball.y = prev_ball_center[1] - self.BALL_RADIUS

            self._spawn_particles(hit_block['rect'].center, hit_block['color'])
            # sfx: block_break
            
            # Increase ball speed every 20 blocks
            if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 20 == 0:
                speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
                new_speed = min(self.MAX_BALL_SPEED, speed + 0.5)
                if speed > 0:
                    scale = new_speed / speed
                    self.ball_vel[0] *= scale
                    self.ball_vel[1] *= scale

        # Lose a life
        if self.ball.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 50.0
            if self.lives > 0:
                # Reset ball
                self.ball.center = (self.paddle.centerx, self.paddle.top - self.BALL_RADIUS * 2)
                self.ball_vel = [self.INITIAL_BALL_SPEED * random.choice([-0.707, 0.707]), -self.INITIAL_BALL_SPEED * 0.707]
                # sfx: lose_life
            else:
                self.game_over = True
                # sfx: game_over
        
        # 4. Update particles
        self._update_particles()
        
        # 5. Check termination
        terminated = self.game_over
        if not self.blocks: # Win condition
            reward += 100.0
            terminated = True
            self.game_over = True
            # sfx: win_game
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            particle = {
                'pos': list(pos),
                'vel': [random.uniform(-2, 2), random.uniform(-2, 2)],
                'lifespan': random.randint(20, 40),
                'color': color,
                'radius': random.uniform(1, 4)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['radius'] -= 0.05
            if p['lifespan'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render background grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Render game elements
        self._render_game_elements()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game_elements(self):
        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
            # Add a subtle highlight
            highlight_color = tuple(min(255, c + 40) for c in block['color'])
            pygame.draw.rect(self.screen, highlight_color, (block['rect'].x, block['rect'].y, block['rect'].width, 3), border_top_left_radius=3, border_top_right_radius=3)

        # Render paddle with glow
        glow_color = (150, 150, 150)
        for i in range(4, 0, -1):
            glow_rect = self.paddle.inflate(i*2, i*2)
            alpha = 100 - i * 25
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*glow_color, alpha), s.get_rect(), border_radius=5)
            self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Render ball with glow
        ball_center = (int(self.ball.centerx), int(self.ball.centery))
        for i in range(10, 0, -1):
            alpha = int(150 * (1 - i / 10))
            pygame.gfxdraw.filled_circle(self.screen, *ball_center, self.BALL_RADIUS + i, (*self.COLOR_BALL, alpha))
        pygame.gfxdraw.filled_circle(self.screen, *ball_center, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, *ball_center, self.BALL_RADIUS, self.COLOR_BALL)

        # Render particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = int(255 * (p['lifespan'] / 40))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(s, (pos[0] - p['radius'], pos[1] - p['radius']))

    def _render_ui(self):
        # Render score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Render lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 150, 10))
        for i in range(self.lives):
            life_rect = pygame.Rect(self.SCREEN_WIDTH - 70 + i * 20, 12, 15, 15)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_rect, border_radius=3)

        # Render Game Over/Win message
        if self.game_over:
            if not self.blocks:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen_human = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print(env.user_guide)

    while not terminated:
        # Map keyboard inputs to the action space
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_human.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        # Cap the frame rate
        clock.tick(60)

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000)
    env.close()