
# Generated: 2025-08-27T21:53:53.668026
# Source Brief: brief_02938.md
# Brief Index: 2938

        
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
        "Controls: Use ← and → to move the paddle. Your goal is to break all the blocks with the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric block-breaker. Bounce the ball off your paddle to destroy all the blocks. "
        "Don't let the ball fall past your paddle, or you'll lose a life!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_PADDLE = (230, 230, 255)
    COLOR_PADDLE_TOP = (255, 255, 255)
    COLOR_PADDLE_SIDE = (180, 180, 210)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_SHADOW = (0, 0, 0, 100)
    COLOR_TEXT = (255, 255, 255)
    BLOCK_COLORS = [
        ((80, 120, 220), (120, 160, 255), (60, 90, 180)), # Blue (1 pt)
        ((80, 220, 120), (120, 255, 160), (60, 180, 90)), # Green (2 pts)
        ((220, 80, 120), (255, 120, 160), (180, 60, 90)), # Red (3 pts)
    ]

    # Game parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 8
    PADDLE_Y = 360
    BALL_RADIUS = 8
    BALL_SPEED_INITIAL = 5
    MAX_STEPS = 2000
    ISO_DEPTH = 6

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
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        
        self.np_random = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.PADDLE_Y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self._reset_ball()
        self._generate_blocks()
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(60)

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Boolean
        # shift_held = action[2] == 1  # Boolean

        reward = -0.01  # Small penalty per step to encourage efficiency

        if not self.game_over:
            # --- 1. Handle Action ---
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            
            # Clamp paddle to screen
            self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle.x))

            # --- 2. Update Game State ---
            self._update_ball()
            block_reward, life_penalty = self._handle_collisions()
            reward += block_reward + life_penalty
            self._update_particles()
            
            self.steps += 1
        
        # --- 3. Check Termination ---
        terminated = False
        if not self.blocks: # Win condition
            terminated = True
            reward += 50
        elif self.lives <= 0: # Lose condition
            terminated = True
            # The -5 reward for losing the last life is the primary penalty
        elif self.steps >= self.MAX_STEPS: # Timeout
            terminated = True
            reward += -20

        if terminated:
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _generate_blocks(self):
        self.blocks = []
        block_width = 50
        block_height = 20
        rows = 5
        cols = 10
        
        grid_width = cols * (block_width + 5)
        start_x = (self.SCREEN_WIDTH - grid_width) / 2
        
        # Generate 50 blocks
        block_positions = []
        for i in range(rows):
            for j in range(cols):
                block_positions.append((i, j))
        
        chosen_positions = self.np_random.choice(len(block_positions), size=50, replace=False)

        for idx in chosen_positions:
            i, j = block_positions[idx]
            val = self.np_random.integers(1, 4)
            block_rect = pygame.Rect(
                start_x + j * (block_width + 5),
                60 + i * (block_height + 5),
                block_width,
                block_height
            )
            self.blocks.append({
                'rect': block_rect,
                'value': val,
                'colors': self.BLOCK_COLORS[val-1]
            })

    def _update_ball(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = max(self.BALL_RADIUS, min(self.SCREEN_WIDTH - self.BALL_RADIUS, self.ball_pos[0]))
            # sfx: wall_bounce

        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sfx: wall_bounce
        
        # Miss condition
        if self.ball_pos[1] >= self.SCREEN_HEIGHT + self.BALL_RADIUS:
            self.lives -= 1
            # sfx: lose_life
            if self.lives > 0:
                self._reset_ball()

    def _handle_collisions(self):
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS, 
            self.ball_pos[1] - self.BALL_RADIUS, 
            self.BALL_RADIUS * 2, 
            self.BALL_RADIUS * 2
        )
        
        block_reward = 0
        life_penalty = 0

        # Paddle collision
        if self.paddle.colliderect(ball_rect) and self.ball_vel[1] > 0:
            # sfx: paddle_hit
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            self.ball_vel[1] *= -1
            
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            self.ball_vel[0] += offset * 2.0
            
            self.ball_vel[0] = max(-self.BALL_SPEED_INITIAL, min(self.BALL_SPEED_INITIAL, self.ball_vel[0]))
            
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_SPEED_INITIAL
                self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_SPEED_INITIAL

        # Block collisions
        for block in self.blocks[:]:
            if block['rect'].colliderect(ball_rect):
                # sfx: block_break
                block_reward += 0.1
                block_reward += block['value']
                self.score += block['value']
                
                self._create_particles(ball_rect.center, block['colors'][0])
                self.blocks.remove(block)
                
                prev_ball_center_y = (self.ball_pos[1] - self.ball_vel[1])
                if prev_ball_center_y <= block['rect'].top or prev_ball_center_y >= block['rect'].bottom:
                    self.ball_vel[1] *= -1
                else:
                    self.ball_vel[0] *= -1
                
                break

        # Check if ball was missed
        if self.ball_pos[1] >= self.SCREEN_HEIGHT + self.BALL_RADIUS:
            life_penalty = -5

        return block_reward, life_penalty

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        self.ball_vel = [
            self.BALL_SPEED_INITIAL * math.sin(angle),
            -self.BALL_SPEED_INITIAL * math.cos(angle)
        ]

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        for block in self.blocks:
            self._draw_iso_rect(self.screen, block['rect'], block['colors'], self.ISO_DEPTH)

        self._draw_iso_rect(self.screen, self.paddle, (self.COLOR_PADDLE, self.COLOR_PADDLE_TOP, self.COLOR_PADDLE_SIDE), self.ISO_DEPTH)
        
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            size = max(1, int(3 * (p['life'] / 30.0)))
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p['color'], alpha), (size, size), size)
            self.screen.blit(s, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

        if self.lives > 0 or self.game_over:
            shadow_pos = (int(self.ball_pos[0]), int(self.ball_pos[1] + self.ISO_DEPTH + 2))
            pygame.gfxdraw.filled_circle(self.screen, shadow_pos[0], shadow_pos[1], self.BALL_RADIUS, self.COLOR_BALL_SHADOW)
            
            ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
            pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _draw_iso_rect(self, surface, rect, colors, depth):
        main_color, top_color, side_color = colors
        
        top_points = [
            (rect.left, rect.top), (rect.right, rect.top),
            (rect.right + depth, rect.top - depth), (rect.left + depth, rect.top - depth)
        ]
        pygame.draw.polygon(surface, top_color, top_points)

        side_points = [
            (rect.right, rect.top), (rect.right, rect.bottom),
            (rect.right + depth, rect.bottom - depth), (rect.right + depth, rect.top - depth)
        ]
        pygame.draw.polygon(surface, side_color, side_points)

        pygame.draw.rect(surface, main_color, rect)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        lives_text = self.font_large.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 180, 5))
        for i in range(self.lives):
            pos = (self.SCREEN_WIDTH - 80 + i * 25, 20)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_BALL)
            
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if not self.blocks else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_TEXT)
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
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Isometric Block Breaker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    total_reward = 0
    
    while running:
        movement = 0 
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        action = [movement, 0, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            terminated = term
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated and not info.get('printed_final_score', False):
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            info['printed_final_score'] = True
            
    env.close()