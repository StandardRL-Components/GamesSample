
# Generated: 2025-08-28T04:40:27.464364
# Source Brief: brief_05326.md
# Brief Index: 5326

        
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
        "Controls: ←→ to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade game. Move the paddle to bounce the ball and destroy all the blocks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.PADDLE_Y = self.HEIGHT - 40
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 6.0
        self.MAX_STEPS = 1000
        self.BORDER_WIDTH = 10

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_BORDER = (100, 100, 120)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_PADDLE_GLOW = (200, 200, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 150)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 87, 87), (255, 170, 87), (232, 255, 87),
            (87, 255, 95), (87, 255, 240), (87, 161, 255),
            (136, 87, 255), (255, 87, 234)
        ]

        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.Font(None, 72)
        self.font_small = pygame.font.Font(None, 36)

        # --- Game State ---
        self.rng = None
        self.paddle_x = 0
        self.ball_pos = [0.0, 0.0]
        self.ball_vel = [0.0, 0.0]
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.paddle_x = self.WIDTH / 2
        self.ball_pos = [self.WIDTH / 2, self.HEIGHT / 2]

        # Random initial ball velocity, ensuring it goes downwards
        angle = self.rng.uniform(math.pi * 0.25, math.pi * 0.75) 
        self.ball_vel = [math.cos(angle) * self.BALL_SPEED, math.sin(angle) * self.BALL_SPEED]
        if self.rng.random() < 0.5:
            self.ball_vel[0] *= -1

        self._create_blocks()
        self.particles = []

        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        self.blocks = []
        block_rows = 2
        block_cols = 10
        block_width = (self.WIDTH - 2 * self.BORDER_WIDTH) / block_cols
        block_height = 20
        gap = 4

        for r in range(block_rows):
            for c in range(block_cols):
                block_x = self.BORDER_WIDTH + c * block_width + gap / 2
                block_y = 60 + r * (block_height + gap)
                rect = pygame.Rect(
                    block_x,
                    block_y,
                    block_width - gap,
                    block_height - gap
                )
                color = self.BLOCK_COLORS[(r * block_cols + c) % len(self.BLOCK_COLORS)]
                self.blocks.append({'rect': rect, 'color': color})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.02 # Time penalty
        self.steps += 1

        # 1. Handle Action
        movement = action[0]
        if movement == 3: # Left
            self.paddle_x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle_x += self.PADDLE_SPEED
        
        self.paddle_x = np.clip(
            self.paddle_x,
            self.BORDER_WIDTH + self.PADDLE_WIDTH / 2,
            self.WIDTH - self.BORDER_WIDTH - self.PADDLE_WIDTH / 2
        )

        # 2. Update Ball Position
        prev_ball_pos_y = self.ball_pos[1]
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS,
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )
        paddle_rect = pygame.Rect(
            self.paddle_x - self.PADDLE_WIDTH / 2,
            self.PADDLE_Y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        # 3. Collision Detection
        # Walls
        if self.ball_pos[0] - self.BALL_RADIUS <= self.BORDER_WIDTH:
            self.ball_pos[0] = self.BORDER_WIDTH + self.BALL_RADIUS
            self.ball_vel[0] *= -1
            # sfx: wall_bounce
        if self.ball_pos[0] + self.BALL_RADIUS >= self.WIDTH - self.BORDER_WIDTH:
            self.ball_pos[0] = self.WIDTH - self.BORDER_WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -1
            # sfx: wall_bounce
        if self.ball_pos[1] - self.BALL_RADIUS <= self.BORDER_WIDTH:
            self.ball_pos[1] = self.BORDER_WIDTH + self.BALL_RADIUS
            self.ball_vel[1] *= -1
            # sfx: wall_bounce

        # Paddle
        if self.ball_vel[1] > 0 and ball_rect.colliderect(paddle_rect):
            self.ball_pos[1] = self.PADDLE_Y - self.BALL_RADIUS
            self.ball_vel[1] *= -1

            offset = (self.ball_pos[0] - self.paddle_x) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = offset * self.BALL_SPEED * 1.2
            
            current_speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if current_speed > 0:
                scale = self.BALL_SPEED / current_speed
                self.ball_vel[0] *= scale
                self.ball_vel[1] *= scale

            if abs(offset) < 0.1: # Center 20%
                reward += 0.1
                # sfx: paddle_center_hit

            self._create_particles(self.ball_pos, 20, self.COLOR_PADDLE)
            # sfx: paddle_hit

        # Blocks
        hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            block = self.blocks[hit_block_idx]
            reward += 1
            self._create_particles(ball_rect.center, 30, block['color'])
            # sfx: block_break

            prev_ball_rect = pygame.Rect(
                self.ball_pos[0] - self.ball_vel[0] - self.BALL_RADIUS,
                self.ball_pos[1] - self.ball_vel[1] - self.BALL_RADIUS,
                self.BALL_RADIUS * 2, self.BALL_RADIUS * 2
            )
            
            if prev_ball_rect.bottom <= block['rect'].top or prev_ball_rect.top >= block['rect'].bottom:
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1

            self.blocks.pop(hit_block_idx)
        
        # 4. Update Particles
        self._update_particles()

        # 5. Check Termination Conditions
        terminated = False
        
        # Near miss check
        if prev_ball_pos_y < self.PADDLE_Y and self.ball_pos[1] >= self.PADDLE_Y:
            dist_left = abs(self.ball_pos[0] - paddle_rect.left)
            dist_right = abs(self.ball_pos[0] - paddle_rect.right)
            if min(dist_left, dist_right) < self.BALL_RADIUS + 5:
                 if not paddle_rect.colliderect(ball_rect):
                    reward -= 5
                    # sfx: near_miss

        # Loss condition
        if self.ball_pos[1] + self.BALL_RADIUS > self.HEIGHT:
            self.game_over = True
            terminated = True
            reward -= 100
            # sfx: game_over_loss

        # Win condition
        if not self.blocks and not self.game_over:
            self.game_over = True
            terminated = True
            reward += 100
            # sfx: game_over_win

        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            p['radius'] -= 0.1

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.rng.uniform(2, 5),
                'color': color,
                'lifespan': self.rng.integers(20, 40)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, (25, 25, 45), (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, (25, 25, 45), (0, i), (self.WIDTH, i))

        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, 0, self.WIDTH, self.HEIGHT), self.BORDER_WIDTH)
        
        for p in self.particles:
            if p['radius'] > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                alpha = max(0, min(255, int(255 * (p['lifespan'] / 30))))
                color_with_alpha = (*p['color'], alpha)
                temp_surf = pygame.Surface((int(p['radius'])*2, int(p['radius'])*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color_with_alpha, (p['radius'], p['radius']), p['radius'])
                self.screen.blit(temp_surf, (pos[0] - p['radius'], pos[1] - p['radius']))

        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
            highlight_color = tuple(min(255, c + 40) for c in block['color'])
            pygame.draw.line(self.screen, highlight_color, block['rect'].topleft, block['rect'].topright, 2)
            pygame.draw.line(self.screen, highlight_color, block['rect'].topleft, block['rect'].bottomleft, 2)

        paddle_rect = pygame.Rect(
            self.paddle_x - self.PADDLE_WIDTH / 2, self.PADDLE_Y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        glow_rect = paddle_rect.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE_GLOW, 50), glow_surf.get_rect(), border_radius=10)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=5)

        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        glow_radius = int(self.BALL_RADIUS * 2.5)
        glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_BALL_GLOW, 100), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (ball_pos_int[0] - glow_radius, ball_pos_int[1] - glow_radius))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.BORDER_WIDTH + 10, self.BORDER_WIDTH + 5))

        blocks_text = self.font_small.render(f"BLOCKS: {len(self.blocks)}", True, self.COLOR_TEXT)
        self.screen.blit(blocks_text, (self.WIDTH - blocks_text.get_width() - self.BORDER_WIDTH - 10, self.BORDER_WIDTH + 5))

        if self.game_over:
            end_text_str = "YOU WIN!" if not self.blocks else "GAME OVER"
            end_text = self.font_big.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_left": len(self.blocks),
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Breakout Grid")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(60)

    env.close()