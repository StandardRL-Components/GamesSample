
# Generated: 2025-08-27T15:50:19.066223
# Source Brief: brief_01086.md
# Brief Index: 1086

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based block breaker. Clear all blocks to win, but lose all your balls and you lose."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    UI_HEIGHT = 40
    GAME_AREA_Y_START = UI_HEIGHT

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PADDLE = (220, 220, 250)
    COLOR_PADDLE_GLOW = (100, 100, 180)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (180, 180, 200)
    COLOR_BORDER = (80, 80, 100)
    COLOR_TEXT = (200, 200, 220)
    BLOCK_COLORS = {
        10: (50, 200, 50),   # Green
        20: (50, 100, 220),  # Blue
        30: (220, 50, 50)    # Red
    }

    # Game parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 7
    BALL_INITIAL_SPEED = 6
    MAX_BALL_SPEED_X = 8
    BLOCK_ROWS = 10
    BLOCK_COLS = 10
    BLOCK_WIDTH = 58
    BLOCK_HEIGHT = 15
    BLOCK_SPACING = 6
    INITIAL_LIVES = 3
    MAX_STEPS = 1000

    # Reward parameters
    REWARD_STEP = -0.02
    REWARD_PADDLE_EDGE = 0.1
    REWARD_PADDLE_CENTER = -0.2
    REWARD_CHAIN_BONUS = 5
    REWARD_WIN = 100
    REWARD_LOSS = -100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)
        
        # Etc...        
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        
        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.win = False
        self.chain_breaks = 0
        self.particles = []
        
        self._create_blocks()
        self._reset_ball_and_paddle()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        self.blocks = []
        total_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - total_width) / 2
        start_y = self.GAME_AREA_Y_START + 30

        for row in range(self.BLOCK_ROWS):
            for col in range(self.BLOCK_COLS):
                if row < 2: points = 30
                elif row < 6: points = 20
                else: points = 10
                color = self.BLOCK_COLORS[points]
                
                x = start_x + col * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = start_y + row * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({"rect": block_rect, "color": color, "points": points})

    def _reset_ball_and_paddle(self):
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_attached = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]
        self.chain_breaks = 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = self.REWARD_STEP
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        self._handle_input(movement, space_held)
        
        if not self.ball_attached:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
        else:
            self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]

        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.lives <= 0:
                self.win = False
                reward += self.REWARD_LOSS
            elif not self.blocks:
                self.win = True
                reward += self.REWARD_WIN
            else: # Max steps reached
                self.win = False

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        return self.lives <= 0 or not self.blocks or self.steps >= self.MAX_STEPS

    def _handle_input(self, movement, space_held):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle.x))
        
        if space_held and self.ball_attached:
            self.ball_attached = False
            launch_angle_factor = 0
            if movement == 3: launch_angle_factor = -0.5
            if movement == 4: launch_angle_factor = 0.5
            self.ball_vel = [self.BALL_INITIAL_SPEED * launch_angle_factor, -self.BALL_INITIAL_SPEED]
            # Sound: launch.wav

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= -1
        elif self.ball_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_pos[0] = self.SCREEN_WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -1
        if self.ball_pos[1] <= self.GAME_AREA_Y_START + self.BALL_RADIUS:
            self.ball_pos[1] = self.GAME_AREA_Y_START + self.BALL_RADIUS
            self.ball_vel[1] *= -1
        # Sound: bounce.wav for all wall hits

        # Bottom wall
        if self.ball_pos[1] >= self.SCREEN_HEIGHT:
            self.lives -= 1
            # Sound: lose_life.wav
            if self.lives > 0: self._reset_ball_and_paddle()
            
        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # Sound: paddle_hit.wav
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            self.ball_vel[1] *= -1
            self.chain_breaks = 0

            hit_offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += hit_offset * 4
            self.ball_vel[0] = max(-self.MAX_BALL_SPEED_X, min(self.MAX_BALL_SPEED_X, self.ball_vel[0]))
            
            reward += self.REWARD_PADDLE_EDGE if abs(hit_offset) > 0.7 else self.REWARD_PADDLE_CENTER

        # Block collisions
        hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            block = self.blocks[hit_block_idx]
            # Sound: block_break.wav
            
            block_rect = block['rect']
            dx = self.ball_pos[0] - block_rect.centerx
            dy = self.ball_pos[1] - block_rect.centery
            if abs(dx / block_rect.width) > abs(dy / block_rect.height): self.ball_vel[0] *= -1
            else: self.ball_vel[1] *= -1

            reward += block['points'] + self.REWARD_CHAIN_BONUS * self.chain_breaks
            self.score += block['points'] + self.REWARD_CHAIN_BONUS * self.chain_breaks
            self.chain_breaks += 1
            
            self._spawn_particles(block_rect.center, block['color'])
            self.blocks.pop(hit_block_idx)

        return reward

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            vel = [random.uniform(-2, 2), random.uniform(-2, 2)]
            radius = random.uniform(2, 5)
            life = random.randint(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'radius': radius, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['radius'] -= 0.15
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, self.GAME_AREA_Y_START, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GAME_AREA_Y_START), 2)
        
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
            highlight_color = tuple(min(255, c + 30) for c in block['color'])
            pygame.draw.rect(self.screen, highlight_color, block['rect'].inflate(-8, -8), border_radius=3)
        
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, p['color'])
        
        glow_rect = self.paddle.inflate(6, 6)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_GLOW, glow_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        
    def _render_ui(self):
        pygame.draw.rect(self.screen, self.COLOR_BG, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 8))
        
        lives_text = self.font_main.render(f"BALLS: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 8))
        
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks)
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        obs_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(obs_surface, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000)
            env.reset()
            
        clock.tick(60)
        
    env.close()