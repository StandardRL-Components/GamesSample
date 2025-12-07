
# Generated: 2025-08-28T05:28:17.477891
# Source Brief: brief_05588.md
# Brief Index: 5588

        
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
        "A fast-paced, retro-arcade block breaker. Destroy all blocks within the time limit to win. Don't let the ball fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_TIME = 60  # seconds

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_PADDLE = (220, 220, 240)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 200, 0)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_BG = (40, 45, 60)
    BLOCK_COLORS = [
        (255, 70, 70),   # Red
        (70, 255, 70),   # Green
        (70, 70, 255),   # Blue
        (255, 165, 0),   # Orange
        (160, 32, 240),  # Purple
    ]

    # Game Parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 4
    MAX_BALL_SPEED = 8
    
    BLOCK_ROWS = 5
    BLOCK_COLS = 10
    BLOCK_WIDTH = 58
    BLOCK_HEIGHT = 20
    BLOCK_SPACING = 6
    BLOCK_AREA_TOP = 60

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.time_remaining = 0
        self.game_over = False
        self.win = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.win = False
        self.time_remaining = self.MAX_TIME * self.FPS

        # Paddle state
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        # Ball state
        self._reset_ball()

        # Blocks state
        self.blocks = []
        total_block_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - total_block_width) / 2
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.BLOCK_AREA_TOP + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                color_index = i % len(self.BLOCK_COLORS)
                points = (self.BLOCK_ROWS - i) * 10
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({"rect": block_rect, "color": self.BLOCK_COLORS[color_index], "points": points})

        # Particles state
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_launched = False
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1], dtype=np.float64)
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float64)
        
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = self._update_game_state(movement, space_held)
        
        self.steps += 1
        self.time_remaining = max(0, self.time_remaining - 1)
        
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        if terminated:
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_game_state(self, movement, space_held):
        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        if space_held and not self.ball_launched:
            # sfx: launch_ball
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
            self.ball_vel = np.array([math.cos(angle), math.sin(angle)]) * self.INITIAL_BALL_SPEED

        # --- Update Ball ---
        if not self.ball_launched:
            self.ball_pos[0] = self.paddle.centerx
        else:
            self.ball_pos += self.ball_vel
        
        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # --- Handle Collisions & Rewards ---
        reward = 0
        if self.ball_launched:
            reward += 0.01  # Small reward for keeping ball in play

            # Wall collisions
            if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
                self.ball_vel[0] *= -1
                self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
                # sfx: wall_bounce
            if self.ball_pos[1] <= self.BALL_RADIUS:
                self.ball_vel[1] *= -1
                self.ball_pos[1] = self.BALL_RADIUS
                # sfx: wall_bounce

            # Paddle collision
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
            if self.paddle.colliderect(ball_rect) and self.ball_vel[1] > 0:
                # sfx: paddle_bounce
                self.ball_vel[1] *= -1
                self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

                # Change horizontal velocity based on hit location
                offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] = offset * self.INITIAL_BALL_SPEED * 1.2
                
                # Penalize safe center hits
                if abs(offset) < 0.1:
                    reward -= 0.2

                # Clamp ball speed
                speed = np.linalg.norm(self.ball_vel)
                if speed > self.MAX_BALL_SPEED:
                    self.ball_vel = self.ball_vel / speed * self.MAX_BALL_SPEED

            # Block collisions
            for block in self.blocks[:]:
                if block['rect'].colliderect(ball_rect):
                    # sfx: block_hit
                    self.blocks.remove(block)
                    reward += 1
                    self.score += block['points']

                    # Bounce logic
                    self._create_particles(block['rect'].center, block['color'])
                    
                    # Determine bounce direction (simple method)
                    dbx = self.ball_pos[0] - block['rect'].centerx
                    dby = self.ball_pos[1] - block['rect'].centery
                    if abs(dbx) > abs(dby):
                        self.ball_vel[0] *= -1
                    else:
                        self.ball_vel[1] *= -1
                    break

            # Bottom boundary (lose life)
            if self.ball_pos[1] > self.SCREEN_HEIGHT + self.BALL_RADIUS:
                # sfx: lose_life
                self.lives -= 1
                self._reset_ball()
        
        return reward

    def _check_termination(self):
        terminal_reward = 0
        terminated = False
        
        if self.lives <= 0:
            terminated = True
            terminal_reward = -100
            self.win = False
        elif not self.blocks:
            terminated = True
            terminal_reward = 100
            self.win = True
        elif self.time_remaining <= 0:
            terminated = True
            terminal_reward = -10
            self.win = False
            
        return terminated, terminal_reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': np.array(pos, dtype=np.float64),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_game(self):
        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block['color']), block['rect'], 2)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = p['color']
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), (*color, alpha))

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Render ball with glow
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 3, (*self.COLOR_BALL_GLOW, 50))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 1, (*self.COLOR_BALL_GLOW, 100))
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # UI Background
        ui_bar = pygame.Rect(0, 0, self.SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_bar)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 40), (self.SCREEN_WIDTH, 40))

        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score:05d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 8))

        # Lives
        lives_text = self.font_medium.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 120, 8))
        
        # Time
        time_secs = self.time_remaining // self.FPS
        time_text = self.font_medium.render(f"TIME: {time_secs:02d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, ((self.SCREEN_WIDTH - time_text.get_width()) // 2, 8))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
            "time_remaining_steps": self.time_remaining,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Block Breaker Gym Environment")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
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
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Display ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # So we need to get the surface from the env's internal screen
        surf = pygame.transform.flip(env.screen, False, False)
        surf = pygame.transform.rotate(surf, -90)
        surf = pygame.transform.flip(surf, True, False)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

    env.close()