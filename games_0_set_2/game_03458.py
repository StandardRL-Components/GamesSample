
# Generated: 2025-08-27T23:25:24.005206
# Source Brief: brief_03458.md
# Brief Index: 3458

        
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

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A fast-paced, visually vibrant block-breaking game. Destroy all neon blocks with the ball to win. You have 3 lives."
    )

    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG_TOP = (10, 0, 20)
    COLOR_BG_BOTTOM = (40, 0, 60)
    COLOR_PADDLE = (0, 150, 255)
    COLOR_PADDLE_GLOW = (0, 100, 200)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (200, 200, 255)
    COLOR_WALL = (100, 100, 120)
    COLOR_TEXT = (220, 220, 240)
    BLOCK_COLORS = [
        (0, 255, 150),  # Green
        (255, 0, 150),  # Pink
        (255, 200, 0),  # Yellow
        (150, 0, 255),  # Purple
    ]

    # Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_Y = SCREEN_HEIGHT - 30
    BALL_RADIUS = 7
    WALL_THICKNESS = 5
    
    # Game parameters
    PADDLE_SPEED = 10
    BALL_INITIAL_SPEED = 5
    BALL_MAX_SPEED = 9
    MAX_EPISODE_STEPS = 30 * 60 # 60 seconds at 30fps
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Game state variables are initialized in reset()
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = None
        self.blocks = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.balls_left = None
        self.combo_hits = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.combo_hits = 0
        
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.PADDLE_Y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self._attach_ball()
        self._create_blocks()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _attach_ball(self):
        self.ball_attached = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _create_blocks(self):
        self.blocks = []
        block_width = 60
        block_height = 20
        gap = 5
        rows = 4
        cols = 9
        start_x = (self.SCREEN_WIDTH - (cols * (block_width + gap))) / 2 + self.WALL_THICKNESS
        start_y = 50
        
        for r in range(rows):
            for c in range(cols):
                color = self.BLOCK_COLORS[(r + c) % len(self.BLOCK_COLORS)]
                block_rect = pygame.Rect(
                    start_x + c * (block_width + gap),
                    start_y + r * (block_height + gap),
                    block_width,
                    block_height
                )
                self.blocks.append({"rect": block_rect, "color": color})

    def step(self, action):
        reward = 0
        self.steps += 1
        
        # Unpack action
        movement = action[0]
        space_pressed = action[1] == 1
        
        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(self.WALL_THICKNESS, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH - self.WALL_THICKNESS))

        # Reward shaping: penalize safe, central movements
        safe_zone_width = self.SCREEN_WIDTH * 0.2
        safe_zone_start = (self.SCREEN_WIDTH - safe_zone_width) / 2
        if safe_zone_start < self.paddle.centerx < safe_zone_start + safe_zone_width and movement in [3, 4]:
            reward -= 0.02

        # Launch ball
        if self.ball_attached and space_pressed:
            # sfx: launch_ball.wav
            self.ball_attached = False
            angle = self.np_random.uniform(-math.pi/4, math.pi/4) # Launch slightly randomized
            self.ball_vel = [
                self.BALL_INITIAL_SPEED * math.sin(angle),
                -self.BALL_INITIAL_SPEED * math.cos(angle)
            ]

        # --- Update Game Logic ---
        if self.ball_attached:
            self.ball_pos[0] = self.paddle.centerx
        else:
            # Reward for keeping ball in play
            reward += 0.01

            # Update ball position
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # --- Collisions ---
            # Wall collisions
            if ball_rect.left <= self.WALL_THICKNESS:
                ball_rect.left = self.WALL_THICKNESS
                self.ball_vel[0] *= -1
                # sfx: wall_bounce.wav
            if ball_rect.right >= self.SCREEN_WIDTH - self.WALL_THICKNESS:
                ball_rect.right = self.SCREEN_WIDTH - self.WALL_THICKNESS
                self.ball_vel[0] *= -1
                # sfx: wall_bounce.wav
            if ball_rect.top <= self.WALL_THICKNESS:
                ball_rect.top = self.WALL_THICKNESS
                self.ball_vel[1] *= -1
                # sfx: wall_bounce.wav

            # Paddle collision
            if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
                # sfx: paddle_bounce.wav
                self.ball_vel[1] *= -1
                
                # Influence horizontal velocity based on hit location
                offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] += offset * 2
                
                # Clamp ball speed
                speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
                if speed > self.BALL_MAX_SPEED:
                    scale = self.BALL_MAX_SPEED / speed
                    self.ball_vel[0] *= scale
                    self.ball_vel[1] *= scale
                
                ball_rect.bottom = self.paddle.top
                self.combo_hits = 0 # Reset combo on paddle hit
            
            self.ball_pos = [ball_rect.centerx, ball_rect.centery]

            # Block collisions
            hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
            if hit_block_idx != -1:
                # sfx: block_break.wav
                hit_block = self.blocks.pop(hit_block_idx)
                self.ball_vel[1] *= -1 # Simple bounce
                self.score += 10
                reward += 1
                self.combo_hits += 1
                if self.combo_hits > 1:
                    combo_bonus = self.combo_hits * 2
                    self.score += combo_bonus
                    reward += 2 # Combo reward
                
                self._create_particles(hit_block['rect'].center, hit_block['color'])
        
        # Update particles
        self._update_particles()

        # --- Termination Conditions ---
        # Ball lost
        if self.ball_pos[1] > self.SCREEN_HEIGHT:
            # sfx: lose_life.wav
            self.balls_left -= 1
            self.combo_hits = 0
            if self.balls_left > 0:
                self._attach_ball()
            else:
                self.game_over = True
        
        # Win condition
        if not self.blocks:
            # sfx: win_level.wav
            self.game_over = True
            self.score += 500
            reward += 100 # Win bonus
        
        # Lose condition
        if self.game_over and self.balls_left == 0:
            reward -= 100 # Lose penalty
            
        terminated = self.game_over or self.steps >= self.MAX_EPISODE_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block['color']), block['rect'], 2)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 3, 3))
            self.screen.blit(temp_surf, (int(p['pos'][0]), int(p['pos'][1])))

        # Paddle
        pygame.gfxdraw.box(self.screen, self.paddle, self.COLOR_PADDLE_GLOW)
        inner_paddle = self.paddle.inflate(-6, -6)
        pygame.gfxdraw.box(self.screen, inner_paddle, self.COLOR_PADDLE)

        # Ball
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 4, (*self.COLOR_BALL_GLOW, 100))
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Balls left
        for i in range(self.balls_left):
            x = self.SCREEN_WIDTH - 30 - (i * (self.BALL_RADIUS * 2 + 10))
            pygame.gfxdraw.filled_circle(self.screen, x, 28, self.BALL_RADIUS, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, x, 28, self.BALL_RADIUS, self.COLOR_PADDLE)

        if self.game_over:
            msg = "LEVEL CLEAR!" if not self.blocks else "GAME OVER"
            msg_surf = self.font_main.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks)
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

if __name__ == "__main__":
    # --- Example Usage ---
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a Pygame window.
    # This part is for demonstration and debugging, not part of the core env.
    
    # Re-initialize pygame for display
    pygame.display.init()
    pygame.display.set_caption("Block Breaker")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Map keyboard keys to actions
    key_to_action = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not terminated:
        # --- Action Mapping for Human ---
        movement = 0 # No-op
        space = 0 # Released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_SPACE]:
            space = 1

        action = [movement, space, 0] # Shift is not used
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to Display ---
        # The observation is a numpy array, we need to convert it back to a surface
        # Transpose from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Control FPS ---
        env.clock.tick(30)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause before restarting
            obs, info = env.reset()
            terminated = False

    env.close()