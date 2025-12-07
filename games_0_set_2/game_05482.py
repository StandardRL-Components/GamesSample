
# Generated: 2025-08-28T05:11:35.768213
# Source Brief: brief_05482.md
# Brief Index: 5482

        
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
        "Fast-paced arcade block breaker. Clear all blocks within the time limit to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_FPS = 30
    MAX_STEPS = 1800  # 60 seconds * 30 FPS

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (30, 10, 50)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_POWERUP_WIDER = (0, 255, 128)
    COLOR_POWERUP_FAST = (255, 100, 100)
    BLOCK_COLORS = [
        (255, 80, 80), (255, 160, 80), (255, 240, 80),
        (80, 255, 80), (80, 255, 255), (80, 160, 255)
    ]

    # Paddle
    PADDLE_HEIGHT = 12
    PADDLE_BASE_WIDTH = 80
    PADDLE_SPEED = 12

    # Ball
    BALL_RADIUS = 7
    BALL_BASE_SPEED = 7

    # Blocks
    BLOCK_ROWS = 6
    BLOCK_COLS = 12
    BLOCK_WIDTH = 50
    BLOCK_HEIGHT = 15
    BLOCK_SPACING = 4
    BLOCK_AREA_TOP = 60

    # Powerups
    POWERUP_CHANCE = 0.15
    POWERUP_SPEED = 2.5
    POWERUP_SIZE = 10
    POWERUP_DURATION = 300 # steps

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
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.ball_attached = True
        
        self.blocks = []
        self.block_rows_map = []
        self.total_blocks = 0
        
        self.particles = []
        self.powerups = []
        self.active_powerups = {}

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS

        # Paddle
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_BASE_WIDTH) // 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_BASE_WIDTH,
            self.PADDLE_HEIGHT
        )

        # Ball
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_speed = self.BALL_BASE_SPEED
        self.ball_attached = True

        # Blocks
        self.blocks = []
        self.block_rows_map = [[] for _ in range(self.BLOCK_ROWS)]
        grid_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.SCREEN_WIDTH - grid_width) // 2
        
        for r in range(self.BLOCK_ROWS):
            for c in range(self.BLOCK_COLS):
                if self.np_random.random() > 0.2: # Create gaps
                    x = start_x + c * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                    y = self.BLOCK_AREA_TOP + r * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                    block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                    block_color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                    block_obj = {"rect": block_rect, "color": block_color, "row": r}
                    self.blocks.append(block_obj)
                    self.block_rows_map[r].append(block_obj)
        self.total_blocks = len(self.blocks)

        # Effects
        self.particles = []
        self.powerups = []
        self.active_powerups = {}

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.GAME_FPS)

        reward = 0
        
        # 1. Handle Input
        movement = action[0]
        space_pressed = action[1] == 1
        
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.SCREEN_WIDTH, self.paddle.right)

        if space_pressed and self.ball_attached:
            # sfx: launch_ball.wav
            self.ball_attached = False
            self.ball_vel = pygame.Vector2(
                self.np_random.uniform(-0.5, 0.5), -1
            ).normalize() * self.ball_speed

        # 2. Update Game State
        self.steps += 1
        self.timer -= 1
        
        self._update_powerups()
        self._update_ball()
        self._update_particles()
        
        # 3. Handle Collisions & Calculate Reward
        reward += self._handle_collisions()

        # 4. Check Termination
        terminated = False
        if not self.blocks:
            terminated = True
            reward += 100 # Win bonus
            self.game_over = True
        elif self.timer <= 0:
            terminated = True
            reward -= 100 # Time out penalty
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 100
            self.game_over = True


        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        if self.ball_attached:
            self.ball_pos.x = self.paddle.centerx
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
        else:
            self.ball_pos += self.ball_vel

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)

    def _update_powerups(self):
        # Move falling powerups
        self.powerups = [p for p in self.powerups if p['rect'].top < self.SCREEN_HEIGHT]
        for p in self.powerups:
            p['rect'].y += self.POWERUP_SPEED
        
        # Update active powerup timers
        ended_powerups = []
        for key, timer in self.active_powerups.items():
            self.active_powerups[key] = timer - 1
            if self.active_powerups[key] <= 0:
                ended_powerups.append(key)

        for key in ended_powerups:
            del self.active_powerups[key]
            if key == 'wider_paddle':
                self.paddle.width = self.PADDLE_BASE_WIDTH
                self.paddle.centerx = self.paddle.centerx # re-center
            elif key == 'fast_ball':
                self.ball_speed = self.BALL_BASE_SPEED
                if not self.ball_attached:
                    self.ball_vel = self.ball_vel.normalize() * self.ball_speed
    
    def _handle_collisions(self):
        reward = 0
        if self.ball_attached:
            return reward

        # Ball and Walls
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.SCREEN_WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.ball_pos.x, self.SCREEN_WIDTH - self.BALL_RADIUS))
            # sfx: bounce_wall.wav

        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # sfx: bounce_wall.wav

        # Ball and Paddle
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
            # sfx: bounce_paddle.wav
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
            
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.paddle.width / 2)
            offset = max(-1, min(1, offset)) # Clamp
            
            new_vel = pygame.Vector2(offset, -self.ball_vel.y).normalize()
            self.ball_vel = new_vel * self.ball_speed

        # Ball and Blocks
        hit_block = None
        for block in self.blocks:
            if block["rect"].colliderect(ball_rect):
                hit_block = block
                break
        
        if hit_block:
            # sfx: block_hit.wav
            self.blocks.remove(hit_block)
            self.score += 10
            reward += 0.1

            # Check for row clear
            row_idx = hit_block["row"]
            self.block_rows_map[row_idx].remove(hit_block)
            if not self.block_rows_map[row_idx]:
                self.score += 100
                reward += 5.0
                # sfx: row_clear.wav

            # Collision response
            overlap = ball_rect.clip(hit_block["rect"])
            if overlap.width < overlap.height:
                self.ball_vel.x *= -1
            else:
                self.ball_vel.y *= -1
            
            # Spawn particles
            for _ in range(10):
                vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2))
                self.particles.append({
                    'pos': pygame.Vector2(ball_rect.center), 'vel': vel,
                    'radius': self.np_random.uniform(2, 4), 'life': 20,
                    'color': hit_block['color']
                })
            
            # Spawn powerup
            if self.np_random.random() < self.POWERUP_CHANCE:
                ptype = self.np_random.choice(['wider_paddle', 'fast_ball'])
                self.powerups.append({
                    'rect': pygame.Rect(hit_block['rect'].centerx - self.POWERUP_SIZE / 2, hit_block['rect'].centery, self.POWERUP_SIZE, self.POWERUP_SIZE),
                    'type': ptype
                })

        # Paddle and Powerups
        collected_powerups = []
        for p in self.powerups:
            if self.paddle.colliderect(p['rect']):
                # sfx: powerup_collect.wav
                collected_powerups.append(p)
                self.score += 50
                reward += 1.0
                self.active_powerups[p['type']] = self.POWERUP_DURATION
                
                if p['type'] == 'wider_paddle':
                    self.paddle.width = self.PADDLE_BASE_WIDTH * 1.5
                    self.paddle.centerx = self.paddle.centerx
                elif p['type'] == 'fast_ball':
                    self.ball_speed = self.BALL_BASE_SPEED * 1.5
                    if not self.ball_attached:
                        self.ball_vel = self.ball_vel.normalize() * self.ball_speed
        
        self.powerups = [p for p in self.powerups if p not in collected_powerups]

        return reward

    def _get_observation(self):
        # Background
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
            # Add a subtle 3D effect
            darker_color = tuple(max(0, c - 40) for c in block['color'])
            pygame.draw.rect(self.screen, darker_color, block['rect'], width=2, border_radius=3)

        # Powerups
        for p in self.powerups:
            color = self.COLOR_POWERUP_WIDER if p['type'] == 'wider_paddle' else self.COLOR_POWERUP_FAST
            # Flashing effect
            if (self.steps // 5) % 2 == 0:
                pygame.draw.rect(self.screen, color, p['rect'], border_radius=2)
                pygame.gfxdraw.rectangle(self.screen, p['rect'], (255,255,255,150))

        # Particles
        for p in self.particles:
            if p['radius'] > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])
        
        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        
        # Ball
        # Glow effect
        glow_color = (*self.COLOR_BALL, 50)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS + 4, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

        # UI
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_left = max(0, self.timer / self.GAME_FPS)
        time_color = (255, 100, 100) if time_left < 10 else self.COLOR_TEXT
        time_text = self.font_large.render(f"TIME: {time_left:.1f}", True, time_color)
        time_text_rect = time_text.get_rect(right=self.SCREEN_WIDTH - 10, top=10)
        self.screen.blit(time_text, time_text_rect)

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
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
    # This block allows you to play the game directly
    # Requires `pygame` to be installed with display support
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override screen to be a display surface
    env.screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    
    terminated = False
    
    # --- Main Game Loop ---
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # The environment returns the observation as a numpy array.
        # For direct play, we just need to re-draw to the display surface.
        env._get_observation() # This draws on the internal surface
        pygame.display.flip() # Update the display
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)

    env.close()