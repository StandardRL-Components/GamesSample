
# Generated: 2025-08-28T05:36:47.131579
# Source Brief: brief_05636.md
# Brief Index: 5636

        
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
    user_guide = "Controls: Use ← and → to move the paddle."

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade game where you control a paddle to break blocks with a bouncing ball."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (16, 16, 24)
    COLOR_PADDLE = (240, 240, 240)
    COLOR_BALL = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (255, 87, 34), (255, 193, 7), (139, 195, 74), (0, 188, 212), (33, 150, 243)
    ]

    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 12
    PADDLE_SPEED = 10
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 6.0
    BALL_SPEED_INCREMENT = 0.6
    MAX_STEPS = 1800 # 60 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.large_font = pygame.font.Font(None, 60)
        
        # State variables (initialized in reset)
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.game_over = False
        self.blocks_destroyed_count = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.blocks_destroyed_count = 0
        self.particles = []
        
        # Create blocks
        self.blocks = []
        block_rows = 2
        block_cols = 10
        block_width = 58
        block_height = 20
        start_x = (self.WIDTH - (block_cols * (block_width + 6))) / 2
        start_y = 50
        for i in range(block_rows):
            for j in range(block_cols):
                x = start_x + j * (block_width + 6)
                y = start_y + i * (block_height + 6)
                self.blocks.append(pygame.Rect(x, y, block_width, block_height))
        
        self._reset_ball()
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        speed = self.INITIAL_BALL_SPEED + (self.blocks_destroyed_count // 5) * self.BALL_SPEED_INCREMENT
        self.ball_vel = [math.cos(angle) * speed, math.sin(angle) * speed]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        
        # Update paddle position
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))
        
        # Update game logic
        reward = self._update_game_state()
        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.lives <= 0:
            reward += -100  # Final penalty for losing
            terminated = True
            self.game_over = True
        elif not self.blocks:
            reward += 100  # Final bonus for winning
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self):
        step_reward = -0.02  # Small penalty per frame to encourage speed

        # --- Ball Movement ---
        prev_ball_pos = list(self.ball_pos)
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # --- Wall Collisions ---
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos[0]))
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = max(self.BALL_RADIUS, self.ball_pos[1])
            # sfx: wall_bounce

        # --- Paddle Collision ---
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle_hit
            step_reward += 0.1
            hit_pos_norm = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            
            if abs(hit_pos_norm) > 0.7:
                step_reward += 2.0  # Risky hit bonus
            else:
                step_reward -= 0.2  # Safe hit penalty

            self.ball_vel[0] = hit_pos_norm * 5.0
            self.ball_vel[1] *= -1
            
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            current_base_speed = self.INITIAL_BALL_SPEED + (self.blocks_destroyed_count // 5) * self.BALL_SPEED_INCREMENT
            if speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / speed) * current_base_speed
                self.ball_vel[1] = (self.ball_vel[1] / speed) * current_base_speed
            
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

        # --- Block Collisions ---
        collided_block_index = ball_rect.collidelist(self.blocks)
        if collided_block_index != -1:
            # sfx: block_break
            block = self.blocks.pop(collided_block_index)
            step_reward += 1.0
            self.score += 10
            
            color_index = (collided_block_index // 10) * 10 + (collided_block_index % 10)
            self._create_particles(block.center, self.BLOCK_COLORS[color_index % len(self.BLOCK_COLORS)])
            
            prev_blocks_destroyed_tier = self.blocks_destroyed_count // 5
            self.blocks_destroyed_count += 1
            if self.blocks_destroyed_count // 5 > prev_blocks_destroyed_tier:
                # sfx: speed_up
                speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
                if speed > 0:
                    self.ball_vel[0] = (self.ball_vel[0] / speed) * (speed + self.BALL_SPEED_INCREMENT)
                    self.ball_vel[1] = (self.ball_vel[1] / speed) * (speed + self.BALL_SPEED_INCREMENT)

            # Collision response
            prev_ball_rect = pygame.Rect(prev_ball_pos[0] - self.BALL_RADIUS, prev_ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if prev_ball_rect.bottom <= block.top or prev_ball_rect.top >= block.bottom:
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1
        
        # --- Miss ---
        if ball_rect.top > self.HEIGHT:
            # sfx: life_lost
            self.lives -= 1
            step_reward -= 1.0
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True

        # --- Particles ---
        self._update_particles()
        
        return step_reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos), "vel": vel, "lifespan": self.np_random.integers(15, 30), "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Blocks
        for i, block in enumerate(self.blocks):
            color_index = (i // 10) * 10 + (i % 10)
            color = self.BLOCK_COLORS[color_index % len(self.BLOCK_COLORS)]
            pygame.draw.rect(self.screen, color, block, border_radius=4)
            darker_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.rect(self.screen, darker_color, block.inflate(-6, -6), border_radius=3)

        # Particles
        for p in self.particles:
            size = max(0, int(p["lifespan"] * 0.25))
            rect = pygame.Rect(int(p["pos"][0] - size/2), int(p["pos"][1] - size/2), size, size)
            pygame.draw.rect(self.screen, p["color"], rect)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        
        # Ball
        if not self.game_over or not self.blocks:
            ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
            glow_radius = self.BALL_RADIUS + 5
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_BALL, 40), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (ball_x - glow_radius, ball_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 12))

        for i in range(self.lives):
            pos = (self.WIDTH - 25 - (i * 25), 25)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, self.COLOR_PADDLE)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, self.COLOR_PADDLE)
        
        if self.game_over:
            msg = "YOU WIN!" if not self.blocks else "GAME OVER"
            end_text = self.large_font.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            bg_rect = text_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker Pong")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        # The other actions are not used in this game
        space_held = 0
        shift_held = 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Frame rate control ---
        clock.tick(30)
        
    pygame.quit()