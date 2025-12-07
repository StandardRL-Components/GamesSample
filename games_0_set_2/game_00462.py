
# Generated: 2025-08-27T13:43:22.534703
# Source Brief: brief_00462.md
# Brief Index: 462

        
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
        "Controls: Use ← and → to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker. Clear all blocks to advance to the next stage. Lose a life if the ball hits the bottom."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_WALL = (180, 180, 180)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (200, 200, 0, 100)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (255, 70, 70), (70, 255, 70), (70, 70, 255),
        (255, 255, 70), (70, 255, 255), (255, 70, 255)
    ]

    # Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WALL_THICKNESS = 10
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_Y_POS = SCREEN_HEIGHT - 30
    BALL_RADIUS = 7
    BLOCK_WIDTH = 58
    BLOCK_HEIGHT = 20
    BLOCK_ROWS = 5
    BLOCK_COLS = 10

    # Game parameters
    PADDLE_SPEED = 10
    BASE_BALL_SPEED = 4.0
    MAX_BALL_X_VEL = 5.0
    MAX_EPISODE_STEPS = 5000
    INITIAL_LIVES = 3
    MAX_STAGES = 3

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.stage = 0
        self.game_over = False
        
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.stage = 1
        self.game_over = False
        self.particles = []

        self._reset_level()

        return self._get_observation(), self._get_info()

    def _reset_level(self):
        """Resets the paddle, ball, and blocks for a new stage or life."""
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) // 2,
            self.PADDLE_Y_POS,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1)
        
        # Start with a random upward direction
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        current_speed = self.BASE_BALL_SPEED + (self.stage - 1) * 0.4
        self.ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)) * current_speed

        if self.stage > 1 and len(self.blocks) == 0: # Only setup blocks if stage is advancing
            self._setup_blocks()
        elif self.stage == 1:
            self._setup_blocks()

    def _setup_blocks(self):
        self.blocks = []
        y_offset = 50
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                block_rect = pygame.Rect(
                    self.WALL_THICKNESS + j * (self.BLOCK_WIDTH + 2) + 10,
                    y_offset + i * (self.BLOCK_HEIGHT + 2),
                    self.BLOCK_WIDTH,
                    self.BLOCK_HEIGHT
                )
                color_index = (i * self.BLOCK_COLS + j) % len(self.BLOCK_COLORS)
                self.blocks.append((block_rect, self.BLOCK_COLORS[color_index]))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement = action[0]

        # 1. Handle player input
        moved = self._move_paddle(movement)
        if moved:
            reward -= 0.01 # Small penalty for movement to encourage efficiency

        # 2. Update game state
        self._update_ball()
        self._update_particles()
        reward += self._handle_collisions()

        # 3. Check for stage clear
        if not self.blocks:
            reward += 50
            self.stage += 1
            if self.stage > self.MAX_STAGES:
                self.game_over = True
                reward += 100 # Win game bonus
            else:
                self._reset_level()
                # sfx: stage_clear_sound()

        self.steps += 1
        terminated = self._check_termination()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _move_paddle(self, movement):
        moved = False
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            moved = True
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            moved = True
        
        self.paddle.x = max(self.WALL_THICKNESS, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH - self.WALL_THICKNESS))
        return moved

    def _update_ball(self):
        self.ball_pos += self.ball_vel

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Walls
        if ball_rect.left <= self.WALL_THICKNESS:
            self.ball_vel.x *= -1
            ball_rect.left = self.WALL_THICKNESS
            # sfx: wall_bounce_sound()
        if ball_rect.right >= self.SCREEN_WIDTH - self.WALL_THICKNESS:
            self.ball_vel.x *= -1
            ball_rect.right = self.SCREEN_WIDTH - self.WALL_THICKNESS
            # sfx: wall_bounce_sound()
        if ball_rect.top <= self.WALL_THICKNESS:
            self.ball_vel.y *= -1
            ball_rect.top = self.WALL_THICKNESS
            # sfx: wall_bounce_sound()

        # Bottom edge (lose life)
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 10
            if self.lives > 0:
                self._reset_level()
                # sfx: lose_life_sound()
            else:
                self.game_over = True
                # sfx: game_over_sound()
            return reward # Exit collision check for this frame

        # Paddle
        if ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
            self.ball_vel.y *= -1
            ball_rect.bottom = self.paddle.top
            
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x = self.MAX_BALL_X_VEL * offset
            
            # Re-normalize to maintain speed
            current_speed = self.ball_vel.length()
            if current_speed > 0:
                self.ball_vel = self.ball_vel.normalize() * current_speed
            
            # Small random nudge to prevent loops
            self.ball_vel.x += self.np_random.uniform(-0.1, 0.1)
            # sfx: paddle_hit_sound()

        # Blocks
        hit_block_idx = ball_rect.collidelist([b[0] for b in self.blocks])
        if hit_block_idx != -1:
            block_rect, color = self.blocks.pop(hit_block_idx)
            reward += 1.1 # +0.1 for hit, +1 for destroy

            # Collision response
            prev_ball_rect = pygame.Rect( (self.ball_pos - self.ball_vel) - (self.BALL_RADIUS, self.BALL_RADIUS), (self.BALL_RADIUS*2, self.BALL_RADIUS*2))
            
            if prev_ball_rect.right <= block_rect.left or prev_ball_rect.left >= block_rect.right:
                 self.ball_vel.x *= -1
            else:
                 self.ball_vel.y *= -1

            self._create_particles(block_rect.center, color)
            # sfx: block_break_sound()

        self.ball_pos.x = ball_rect.centerx
        self.ball_pos.y = ball_rect.centery
        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(10, 25),
                'color': color
            })

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, (20, 30, 50), (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, (20, 30, 50), (0, y), (self.SCREEN_WIDTH, y))

        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.SCREEN_HEIGHT))

        # Blocks
        for block, color in self.blocks:
            pygame.draw.rect(self.screen, color, block)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), block, 2) # Border

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 25.0))
            color_with_alpha = (*p['color'], alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color_with_alpha, (0,0,3,3))
            self.screen.blit(temp_surf, (int(p['pos'].x - 1), int(p['pos'].y - 1)))


        # Ball with glow
        ball_center = (int(self.ball_pos.x), int(self.ball_pos.y))
        glow_surf = pygame.Surface((self.BALL_RADIUS * 4, self.BALL_RADIUS * 4), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, self.BALL_RADIUS*2, self.BALL_RADIUS*2, self.BALL_RADIUS * 2, self.COLOR_BALL_GLOW)
        self.screen.blit(glow_surf, (ball_center[0] - self.BALL_RADIUS*2, ball_center[1] - self.BALL_RADIUS*2), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)


    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.SCREEN_HEIGHT - 35))

        lives_text = self.font_large.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - self.WALL_THICKNESS - 10, self.SCREEN_HEIGHT - 35))
        
        stage_text = self.font_large.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, ( (self.SCREEN_WIDTH - stage_text.get_width())//2, 15))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_message = "YOU WIN!" if self.stage > self.MAX_STAGES else "GAME OVER"
            end_text = self.font_large.render(end_message, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(end_text, end_rect)

            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, (200, 200, 200))
            final_score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)


    def _get_info(self):
        self.score = int(self.score) # Ensure score is an integer for info dict
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
        }
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Match the intended FPS

    env.close()
    pygame.quit()