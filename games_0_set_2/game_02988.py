
# Generated: 2025-08-27T22:01:22.378888
# Source Brief: brief_02988.md
# Brief Index: 2988

        
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
    """
    A fast-paced, top-down block breaker where strategic paddle positioning and
    risk-taking are rewarded. Clear all blocks to win, but lose all your balls and you lose.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A retro neon block-breaker. Use the paddle to deflect the ball, "
        "destroy all the blocks, and achieve a high score. Don't let the ball fall!"
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    BALL_SPEED = 7
    MAX_STEPS = 2500
    INITIAL_LIVES = 3

    # --- Colors (Neon Theme) ---
    COLOR_BG = (10, 5, 30)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_PADDLE_GLOW = (200, 200, 255, 60)
    COLOR_BALL = (0, 255, 255)
    COLOR_BALL_GLOW = (0, 200, 255, 100)
    COLOR_WALL = (100, 100, 150)
    COLOR_TEXT = (220, 220, 255)
    BLOCK_COLORS = [
        (255, 0, 128), (255, 128, 0), (255, 255, 0),
        (0, 255, 0), (0, 128, 255)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables are initialized in reset()
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_on_paddle = None
        self.blocks = None
        self.particles = None
        self.steps = None
        self.score = None
        self.lives = None
        self.game_over = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False

        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.particles = []
        self._reset_ball()
        self._create_blocks()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        self.steps += 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        else: # No-op penalty
            reward -= 0.02

        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        if self.ball_on_paddle and space_held:
            # Launch ball
            angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
            self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED
            self.ball_on_paddle = False
            # Sound: Ball launch

        # --- Update Game Logic ---
        if self.ball_on_paddle:
            self.ball_pos.x = self.paddle.centerx
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
        else:
            self.ball_pos += self.ball_vel
            reward_from_collisions, terminated_by_collision = self._handle_collisions()
            reward += reward_from_collisions
            if terminated_by_collision:
                self.game_over = True

        self._update_particles()

        # --- Check Termination Conditions ---
        terminated = self.game_over
        if not self.blocks: # All blocks cleared
            self.score += 100
            reward += 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _reset_ball(self):
        self.ball_on_paddle = True
        self.ball_pos = pygame.Vector2(
            self.paddle.centerx, self.paddle.top - self.BALL_RADIUS
        )
        self.ball_vel = pygame.Vector2(0, 0)

    def _create_blocks(self):
        self.blocks = []
        block_width = 58
        block_height = 20
        gap = 6
        rows = 5
        cols = 10
        start_x = (self.WIDTH - (cols * (block_width + gap) - gap)) // 2
        start_y = 50

        for i in range(rows):
            for j in range(cols):
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                rect = pygame.Rect(
                    start_x + j * (block_width + gap),
                    start_y + i * (block_height + gap),
                    block_width,
                    block_height,
                )
                self.blocks.append({"rect": rect, "color": color})

    def _handle_collisions(self):
        reward = 0
        terminated = False

        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)

        # Walls
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # Sound: Wall bounce
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
            # Sound: Wall bounce

        # Floor (lose life)
        if self.ball_pos.y + self.BALL_RADIUS >= self.HEIGHT:
            self.lives -= 1
            # Sound: Lose life
            if self.lives <= 0:
                reward -= 100 # Terminal penalty
                terminated = True
            else:
                self._reset_ball()

        # Paddle
        if ball_rect.colliderect(self.paddle):
            # Sound: Paddle bounce
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS - 1

            # Add spin based on hit location
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * 2.5
            # Re-normalize speed
            self.ball_vel.scale_to_length(self.BALL_SPEED)

        # Blocks
        hit_block_idx = ball_rect.collidelistall([b["rect"] for b in self.blocks])
        if hit_block_idx:
            for idx in sorted(hit_block_idx, reverse=True):
                block_item = self.blocks.pop(idx)
                block = block_item["rect"]
                # Sound: Block break
                self.score += 1
                reward += 1.1 # +1 for breaking, +0.1 continuous

                # Create particles
                self._create_particles(block.center, block_item["color"])

                # Determine bounce direction
                # A simple but effective method: check if the collision was more horizontal or vertical
                overlap = ball_rect.clip(block)
                if overlap.width < overlap.height:
                    self.ball_vel.x *= -1
                else:
                    self.ball_vel.y *= -1
            # To prevent getting stuck inside multiple blocks
            self.ball_vel.scale_to_length(self.BALL_SPEED)

        return reward, terminated

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([pygame.Vector2(pos), vel, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1] # Update position
            p[2] -= 1 # Decrease lifetime
        self.particles = [p for p in self.particles if p[2] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_left": len(self.blocks),
        }

    def _render_game(self):
        # Draw particles
        for pos, vel, life, color in self.particles:
            alpha = max(0, min(255, int(255 * (life / 30.0))))
            radius = int(self.BALL_RADIUS / 2 * (life / 30.0))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), radius, (*color, alpha))

        # Draw blocks
        for block_item in self.blocks:
            block = block_item["rect"]
            color = block_item["color"]
            pygame.draw.rect(self.screen, color, block, border_radius=3)
            # Inner bevel effect for depth
            light_color = tuple(min(255, c + 50) for c in color)
            pygame.draw.rect(self.screen, light_color, block.inflate(-6, -6), border_radius=2)


        # Draw paddle with glow
        glow_rect = self.paddle.inflate(10, 10)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, self.COLOR_PADDLE_GLOW, glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Draw ball with glow
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_large.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 180, 10))
        for i in range(self.lives):
            pos_x = self.WIDTH - 80 + (i * (self.BALL_RADIUS * 2 + 5))
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 28, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos_x, 28, self.BALL_RADIUS, self.COLOR_BALL)

        # Game Over Message
        if self.game_over:
            win = not self.blocks
            msg = "YOU WIN!" if win else "GAME OVER"
            color = (0, 255, 128) if win else (255, 0, 100)
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            game_over_text = self.font_large.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(game_over_text, text_rect)

            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 20))
            self.screen.blit(final_score_text, score_rect)

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
    # --- Example of how to run the environment ---
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame window for human play testing ---
    human_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    
    obs, info = env.reset()
    done = False
    
    # Game loop for human play
    while not done:
        # --- Map keyboard inputs to action space ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Process Pygame events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to the human-visible screen ---
        # The observation is (H, W, C), but pygame needs (W, H, C) surface
        # So we need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Control frame rate ---
        env.clock.tick(30)
        
    print(f"Game Over! Final Info: {info}")
    env.close()