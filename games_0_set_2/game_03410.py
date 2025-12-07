
# Generated: 2025-08-27T23:16:12.012945
# Source Brief: brief_03410.md
# Brief Index: 3410

        
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
        "Controls: Use ← and → to move the paddle. Break all the blocks to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro block-breaker. Use the paddle to keep the ball in play, break blocks for points, and clear the screen to win. Ball speed increases as you progress!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 12
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 4.0
    MAX_BALL_SPEED = 8.0
    MAX_EPISODE_STEPS = 10000

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_PADDLE = (240, 240, 240)
    COLOR_BALL = (255, 255, 0)
    COLOR_UI_TEXT = (200, 200, 220)
    BLOCK_COLORS = {
        10: (0, 200, 100),   # Green
        20: (0, 150, 255),   # Blue
        30: (255, 80, 80),    # Red
    }

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.render_mode = render_mode
        self.np_random = None

        # These attributes are defined in reset()
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.blocks_broken_count = None
        self.particles = None
        self.ball_trail = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1], dtype=float)
        
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        self.ball_vel = np.array([math.sin(angle), -math.cos(angle)]) * self.INITIAL_BALL_SPEED
        
        self.blocks = self._generate_blocks()
        self.lives = 3
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.blocks_broken_count = 0
        self.particles = []
        self.ball_trail = []
        
        return self._get_observation(), self._get_info()

    def _generate_blocks(self):
        blocks = []
        block_width = 50
        block_height = 20
        gap = 4
        rows = 5
        cols = self.SCREEN_WIDTH // (block_width + gap)
        start_x = (self.SCREEN_WIDTH - cols * (block_width + gap) + gap) / 2
        start_y = 50

        point_values = list(self.BLOCK_COLORS.keys())

        for r in range(rows):
            for c in range(cols):
                points = point_values[r % len(point_values)]
                color = self.BLOCK_COLORS[points]
                x = start_x + c * (block_width + gap)
                y = start_y + r * (block_height + gap)
                blocks.append({
                    "rect": pygame.Rect(x, y, block_width, block_height),
                    "color": color,
                    "points": points
                })
        return blocks

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.02  # Small time penalty

        # 1. Handle Input & Update Paddle
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        # 2. Update Ball
        self._update_ball()
        
        # 3. Handle Collisions
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # 4. Update Particles and Trail
        self._update_particles()
        self._update_trail()

        # 5. Check Game State
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.lives <= 0:
                reward -= 100
            elif not self.blocks:
                reward += 100
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        self.ball_trail.append(list(self.ball_pos))
        if len(self.ball_trail) > 10:
            self.ball_trail.pop(0)

        self.ball_pos += self.ball_vel

        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce

        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.SCREEN_HEIGHT)
            # sfx: wall_bounce

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            
            # Influence horizontal velocity based on hit location
            offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = offset * self.INITIAL_BALL_SPEED * 1.2
            
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # Prevent sticking
            # sfx: paddle_hit

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                reward += 0.1 # Hit feedback
                reward += block["points"]
                self.score += block["points"]
                self._spawn_particles(block["rect"].center, block["color"], 20)
                self.blocks.remove(block)
                # sfx: block_break

                # Determine bounce direction
                prev_ball_pos = self.ball_pos - self.ball_vel
                prev_ball_rect = pygame.Rect(prev_ball_pos[0] - self.BALL_RADIUS, prev_ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
                
                if prev_ball_rect.bottom <= block["rect"].top or prev_ball_rect.top >= block["rect"].bottom:
                    self.ball_vel[1] *= -1
                else:
                    self.ball_vel[0] *= -1

                # Increase difficulty
                self.blocks_broken_count += 1
                if self.blocks_broken_count % 10 == 0:
                    speed_multiplier = min(self.MAX_BALL_SPEED / np.linalg.norm(self.ball_vel), 1.05)
                    self.ball_vel *= speed_multiplier
                break # Only handle one block collision per frame
        
        # Ensure ball speed doesn't get too slow or too fast
        current_speed = np.linalg.norm(self.ball_vel)
        if current_speed > 0: # Avoid division by zero
            if current_speed < self.INITIAL_BALL_SPEED * 0.8:
                self.ball_vel = self.ball_vel * (self.INITIAL_BALL_SPEED * 0.8 / current_speed)
            if current_speed > self.MAX_BALL_SPEED:
                self.ball_vel = self.ball_vel * (self.MAX_BALL_SPEED / current_speed)

        # Lose life
        if self.ball_pos[1] > self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 10
            # sfx: lose_life
            if self.lives > 0:
                self._reset_ball()
        
        return reward

    def _reset_ball(self):
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1], dtype=float)
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        self.ball_vel = np.array([math.sin(angle), -math.cos(angle)]) * self.INITIAL_BALL_SPEED
        self.ball_trail.clear()

    def _check_termination(self):
        return self.lives <= 0 or not self.blocks or self.steps >= self.MAX_EPISODE_STEPS

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "radius": self.np_random.uniform(2, 5),
                "color": color,
                "lifespan": 20
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            p["radius"] -= 0.2
            if p["lifespan"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _update_trail(self):
        # Trail is updated in _update_ball
        pass

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_background_grid()
        self._draw_blocks()
        self._draw_particles()
        self._draw_ball_and_trail()
        self._draw_paddle()
        
    def _draw_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_blocks(self):
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            # Add a subtle highlight
            highlight_color = tuple(min(255, c + 40) for c in block["color"])
            pygame.draw.rect(self.screen, highlight_color, (block["rect"].x, block["rect"].y, block["rect"].width, 2), border_top_left_radius=3, border_top_right_radius=3)

    def _draw_paddle(self):
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        highlight_color = (255, 255, 255)
        pygame.draw.rect(self.screen, highlight_color, (self.paddle.x, self.paddle.y, self.paddle.width, 2), border_top_left_radius=4, border_top_right_radius=4)


    def _draw_ball_and_trail(self):
        # Trail
        for i, pos in enumerate(self.ball_trail):
            alpha = int(255 * (i / len(self.ball_trail)))
            color = (self.COLOR_BALL[0], self.COLOR_BALL[1], self.COLOR_BALL[2], alpha)
            radius = self.BALL_RADIUS * (i / len(self.ball_trail))
            if radius > 1:
                self._draw_transparent_circle(int(pos[0]), int(pos[1]), int(radius), color)

        # Main ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)

    def _draw_particles(self):
        for p in self.particles:
            color_with_alpha = (*p["color"], max(0, int(255 * (p["lifespan"] / 20))))
            self._draw_transparent_circle(int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), color_with_alpha)

    def _draw_transparent_circle(self, x, y, radius, color):
        temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color, (radius, radius), radius)
        self.screen.blit(temp_surf, (x - radius, y - radius))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        heart_size = 15
        for i in range(self.lives):
            x = self.SCREEN_WIDTH - (i + 1) * (heart_size + 5) - 5
            y = 10 + heart_size // 2
            points = [
                (x, y),
                (x - heart_size // 2, y - heart_size // 2),
                (x - heart_size, y),
                (x - heart_size, y + heart_size // 2),
                (x - heart_size // 2, y + heart_size),
                (x, y + heart_size // 2)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, (255, 50, 50))
            pygame.gfxdraw.filled_polygon(self.screen, points, (255, 50, 50))

        if self.game_over:
            msg = "YOU WIN!" if not self.blocks else "GAME OVER"
            text_surf = self.font_game_over.render(msg, True, self.COLOR_PADDLE)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
            
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
        
        # Test game-specific assertions
        self.reset()
        self.paddle.x = -100
        self.step(self.action_space.sample())
        assert self.paddle.x >= 0

        self.paddle.x = self.SCREEN_WIDTH + 100
        self.step(self.action_space.sample())
        assert self.paddle.right <= self.SCREEN_WIDTH

        initial_block_count = len(self.blocks)
        ball_rect = self.blocks[0]["rect"].copy()
        ball_rect.x += 1
        ball_rect.y += 1
        self.ball_pos = np.array(ball_rect.center, dtype=float)
        self.ball_vel = np.array([0, 1], dtype=float)
        self._handle_collisions()
        assert len(self.blocks) == initial_block_count - 1

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a different screen for display to avoid conflicts with headless rendering
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker Test")
    
    terminated = False
    total_reward = 0
    
    # Map keyboard keys to actions
    key_to_action = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4
    }
    
    action = env.action_space.sample()
    action[0] = 0 # Default to no-op
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get key presses for manual control
        keys = pygame.key.get_pressed()
        current_move_action = 0 # No-op
        if keys[pygame.K_LEFT]:
            current_move_action = 3
        elif keys[pygame.K_RIGHT]:
            current_move_action = 4
            
        action = [current_move_action, 0, 0] # Movement, space, shift
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause before resetting
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(30) # Control the frame rate

    env.close()