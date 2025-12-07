import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set Pygame to run in headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple particle for effects."""
    def __init__(self, x, y, color, size, life, angle, speed):
        self.pos = pygame.Vector2(x, y)
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life
        
        self.vel = pygame.Vector2(speed, 0).rotate(angle)

    def update(self):
        self.pos += self.vel
        self.life -= 1
        self.size = max(0, self.size - 0.1)
        self.vel *= 0.98 # friction

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            temp_surf = pygame.Surface((int(self.size) * 2, int(self.size) * 2), pygame.SRCALPHA)
            color_with_alpha = self.color + (alpha,)
            pygame.draw.rect(temp_surf, color_with_alpha, temp_surf.get_rect())
            surface.blit(temp_surf, (int(self.pos.x - self.size), int(self.pos.y - self.size)), special_flags=pygame.BLEND_RGBA_ADD)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A fast-paced, top-down block breaker where risky plays are rewarded. Destroy all blocks to win."
    )

    auto_advance = True

    # --- Game Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 5000
    
    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_PADDLE = (240, 240, 240)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (200, 200, 0)
    BLOCK_COLORS = [(255, 0, 128), (0, 255, 255), (0, 255, 0), (255, 128, 0)]
    COLOR_UI_TEXT = (220, 220, 220)

    # Paddle
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_Y_POS = HEIGHT - 30
    PADDLE_SPEED = 10

    # Ball
    BALL_RADIUS = 8
    BALL_SPEED = 7

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 60)
        
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = None
        self.blocks = None
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False
        
        self.prev_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.win = False
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.PADDLE_Y_POS,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self.ball_attached = True
        self._reset_ball()

        self.blocks = self._create_blocks()
        self.particles = []
        self.prev_space_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward_events = []

        self._handle_input(movement, space_held)
        self._update_ball()
        self._update_particles()
        
        block_reward = self._handle_block_collisions()
        if block_reward > 0:
            reward_events.append(block_reward)
        
        if self._check_life_lost():
            self.lives -= 1
            if self.lives > 0:
                self.ball_attached = True
                self._reset_ball()
            else:
                self.game_over = True
                self.win = False
        
        self.steps += 1
        terminated = self._check_termination()
        
        reward = self._calculate_reward(reward_events, terminated)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        if self.ball_attached and space_held and not self.prev_space_held:
            # sfx: launch_ball.wav
            self.ball_attached = False
            # FIX: np.random.uniform(low, high) requires low <= high.
            launch_angle = self.np_random.uniform(-120, -60)
            self.ball_vel = pygame.Vector2(self.BALL_SPEED, 0).rotate(launch_angle)
        
        self.prev_space_held = space_held

    def _reset_ball(self):
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)

    def _update_ball(self):
        if self.ball_attached:
            self.ball_pos.x = self.paddle.centerx
            return

        self.ball_pos += self.ball_vel

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce.wav
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
            # sfx: wall_bounce.wav

        # Paddle collision
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel.y > 0:
            # sfx: paddle_bounce.wav
            self.ball_vel.y *= -1
            
            # Change angle based on hit position
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * 3
            
            # Clamp velocity to prevent runaway speeds
            self.ball_vel.scale_to_length(self.BALL_SPEED)
            
            # Ensure ball is above paddle to prevent sticking
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _handle_block_collisions(self):
        if self.ball_attached:
            return 0
            
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        
        for block in self.blocks[:]:
            if block['rect'].colliderect(ball_rect):
                # sfx: block_break.wav
                self.blocks.remove(block)
                self.score += 1
                
                # Spawn particles
                for _ in range(15):
                    angle = self.np_random.uniform(0, 360)
                    speed = self.np_random.uniform(1, 4)
                    size = self.np_random.uniform(3, 8)
                    life = self.np_random.integers(15, 30)
                    self.particles.append(Particle(self.ball_pos.x, self.ball_pos.y, block['color'], size, life, angle, speed))
                
                # Determine bounce direction
                prev_ball_pos = self.ball_pos - self.ball_vel
                prev_ball_rect = pygame.Rect(prev_ball_pos.x - self.BALL_RADIUS, prev_ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)

                # Simple but effective bounce logic
                if (prev_ball_rect.right <= block['rect'].left or prev_ball_rect.left >= block['rect'].right):
                     self.ball_vel.x *= -1
                else:
                     self.ball_vel.y *= -1

                return 1 # Reward for breaking a block
        return 0

    def _check_life_lost(self):
        return self.ball_pos.y + self.BALL_RADIUS > self.HEIGHT

    def _check_termination(self):
        if not self.blocks:
            self.game_over = True
            self.win = True
        if self.lives <= 0:
            self.game_over = True
            self.win = False
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            # If time runs out, it's not a win
            if not self.win:
                self.win = False

        return self.game_over

    def _calculate_reward(self, event_rewards, terminated):
        total_reward = sum(event_rewards) - 0.02
        
        if terminated:
            if self.win:
                total_reward += 100
            elif self.lives <= 0:
                total_reward -= 100
        
        return total_reward

    def _create_blocks(self):
        blocks = []
        num_rows = 5
        num_cols = 10
        block_width = 58
        block_height = 20
        total_grid_width = num_cols * (block_width + 4)
        start_x = (self.WIDTH - total_grid_width) / 2
        start_y = 50

        for i in range(num_rows):
            for j in range(num_cols):
                color = self.BLOCK_COLORS[(i + j) % len(self.BLOCK_COLORS)]
                rect = pygame.Rect(
                    start_x + j * (block_width + 4),
                    start_y + i * (block_height + 4),
                    block_width,
                    block_height
                )
                blocks.append({'rect': rect, 'color': color})
        return blocks

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_game(self):
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block['color']), block['rect'], 2)
        
        for p in self.particles:
            p.draw(self.screen)
            
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball glow
        glow_radius = int(self.BALL_RADIUS * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW + (50,), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (int(self.ball_pos.x - glow_radius), int(self.ball_pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Ball
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                end_text = self.font_game_over.render("YOU WIN!", True, self.COLOR_BALL)
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.BLOCK_COLORS[0])
            
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_left": len(self.blocks),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Helper function to check if the implementation follows the Gymnasium API."""
        print("Validating implementation...")
        # Check spaces
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Check reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Check step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # Unset the dummy video driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    
    # Run validation
    try:
        env.validate_implementation()
    except Exception as e:
        print(f"Implementation validation failed: {e}")
        env.close()
        exit()

    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    pygame.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            env.reset()

        # --- Rendering ---
        # The observation is already a rendered frame
        # So we convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling for quitting ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(60)
        
    env.close()