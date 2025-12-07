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


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-arcade block breaker. Destroy all blocks to win, but don't lose your last ball! Hitting multiple blocks in a row builds a combo score bonus."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500  # Approx 50 seconds

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PADDLE = (0, 180, 255)
        self.COLOR_PADDLE_SHADOW = (0, 120, 190)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BALL_GLOW = (200, 220, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.BLOCK_COLORS = {
            10: (0, 200, 100),  # Green
            20: (255, 200, 0),  # Yellow
            30: (230, 50, 50),  # Red
        }

        # Game parameters
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.BALL_MAX_SPEED = 10
        self.BALL_MIN_SPEED_Y = 3.5
        self.INITIAL_LIVES = 3

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = None
        self.blocks = None
        self.particles = None
        self.lives = 0
        self.score = 0
        self.combo = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False

        # self.reset() # This is called by the wrapper, but good practice to have it here for standalone use
        # self.validate_implementation() # For self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.combo = 0
        self.game_over = False
        self.last_space_held = False
        self.particles = []

        self._create_paddle()
        self._create_blocks()
        self._reset_ball()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_pressed = action[1] == 1 and not self.last_space_held
            self.last_space_held = action[1] == 1

            # --- Update Game Logic ---
            self._handle_input(movement, space_pressed)

            prev_combo = self.combo
            ball_hit_block, block_reward = self._update_ball()

            if ball_hit_block:
                self.combo += 1
                reward += 10  # Reward for destroying a block
                reward += block_reward  # Points value of the block
                reward += 5 * self.combo  # Combo bonus
            elif self.combo > 0 and prev_combo == self.combo:
                # If combo didn't increase (e.g. ball hit paddle), reset it
                self.combo = 0

            self._update_particles()

        # --- Check Termination ---
        win = len(self.blocks) == 0
        lose = self.lives <= 0
        timeout = self.steps >= self.MAX_STEPS
        terminated = win or lose or timeout

        if terminated and not self.game_over:
            if win:
                reward += 100
            if lose:
                reward -= 100
            self.game_over = True

        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _create_paddle(self):
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

    def _create_blocks(self):
        self.blocks = []
        block_width, block_height = 40, 20
        rows, cols = 5, 14

        for r in range(rows):
            for c in range(cols):
                points = [10, 10, 20, 20, 30][r]
                color = self.BLOCK_COLORS[points]
                rect = pygame.Rect(
                    c * (block_width + 5) + 20,
                    r * (block_height + 5) + 40,
                    block_width,
                    block_height,
                )
                self.blocks.append({"rect": rect, "color": color, "points": points})

    def _reset_ball(self):
        self.ball_attached = True
        self.ball_pos = np.array(
            [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=np.float64
        )
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float64)
        self.combo = 0

    def _handle_input(self, movement, space_pressed):
        # Paddle movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED

        self.paddle.clamp_ip(self.screen.get_rect())

        # Ball launch
        if self.ball_attached and space_pressed:
            # sfx: launch_ball.wav
            self.ball_attached = False
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            initial_speed = self.BALL_MAX_SPEED * 0.7
            self.ball_vel = np.array(
                [math.cos(angle) * initial_speed, math.sin(angle) * initial_speed]
            )

    def _update_ball(self):
        if self.ball_attached:
            self.ball_pos[0] = self.paddle.centerx
            return False, 0

        self.ball_pos += self.ball_vel

        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS,
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2,
        )

        # Wall collisions
        if ball_rect.left <= 0:
            ball_rect.left = 0
            self.ball_vel[0] *= -1
            # sfx: bounce_wall.wav
        if ball_rect.right >= self.WIDTH:
            ball_rect.right = self.WIDTH
            self.ball_vel[0] *= -1
            # sfx: bounce_wall.wav
        if ball_rect.top <= 0:
            ball_rect.top = 0
            self.ball_vel[1] *= -1
            # sfx: bounce_wall.wav

        # Floor collision (lose life)
        if ball_rect.top >= self.HEIGHT:
            self.lives -= 1
            # sfx: lose_life.wav
            if self.lives > 0:
                self._reset_ball()
            return False, 0

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: bounce_paddle.wav
            self.ball_vel[1] *= -1
            # Add spin based on where it hit the paddle
            hit_offset = (ball_rect.centerx - self.paddle.centerx) / (
                self.PADDLE_WIDTH / 2
            )
            self.ball_vel[0] += hit_offset * 3.0

            # Ensure vertical speed is not too low
            self.ball_vel[1] = -max(abs(self.ball_vel[1]), self.BALL_MIN_SPEED_Y)

            # Clamp speed
            speed = np.linalg.norm(self.ball_vel)
            if speed > self.BALL_MAX_SPEED:
                self.ball_vel = self.ball_vel / speed * self.BALL_MAX_SPEED

            # Reset combo on paddle hit
            self.combo = 0

        # Block collision
        hit_index = ball_rect.collidelist([b["rect"] for b in self.blocks])
        if hit_index != -1:
            # sfx: break_block.wav
            block_hit = self.blocks.pop(hit_index)
            self.score += block_hit["points"]

            # Determine collision side to correctly reverse velocity
            # A simple approach is usually good enough for this genre
            self.ball_vel[1] *= -1

            self._create_particles(ball_rect.center, block_hit["color"])
            return True, block_hit["points"]

        self.ball_pos[0] = ball_rect.centerx
        self.ball_pos[1] = ball_rect.centery
        return False, 0

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 20)
            self.particles.append(
                {"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color}
            )

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # --- Render Game Elements ---
        self._render_particles()
        self._render_blocks()
        self._render_paddle()
        self._render_ball()

        # --- Render UI Overlay ---
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 20.0))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, temp_surf.get_rect())
            self.screen.blit(temp_surf, (int(p["pos"][0]), int(p["pos"][1])))

    def _render_blocks(self):
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            # Add a slight 3D effect
            shadow_color = tuple(max(0, c - 40) for c in block["color"])
            pygame.draw.rect(
                self.screen, shadow_color, block["rect"].move(2, 2), border_radius=3
            )

    def _render_paddle(self):
        # Shadow
        pygame.draw.rect(
            self.screen, self.COLOR_PADDLE_SHADOW, self.paddle.move(0, 3), border_radius=5
        )
        # Main paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

    def _render_ball(self):
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        # FIX: filled_circle requires 5 arguments: surface, x, y, radius, color.
        # The 'radius' argument was missing.
        pygame.gfxdraw.filled_circle(
            glow_surf,
            glow_radius,
            glow_radius,
            glow_radius,
            (*self.COLOR_BALL_GLOW, 50),
        )
        self.screen.blit(glow_surf, (x - glow_radius, y - glow_radius))

        # Ball
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, pos, color, shadow_color):
            shadow = font.render(text, True, shadow_color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            content = font.render(text, True, color)
            self.screen.blit(content, pos)

        # Score
        score_text = f"SCORE: {self.score}"
        draw_text(
            score_text, self.font_medium, (10, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW
        )

        # Lives
        lives_text = f"BALLS: {self.lives}"
        draw_text(
            lives_text,
            self.font_medium,
            (self.WIDTH - 120, 10),
            self.COLOR_TEXT,
            self.COLOR_TEXT_SHADOW,
        )

        # Combo
        if self.combo > 1:
            combo_text = f"COMBO x{self.combo}"
            text_surf = self.font_large.render(combo_text, True, self.COLOR_TEXT)
            pos = ((self.WIDTH - text_surf.get_width()) / 2, self.HEIGHT / 2 - 100)
            draw_text(
                combo_text, self.font_large, pos, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW
            )

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if len(self.blocks) == 0 else "GAME OVER"
            color = (100, 255, 100) if len(self.blocks) == 0 else (255, 100, 100)
            text_surf = self.font_large.render(message, True, color)
            pos = (
                (self.WIDTH - text_surf.get_width()) / 2,
                (self.HEIGHT - text_surf.get_height()) / 2,
            )
            draw_text(
                message, self.font_large, pos, color, self.COLOR_TEXT_SHADOW
            )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
            "combo": self.combo,
        }

    def close(self):
        pygame.quit()

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
    # This block allows you to play the game directly
    # It's a demonstration of how to use the environment
    # For human play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()

    # Set up the display window
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    terminated = False

    print(env.user_guide)

    while not terminated:
        # --- Human Controls ---
        movement = 0  # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space_held = 1

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        env.clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    env.close()