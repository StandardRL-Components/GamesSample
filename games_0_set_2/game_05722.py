import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A retro arcade block breaker. Clear all the blocks by deflecting the ball with your paddle. "
        "Don't let the ball fall!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # --- Colors ---
        self.COLOR_BG = (26, 26, 46)  # #1A1A2E
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_WALL = (200, 200, 220)
        self.COLOR_TEXT = (230, 230, 230)
        self.BLOCK_COLORS = [
            (255, 51, 102),  # Red
            (51, 255, 153),  # Green
            (51, 153, 255),  # Blue
            (255, 255, 102),  # Yellow
            (255, 102, 255),  # Magenta
        ]

        # --- Game Constants ---
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 12
        self.PADDLE_SPEED = 15
        self.BALL_RADIUS = 8
        self.BALL_SPEED_MAGNITUDE = 6.0
        self.MAX_STEPS = 1000
        self.INITIAL_LIVES = 3

        # --- Game State ---
        # These are initialized properly in reset()
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_state = None
        self.blocks = None
        self.particles = None
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_block_hit_step = -10
        self.combo_chain = 0
        self.current_reward = 0.0

        # Self-check to ensure implementation correctness
        # self.validate_implementation() # Commented out for submission, as it requires a display

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Paddle ---
        paddle_y = self.screen_height - 40
        paddle_x = (self.screen_width - self.PADDLE_WIDTH) / 2
        self.paddle_rect = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # --- Reset Ball ---
        self._reset_ball()

        # --- Reset Blocks ---
        self.blocks = []
        block_width = 60
        block_height = 20
        block_spacing = 4
        num_cols = 10
        num_rows = 5
        start_x = (self.screen_width - (num_cols * (block_width + block_spacing) - block_spacing)) / 2
        start_y = 60
        for i in range(num_rows):
            for j in range(num_cols):
                x = start_x + j * (block_width + block_spacing)
                y = start_y + i * (block_height + block_spacing)
                rect = pygame.Rect(x, y, block_width, block_height)
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                self.blocks.append({"rect": rect, "color": color})

        # --- Reset Game State ---
        self.lives = self.INITIAL_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.last_block_hit_step = -10
        self.combo_chain = 0

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_state = "ready"
        self.ball_pos = [0, 0]  # Position is relative to paddle when ready
        self.ball_vel = [0, 0]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.current_reward = -0.02  # Time penalty

        self._handle_input(action)
        self._update_game_logic()

        self.steps += 1
        terminated = self._check_termination()

        # Add terminal rewards
        if terminated:
            if self.win:
                self.current_reward += 100
            elif self.lives <= 0:
                self.current_reward += -100

        reward = self.current_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Paddle Movement
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED

        self.paddle_rect.x = max(0, min(self.screen_width - self.PADDLE_WIDTH, self.paddle_rect.x))

        # Ball Launch
        if space_held and self.ball_state == "ready":
            # Sound: Ball Launch
            self.ball_state = "moving"
            launch_angle = math.pi * 0.75  # Default up-left

            # Aim based on paddle position
            paddle_center = self.paddle_rect.centerx
            center_offset = (paddle_center - self.screen_width / 2) / (self.screen_width / 2)  # -1 to 1
            launch_angle = math.pi * (0.75 - 0.5 * center_offset)  # Range from 1/4 pi to 3/4 pi

            self.ball_vel = [
                self.BALL_SPEED_MAGNITUDE * math.cos(launch_angle),
                -self.BALL_SPEED_MAGNITUDE * math.sin(launch_angle)
            ]
            self.ball_pos = [self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS]

    def _update_game_logic(self):
        if self.ball_state == "moving":
            self._update_ball()
        self._update_particles()

        # Update combo timer
        if self.steps - self.last_block_hit_step > 3:  # ~0.1s at 30fps
            self.combo_chain = 0

    def _update_ball(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS,
                                self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left < 0 or ball_rect.right > self.screen_width:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.screen_width, ball_rect.right)
            self.ball_pos[0] = ball_rect.centerx
            # Sound: Wall Bounce
        if ball_rect.top < 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            self.ball_pos[1] = ball_rect.centery
            # Sound: Wall Bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel[1] > 0:
            # Sound: Paddle Hit
            self.ball_vel[1] *= -1

            # Influence horizontal velocity based on hit location
            hit_offset = self.ball_pos[0] - self.paddle_rect.centerx
            self.ball_vel[0] += hit_offset * 0.1

            # Normalize speed
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_SPEED_MAGNITUDE
                self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_SPEED_MAGNITUDE

            # Penalize ineffective hits (too vertical)
            if abs(self.ball_vel[0]) < 1.0:
                self.current_reward -= 2.0

            self.ball_pos[1] = self.paddle_rect.top - self.BALL_RADIUS

        # Block collisions
        hit_block = None
        for block in self.blocks:
            if ball_rect.colliderect(block["rect"]):
                hit_block = block
                break

        if hit_block:
            # Sound: Block Break
            self.blocks.remove(hit_block)
            self.score += 10
            self.current_reward += 1.0

            # Combo logic
            if self.steps - self.last_block_hit_step <= 3:
                self.combo_chain += 1
                if self.combo_chain >= 2:
                    self.score += 20 * self.combo_chain
                    self.current_reward += 5.0
                    # Sound: Combo Bonus
            else:
                self.combo_chain = 1
            self.last_block_hit_step = self.steps

            self._create_particles(hit_block["rect"].center, hit_block["color"])

            # Bounce logic
            clip_rect = ball_rect.clip(hit_block["rect"])
            if clip_rect.width > clip_rect.height:
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1

        # Ball lost
        if ball_rect.top > self.screen_height:
            # Sound: Ball Lost
            self.lives -= 1
            self._reset_ball()
            if self.lives > 0:
                self.current_reward -= 2.0  # Penalty for losing a ball

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "color": color,
                "life": 20
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if not self.blocks:  # Win condition
            self.game_over = True
            self.win = True
        elif self.lives <= 0:  # Lose condition
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:  # Max steps
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, block["rect"], 1)  # Border

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)

        # Draw ball
        if self.ball_state == "ready":
            pos = (self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS)
        else:
            pos = (int(self.ball_pos[0]), int(self.ball_pos[1]))

        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        # FIX: filled_circle takes 5 arguments: surface, x, y, radius, color.
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (255, 255, 255, 60))
        self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, p["life"] * 12)
            color = (*p["color"], alpha)
            radius = int(max(0, p["life"] / 5))
            if radius > 0:
                pos = (int(p["pos"][0]), int(p["pos"][1]))
                # Create a temporary surface for alpha blending
                particle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(particle_surf, radius, radius, radius, color)
                self.screen.blit(particle_surf, (pos[0] - radius, pos[1] - radius))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            pos = (self.screen_width - 20 - i * (self.BALL_RADIUS * 2 + 5), 10 + self.BALL_RADIUS)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            message = "YOU WIN!" if self.win else "GAME OVER"
            text_surf = self.font_large.render(message, True, self.COLOR_PADDLE)
            text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(text_surf, text_rect)

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
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        obs, _ = self.reset()
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8

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
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls
    # Not part of the Gymnasium environment, but useful for testing
    
    # Set a real video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a display surface
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((640, 400))

    # Game loop
    while not done:
        # --- Action selection ---
        movement = 0  # No-op by default
        space_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space_held = 1

        action = [movement, space_held, 0]  # Shift is not used

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering for human play ---
        # The observation is already the rendered frame
        # We just need to display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Frame rate ---
        env.clock.tick(30)  # Limit to 30 FPS for consistent gameplay

    print(f"Game Over! Final Info: {info}")
    env.close()