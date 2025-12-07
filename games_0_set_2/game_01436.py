
# Generated: 2025-08-27T17:08:41.516340
# Source Brief: brief_01436.md
# Brief Index: 1436

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro arcade game. Bounce the ball to break all the blocks and clear three stages before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME_SECONDS = 180
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (15, 15, 40)
        self.COLOR_GRID = (30, 30, 60)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 50, 50), (50, 255, 50), (50, 50, 255),
            (255, 255, 50), (50, 255, 255), (255, 50, 255)
        ]

        # Game parameters
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 6.0
        self.BALL_SPEED_INCREMENT = 0.5
        self.MAX_BALL_REFLECTION_X = 0.85

        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        # --- State Variables ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.stage = 0
        self.balls_left = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.ball_attached = False
        self.blocks = []
        self.particles = []
        self.frame_rewards = 0.0

        # Initialize state and validate
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        self.stage = 1
        self.balls_left = 3

        self.particles = []
        self.frame_rewards = 0.0

        self._setup_stage()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.frame_rewards = 0.0

        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_game_state()

        # --- Update Timers and Steps ---
        self.steps += 1
        self.time_left -= 1

        # --- Calculate Reward ---
        # Time penalty
        self.frame_rewards -= 0.01
        reward = self.frame_rewards
        # Add a separate score for display that isn't the same as reward
        # self.score += reward # This would make score a float, let's keep it int.

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _setup_stage(self):
        # Reset paddle and ball
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_speed = self.INITIAL_BALL_SPEED + (self.stage - 1) * self.BALL_SPEED_INCREMENT
        self._reset_ball()

        # Generate blocks
        self.blocks = []
        block_width, block_height = 50, 20
        rows, cols = 0, 0

        if self.stage == 1:
            rows, cols = 4, 10
        elif self.stage == 2:
            rows, cols = 5, 11
        elif self.stage == 3:
            rows, cols = 6, 12

        # Staggered layout
        y_offset = 60
        for r in range(rows):
            for c in range(cols):
                if self.stage == 2 and (r % 2 == 0): continue
                if self.stage == 3 and (c % 3 == 0): continue

                block_x = c * (block_width + 4) + 38
                block_y = r * (block_height + 4) + y_offset
                color = self.BLOCK_COLORS[(r + c) % len(self.BLOCK_COLORS)]
                self.blocks.append({"rect": pygame.Rect(block_x, block_y, block_width, block_height), "color": color})

    def _reset_ball(self):
        self.ball_attached = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _handle_input(self, action):
        movement = action[0]
        space_pressed = action[1] == 1

        # Paddle movement
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED

        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        # Launch ball
        if self.ball_attached and space_pressed:
            # sfx: launch_ball.wav
            self.ball_attached = False
            launch_angle = (self.np_random.random() * 0.6 + 0.2) * math.pi # 36 to 144 degrees
            self.ball_vel = [math.cos(launch_angle) * self.ball_speed, -math.sin(launch_angle) * self.ball_speed]

            # Anti-stuck: ensure we have some horizontal velocity
            if abs(self.ball_vel[0]) < 0.1:
                self.ball_vel[0] = 0.1 * np.sign(self.ball_vel[0] or 1)

    def _update_game_state(self):
        # Update attached ball
        if self.ball_attached:
            self.ball_pos[0] = self.paddle.centerx
            return

        # --- Ball Physics ---
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            # sfx: wall_bounce.wav
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.WIDTH, ball_rect.right)
            self.ball_pos[0] = ball_rect.centerx
        if ball_rect.top <= 0:
            # sfx: wall_bounce.wav
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            self.ball_pos[1] = ball_rect.centery

        # Ball lost
        if ball_rect.top >= self.HEIGHT:
            # sfx: lose_ball.wav
            self.balls_left -= 1
            self.frame_rewards -= 5.0
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
            return

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle_bounce.wav
            self.frame_rewards += 0.1 # Reward for keeping ball in play

            # Calculate reflection based on hit location
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            reflection_x = self.MAX_BALL_REFLECTION_X * offset

            # New velocity vector
            new_vel_x = reflection_x
            new_vel_y = -math.sqrt(max(0, 1 - new_vel_x**2)) # Pythagorean theorem to keep speed constant

            self.ball_vel = [new_vel_x * self.ball_speed, new_vel_y * self.ball_speed]

            # Anti-stuck mechanism for vertical movement
            if abs(self.ball_vel[1]) < 0.2:
                self.ball_vel[1] = -0.2 * self.ball_speed

        # Block collisions
        hit_block_idx = ball_rect.collidelist([b["rect"] for b in self.blocks])
        if hit_block_idx != -1:
            # sfx: block_break.wav
            block_hit = self.blocks.pop(hit_block_idx)
            self.frame_rewards += 1.0
            self.score += 100 # Add points for breaking a block

            # Create particles
            self._create_particles(block_hit["rect"].center, block_hit["color"])

            # Determine bounce direction
            self.ball_vel[1] *= -1

        # Check for stage clear
        if not self.blocks:
            # sfx: stage_clear.wav
            self.frame_rewards += 50.0
            self.score += 1000 # Stage clear bonus
            self.stage += 1
            if self.stage > 3:
                self.frame_rewards += 100.0 # Game win bonus
                self.score += 5000 # Game win bonus
                self.game_over = True
            else:
                self._setup_stage()

        # --- Update Particles ---
        self._update_particles()

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _check_termination(self):
        return self.balls_left <= 0 or self.time_left <= 0 or (self.stage > 3)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block["color"]), block["rect"], 2)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 30.0))))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p["pos"][0])-2, int(p["pos"][1])-2))

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Ball with glow
        center = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.BALL_RADIUS + 3, (100, 100, 0, 100))
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.BALL_RADIUS + 3, (100, 100, 0, 100))
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.BALL_RADIUS, self.COLOR_BALL)


    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls
        balls_text = self.font_small.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - balls_text.get_width() - 10, 10))

        # Timer
        time_seconds = self.time_left // self.FPS
        timer_text = self.font_large.render(f"{time_seconds}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH // 2 - timer_text.get_width() // 2, 10))

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, self.HEIGHT - 30))

        # Game Over / Win message
        if self.game_over:
            if self.stage > 3:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"

            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "balls_left": self.balls_left,
            "time_left": self.time_left // self.FPS,
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
    import os
    # Set this to "dummy" to run the validation check headless
    # Set it to your video driver ("x11", "windows", "macOS") for interactive play
    run_mode = "interactive" # or "headless"

    if run_mode == "headless":
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        env = GameEnv(render_mode="rgb_array")
        env.close()
    else:
        env = GameEnv(render_mode="rgb_array")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption(env.game_description)
        clock = pygame.time.Clock()

        obs, info = env.reset()
        done = False
        running = True
        while running:
            movement, space, shift = 0, 0, 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

            if keys[pygame.K_SPACE]:
                space = 1

            if keys[pygame.K_r]: # Press R to reset
                obs, info = env.reset()
                done = False

            action = [movement, space, shift]

            if not done:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # Render to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if done:
                # Keep rendering the final screen until a new game starts
                pass

            clock.tick(env.FPS)

        env.close()
        pygame.quit()