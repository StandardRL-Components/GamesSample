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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A rhythm-based block breaker. Bounce the ball to destroy blocks. "
        "Hit blocks on the beat for a combo multiplier and higher score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 8
        self.BALL_BASE_SPEED = 5
        self.BALL_MAX_X_VEL_MOD = 1.2
        self.MAX_STEPS = 5000  # ~2.7 minutes at 30fps
        self.INITIAL_BALLS = 3
        self.BPM = 120
        self.FPS = 30
        self.FRAMES_PER_BEAT = self.FPS / (self.BPM / 60)
        self.BEAT_WINDOW = 3  # frames before/after beat

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PADDLE = (240, 240, 240)
        self.COLOR_BALL = (100, 255, 100)
        self.COLOR_BALL_GLOW = (100, 255, 100, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = {
            1: (0, 200, 200),
            2: (200, 0, 200),
            3: (200, 200, 0)
        }

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        # These need to be initialized here so reset() can access them
        self.ball_trail = []
        self.particles = []
        self.blocks = []
        self.block_data = []
        self.paddle_rect = pygame.Rect(0, 0, 0, 0)
        self.ball_rect = pygame.Rect(0, 0, 0, 0)
        self.ball_pos = [0, 0]
        self.ball_vel = [0, 0]

        self.reset()

        # Run validation check
        # self.validate_implementation() # Commented out for submission, but useful for testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = self.INITIAL_BALLS
        self.combo = 1

        self.paddle_rect = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        self.particles = []
        self.ball_trail = []
        self.screen_shake = 0
        self.beat_timer = 0
        
        self._reset_ball()
        self._generate_blocks()

        self.reward_this_step = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1

        self._handle_input(movement, space_pressed)
        self._update_game_state()

        self.steps += 1
        reward = self.reward_this_step
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if not self.blocks:  # Win condition
                self.reward_this_step += 100
                reward += 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        if movement == 3:  # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_rect.x += self.PADDLE_SPEED

        self.paddle_rect.clamp_ip(self.screen.get_rect())

        if self.ball_on_paddle and space_pressed:
            # sfx: launch_ball.wav
            self.ball_on_paddle = False
            self.ball_vel = [self.np_random.uniform(-1, 1), -self.BALL_BASE_SPEED]
            self._normalize_ball_velocity()

    def _update_game_state(self):
        self.beat_timer = (self.beat_timer + 1) % self.FRAMES_PER_BEAT

        if not self.ball_on_paddle:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]
            self.ball_rect.center = self.ball_pos
            self.ball_trail.append(list(self.ball_pos))
            if len(self.ball_trail) > 10:
                self.ball_trail.pop(0)
            self._handle_collisions()
        else:
            self.ball_pos[0] = self.paddle_rect.centerx
            self.ball_pos[1] = self.paddle_rect.top - self.BALL_RADIUS
            self.ball_rect.center = self.ball_pos

        self._update_particles()
        if self.screen_shake > 0:
            self.screen_shake -= 1

    def _handle_collisions(self):
        # Wall collisions
        if self.ball_rect.left <= 0 or self.ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball_rect.left = max(0, self.ball_rect.left)
            self.ball_rect.right = min(self.WIDTH, self.ball_rect.right)
            # sfx: wall_bounce.wav
        if self.ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            self.ball_rect.top = max(0, self.ball_rect.top)
            # sfx: wall_bounce.wav

        # Paddle collision
        if self.ball_rect.colliderect(self.paddle_rect) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball_rect.bottom = self.paddle_rect.top

            offset = (self.ball_rect.centerx - self.paddle_rect.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * self.BALL_MAX_X_VEL_MOD

            self._normalize_ball_velocity()
            self.combo = 1  # Reset combo on paddle hit
            # sfx: paddle_bounce.wav

        # Block collisions
        hit_block_idx = self.ball_rect.collidelist(self.blocks)
        if hit_block_idx != -1:
            block = self.blocks[hit_block_idx]

            # Determine collision side to correctly reflect velocity
            prev_ball_rect = self.ball_rect.copy()
            prev_ball_rect.x -= self.ball_vel[0]
            prev_ball_rect.y -= self.ball_vel[1]

            if prev_ball_rect.right <= block.left or prev_ball_rect.left >= block.right:
                self.ball_vel[0] *= -1
            else:
                self.ball_vel[1] *= -1

            # On-beat check
            if self._is_on_beat():
                self.combo += 1
                self.reward_this_step += 0.5 * self.combo
                # sfx: beat_hit.wav
            else:
                self.combo = 1
                # sfx: block_hit.wav

            self.reward_this_step += 0.1  # Reward for hitting

            block_data = self.block_data[hit_block_idx]
            block_data['hp'] -= 1

            if block_data['hp'] <= 0:
                self.reward_this_step += 1.0 * self.combo  # Reward for destroying
                self.score += 10 * self.combo
                self._spawn_particles(block.centerx, block.centery, self.BLOCK_COLORS[block_data['initial_hp']], 30)
                self.blocks.pop(hit_block_idx)
                self.block_data.pop(hit_block_idx)
                self.screen_shake = 8
                # sfx: block_destroy.wav
            else:
                self.score += 1 * self.combo
                self._spawn_particles(block.centerx, block.centery, (128, 128, 128), 5)

        # Bottom of screen (lose ball)
        if self.ball_rect.top >= self.HEIGHT:
            self._lose_ball()

    def _lose_ball(self):
        self.balls_left -= 1
        self.combo = 1
        self.reward_this_step -= 10  # Penalty for losing ball
        self.screen_shake = 15
        # sfx: lose_ball.wav
        if self.balls_left > 0:
            self._reset_ball()
        else:
            self.game_over = True

    def _reset_ball(self):
        self.ball_on_paddle = True
        self.ball_pos = [self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]
        self.ball_rect = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball_rect.center = self.ball_pos
        self.ball_trail.clear()

    def _generate_blocks(self):
        self.blocks = []
        self.block_data = []
        block_width = 40
        block_height = 20
        rows = 5
        cols = 14

        for r in range(rows):
            for c in range(cols):
                hp = self.np_random.integers(1, 4)
                block_rect = pygame.Rect(
                    c * (block_width + 2) + 30,
                    r * (block_height + 2) + 40,
                    block_width,
                    block_height
                )
                self.blocks.append(block_rect)
                self.block_data.append({'hp': hp, 'initial_hp': hp})

    def _normalize_ball_velocity(self):
        speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
        if speed == 0: return
        scale = self.BALL_BASE_SPEED / speed
        self.ball_vel[0] *= scale
        self.ball_vel[1] *= scale
        # Anti-stuck mechanism
        if abs(self.ball_vel[1]) < 0.2:
            self.ball_vel[1] = 0.2 * np.sign(self.ball_vel[1]) if self.ball_vel[1] != 0 else 0.2

    def _is_on_beat(self):
        return self.beat_timer < self.BEAT_WINDOW or self.beat_timer > (self.FRAMES_PER_BEAT - self.BEAT_WINDOW)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)

        offset_x, offset_y = 0, 0
        if self.screen_shake > 0:
            offset_x = self.np_random.uniform(-self.screen_shake, self.screen_shake)
            offset_y = self.np_random.uniform(-self.screen_shake, self.screen_shake)
        render_offset = (int(offset_x), int(offset_y))

        self._render_background(render_offset)
        self._render_game(render_offset)
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, offset):
        # Grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i + offset[0], offset[1]),
                             (i + offset[0], self.HEIGHT + offset[1]))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (offset[0], i + offset[1]),
                             (self.WIDTH + offset[0], i + offset[1]))

        # Beat indicator
        beat_progress = self.beat_timer / self.FRAMES_PER_BEAT
        pulse = abs(math.sin(beat_progress * math.pi))
        if self._is_on_beat():
            pulse_color = (255, 255, 255, int(pulse * 60))
            pulse_radius = int(20 + pulse * 40)
        else:
            pulse_color = (255, 255, 255, int(pulse * 20))
            pulse_radius = int(20 + pulse * 10)

        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.aacircle(s, self.WIDTH // 2, self.HEIGHT // 2, pulse_radius, pulse_color)
        self.screen.blit(s, (offset[0], offset[1]))

    def _render_game(self, offset):
        # Blocks
        for i, block in enumerate(self.blocks):
            hp = self.block_data[i]['hp']
            color = self.BLOCK_COLORS[self.block_data[i]['initial_hp']]

            # Brighten color based on current HP
            brightness = (hp / self.block_data[i]['initial_hp']) * 0.5 + 0.5
            final_color = tuple(int(c * brightness) for c in color)

            block_offset = block.move(offset)
            pygame.draw.rect(self.screen, final_color, block_offset, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_BG, block_offset.inflate(-4, -4), border_radius=3)

        # Particles
        for p in self.particles:
            p_pos = (int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1]))
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p_pos[0] - p['size'], p_pos[1] - p['size']))

        # Ball trail
        if self.ball_trail:
            for i, pos in enumerate(self.ball_trail):
                alpha = int((i / len(self.ball_trail)) * 50)
                color = self.COLOR_BALL + (alpha,)
                pos_offset = (pos[0] + offset[0], pos[1] + offset[1])
                pygame.gfxdraw.filled_circle(self.screen, int(pos_offset[0]), int(pos_offset[1]), self.BALL_RADIUS,
                                             color)

        # Ball
        ball_pos_offset = (int(self.ball_pos[0] + offset[0]), int(self.ball_pos[1] + offset[1]))
        pygame.gfxdraw.aacircle(self.screen, ball_pos_offset[0], ball_pos_offset[1], self.BALL_RADIUS,
                                self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_offset[0], ball_pos_offset[1], self.BALL_RADIUS,
                                     self.COLOR_BALL)

        # Paddle
        paddle_offset = self.paddle_rect.move(offset)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_offset, border_radius=4)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        balls_text = self.font_main.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - balls_text.get_width() - 10, 5))

        if self.combo > 1:
            combo_text = self.font_main.render(f"x{self.combo}", True, self.BLOCK_COLORS[2])
            self.screen.blit(combo_text, (self.WIDTH // 2 - combo_text.get_width() // 2, 5))

        if self.game_over:
            msg = "GAME OVER" if self.balls_left <= 0 else "LEVEL CLEAR!"
            end_text = self.font_main.render(msg, True, (255, 50, 50))
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
            "combo": self.combo,
        }

    def _check_termination(self):
        return self.balls_left <= 0 or not self.blocks or self.steps >= self.MAX_STEPS

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8

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
        assert trunc is False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game
    # This part requires a display. Set SDL_VIDEODRIVER to something else.
    # For example:
    # os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    # For headless testing, the main class works as is.
    
    # Example of running the environment
    env = GameEnv()
    obs, info = env.reset()
    done = False
    step_count = 0
    while not done and step_count < 1000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
    print("Headless run finished.")
    print(f"Final Info: {info}")

    # To visualize the game, you need to unset the dummy videodriver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    pygame.quit() # Quit the dummy driver instance
    pygame.init() # Re-init with a real driver

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Rhythm Block Breaker")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement = 0  # No-op
        space = 0

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

        action = [movement, space, 0]  # Shift is not used

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000)  # Pause before restarting

        clock.tick(env.FPS)

    pygame.quit()