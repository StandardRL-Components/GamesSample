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
        "Controls: Arrow keys to move the cursor. Press Space to draw a line from the last cursor point. Hold Shift to reset the cursor to the start."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a track for the physics-based rider to reach the finish line. Be quick and efficient with your lines to maximize your score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.GRAVITY = np.array([0.0, 0.2])
        self.CURSOR_SPEED = 10.0
        self.START_X, self.FINISH_X = 50, 590

        # --- Colors ---
        self.COLOR_BG = (15, 19, 23)
        self.COLOR_GRID = (30, 35, 40)
        self.COLOR_TRACK = (200, 200, 200)
        self.COLOR_RIDER = (255, 255, 255)
        self.COLOR_RIDER_GLOW = (255, 255, 255, 50)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_START = (0, 255, 100)
        self.COLOR_FINISH = (255, 50, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_SHADOW = (10, 10, 10)

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.rider_pos = None
        self.rider_vel = None
        self.rider_radius = 10
        self.lines = []
        self.cursor_pos = None
        self.last_cursor_pos = None
        self.prev_space_held = False
        self.particles = []

        # Initialize state variables
        self.reset()

        # Run self-check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # --- Rider ---
        self.rider_pos = np.array([float(self.START_X), 150.0])
        self.rider_vel = np.array([2.5, 0.0])

        # --- Track ---
        # Initial flat platform to prevent immediate failure
        self.lines = [[np.array([20.0, 200.0]), np.array([100.0, 200.0])]]

        # --- Player Input State ---
        self.cursor_pos = np.array([120.0, 200.0])
        self.last_cursor_pos = self.cursor_pos.copy()
        self.prev_space_held = False

        # --- Effects ---
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        shift_held = action[2] == 1

        reward = 0

        # --- Handle Actions ---
        self.last_cursor_pos = self.cursor_pos.copy()

        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        if shift_held:
            self.cursor_pos = np.array([120.0, 200.0])

        if space_held and not self.prev_space_held:
            # Only draw a line on the frame the spacebar is pressed
            if np.linalg.norm(self.cursor_pos - self.last_cursor_pos) > 1:
                self.lines.append([self.last_cursor_pos.copy(), self.cursor_pos.copy()])
                reward -= 1 # Penalty for drawing a line

        self.prev_space_held = space_held

        # --- Update Game Logic ---
        self._update_physics()
        self._update_particles()

        self.steps += 1
        reward += 0.1  # Survival reward

        # --- Check Termination ---
        terminated = False
        if self.rider_pos[0] >= self.FINISH_X:
            reward += 50
            terminated = True
            self.win = True
        elif not (0 < self.rider_pos[0] < self.WIDTH and -self.rider_radius < self.rider_pos[1] < self.HEIGHT):
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        if terminated:
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_physics(self):
        # Apply gravity
        self.rider_vel += self.GRAVITY

        # Clamp velocity
        speed = np.linalg.norm(self.rider_vel)
        if speed > 15:
            self.rider_vel = self.rider_vel / speed * 15

        # Move rider
        self.rider_pos += self.rider_vel

        # Collision detection and response
        for line_start, line_end in self.lines:
            line_vec = line_end - line_start
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue

            p_vec = self.rider_pos - line_start
            t = np.dot(p_vec, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)

            closest_point = line_start + t * line_vec
            dist_vec = self.rider_pos - closest_point
            dist = np.linalg.norm(dist_vec)

            if dist < self.rider_radius:
                # Collision
                penetration = self.rider_radius - dist
                normal = dist_vec / dist if dist > 0 else np.array([0.0, -1.0])

                # Positional correction
                self.rider_pos += normal * penetration

                # Velocity response (slide and bounce)
                restitution = 0.4 # Bounciness
                friction = 0.1 # Sliding friction

                v_normal_component = np.dot(self.rider_vel, normal)
                if v_normal_component < 0:
                    # Reflect velocity
                    self.rider_vel -= (1 + restitution) * v_normal_component * normal

                    # Apply friction
                    tangent = np.array([-normal[1], normal[0]])
                    v_tangent_component = np.dot(self.rider_vel, tangent)
                    self.rider_vel -= v_tangent_component * friction * tangent

                # Create particles
                for _ in range(5):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 3)
                    p_vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                    p_life = self.np_random.integers(10, 20)
                    self.particles.append([closest_point.copy(), p_vel, p_life])

    def _update_particles(self):
        self.particles = [
            [p[0] + p[1], p[1] * 0.9, p[2] - 1]
            for p in self.particles if p[2] > 0
        ]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, pos, font, color, shadow_color, align="topleft"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        # Use keyword argument expansion to set the rect's position anchor
        text_rect = text_surf.get_rect(**{align: pos})
        
        # The shadow needs to be offset from the final text position
        shadow_pos = (text_rect.x + 2, text_rect.y + 2)
        
        self.screen.blit(shadow_surf, shadow_pos)
        self.screen.blit(text_surf, text_rect)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw start/finish lines
        pygame.draw.line(self.screen, self.COLOR_START, (self.START_X, 0), (self.START_X, self.HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.FINISH_X, 0), (self.FINISH_X, self.HEIGHT), 3)

        # Draw track
        for start, end in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, start, end, 2)

        # Draw particles
        for pos, vel, life in self.particles:
            alpha = max(0, min(255, int(life * 15)))
            radius = max(0, int(life / 4))
            color = (*self.COLOR_RIDER, alpha)
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (int(pos[0] - radius), int(pos[1] - radius)))

        # Draw rider
        rider_x, rider_y = int(self.rider_pos[0]), int(self.rider_pos[1])
        # Glow effect
        glow_surf = pygame.Surface((self.rider_radius * 4, self.rider_radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_RIDER_GLOW, (self.rider_radius * 2, self.rider_radius * 2), self.rider_radius * 2)
        self.screen.blit(glow_surf, (rider_x - self.rider_radius * 2, rider_y - self.rider_radius * 2))
        # Main body
        pygame.gfxdraw.aacircle(self.screen, rider_x, rider_y, self.rider_radius, self.COLOR_RIDER)
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, self.rider_radius, self.COLOR_RIDER)

        # Draw cursor
        cursor_x, cursor_y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_x - 8, cursor_y), (cursor_x + 8, cursor_y), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y - 8), (cursor_x, cursor_y + 8), 2)

    def _render_ui(self):
        # Score and Steps
        self._render_text(f"SCORE: {self.score:.1f}", (self.WIDTH - 10, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_SHADOW, align="topright")
        
        time_left = (self.MAX_STEPS - self.steps) / 30
        self._render_text(f"TIME: {time_left:.1f}", (10, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_SHADOW, align="topleft")
        
        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "FINISH!"
                color = self.COLOR_START
            else:
                msg = "CRASHED!"
                color = self.COLOR_FINISH
            self._render_text(msg, (self.WIDTH // 2, self.HEIGHT // 2), self.font_msg, color, self.COLOR_SHADOW, align="center")

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
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
        assert trunc == False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # --- Pygame setup for human play ---
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Line Rider Gym")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}. Score: {info['score']:.2f}")
            # Wait for a moment before auto-resetting or wait for 'r' key
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Human play runs at 30 FPS

    env.close()