import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Use arrow keys to draw short lines around the rider. "
        "Hold space to draw a longer line in the rider's direction of travel. "
        "Hold shift to advance time without drawing."
    )

    game_description = (
        "A physics-based puzzle game. Draw lines to guide the sledder from the start to the finish, "
        "collecting checkpoints along the way. Be careful, you have a limited number of steps!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    STATIONARY_LIMIT = 15

    # Colors
    COLOR_BG = (220, 220, 230)
    COLOR_GRID = (200, 200, 210)
    COLOR_RIDER = (20, 20, 20)
    COLOR_PLAYER_LINE = (227, 63, 51)  # Bright Red
    COLOR_TERRAIN = (100, 100, 110)
    COLOR_CHECKPOINT = (66, 186, 150)  # Green
    COLOR_FINISH = (255, 191, 0)  # Gold
    COLOR_TRAJECTORY = (51, 153, 255, 100)  # Semi-transparent Blue
    COLOR_UI_TEXT = (10, 10, 20)

    # Physics
    GRAVITY = 0.15
    FRICTION = 0.998
    RIDER_RADIUS = 6
    MAX_SPEED = 15
    LINE_DRAW_DIST_SHORT = 10
    LINE_DRAW_DIST_LONG = 25
    COLLISION_THRESHOLD = 8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 36)

        self.rider_pos = pygame.math.Vector2(0, 0)
        self.rider_vel = pygame.math.Vector2(0, 0)
        self.lines = []
        self.terrain = []
        self.checkpoints = []
        self.finish_line_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_rider_x = 0
        self.stationary_counter = 0

        # Reset the environment to initialize game state before validation
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stationary_counter = 0

        self._generate_level()
        self.rider_pos = pygame.math.Vector2(50, self.terrain[0][0][1] - self.RIDER_RADIUS)
        self.rider_vel = pygame.math.Vector2(0, 0)
        self.last_rider_x = self.rider_pos.x

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.lines = []
        self.terrain = [
            ((20, 250), (120, 250)),
            ((180, 300), (280, 280)),
            ((320, 350), (450, 340)),
            ((500, 300), (self.SCREEN_WIDTH - 20, 300)),
        ]
        self.checkpoints = [
            {"pos": pygame.math.Vector2(230, 260), "reached": False},
            {"pos": pygame.math.Vector2(400, 320), "reached": False},
        ]
        self.finish_line_x = self.SCREEN_WIDTH - 30

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Time penalty

        self._handle_action(movement, space_held, shift_held)
        self._update_physics()

        # Rewards and state checks
        reward += self._check_progress()
        reward += self._check_checkpoints()

        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.game_over = terminated

        self.score += reward
        self.steps += 1

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, if truncated is true, terminated should also be true

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_action(self, movement, space_held, shift_held):
        if shift_held:
            # Advance simulation without drawing
            return

        start_pos = self.rider_pos

        if space_held:
            # Draw long line in direction of velocity
            if self.rider_vel.length() > 0.5:
                direction = self.rider_vel.normalize()
                end_pos = start_pos + direction * self.LINE_DRAW_DIST_LONG
                self.lines.append((tuple(start_pos), tuple(end_pos)))
                # sfx: pen_draw_long
        else:
            # Draw short line based on movement action
            direction = None
            if movement == 1: direction = pygame.math.Vector2(0, -1)  # Up
            elif movement == 2: direction = pygame.math.Vector2(0, 1)  # Down
            elif movement == 3: direction = pygame.math.Vector2(-1, 0)  # Left
            elif movement == 4: direction = pygame.math.Vector2(1, 0)  # Right

            if direction:
                end_pos = start_pos + direction * self.LINE_DRAW_DIST_SHORT
                self.lines.append((tuple(start_pos), tuple(end_pos)))
                # sfx: pen_draw_short

    def _update_physics(self):
        # 1. Find closest ground contact
        on_surface = False
        best_line = None
        min_dist = float('inf')

        all_lines = self.lines + self.terrain

        for line in all_lines:
            p1, p2 = pygame.math.Vector2(line[0]), pygame.math.Vector2(line[1])
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue

            # Project rider onto line
            rider_vec = self.rider_pos - p1
            t = rider_vec.dot(line_vec) / line_vec.length_squared()
            t = max(0, min(1, t))  # Clamp to segment

            closest_point = p1 + t * line_vec
            dist_vec = self.rider_pos - closest_point

            # Check if rider is above the line and close enough
            if dist_vec.length() < self.COLLISION_THRESHOLD:
                if dist_vec.length() < min_dist:
                    min_dist = dist_vec.length()
                    best_line = (p1, p2, closest_point, dist_vec)

        # 2. Apply forces based on contact
        if best_line:
            p1, p2, closest_point, dist_vec = best_line

            # Snap to surface
            self.rider_pos = closest_point + pygame.math.Vector2(0, -self.RIDER_RADIUS)

            # Get line properties
            line_vec_norm = (p2 - p1)
            if line_vec_norm.length_squared() > 0:
                line_vec_norm.normalize_ip()
            line_angle = math.atan2(line_vec_norm.y, line_vec_norm.x)

            # Project velocity onto line
            speed = self.rider_vel.dot(line_vec_norm)

            # Add gravity component
            speed += self.GRAVITY * math.sin(line_angle)

            # Apply friction
            speed *= self.FRICTION

            self.rider_vel = line_vec_norm * speed
            on_surface = True
            # sfx: sled_scrape
        else:
            # Freefall
            self.rider_vel.y += self.GRAVITY
            # sfx: whoosh

        # 3. Update position and cap speed
        self.rider_vel.x = max(-self.MAX_SPEED, min(self.MAX_SPEED, self.rider_vel.x))
        self.rider_vel.y = max(-self.MAX_SPEED, min(self.MAX_SPEED, self.rider_vel.y))
        self.rider_pos += self.rider_vel

    def _check_progress(self):
        progress = self.rider_pos.x - self.last_rider_x
        self.last_rider_x = self.rider_pos.x
        return progress * 0.1 if progress > 0 else progress * 0.05

    def _check_checkpoints(self):
        reward = 0
        for cp in self.checkpoints:
            if not cp["reached"] and self.rider_pos.distance_to(cp["pos"]) < 20:
                cp["reached"] = True
                reward += 1.0
                # sfx: checkpoint_ding
        return reward

    def _check_termination(self):
        # Check for crash (out of bounds)
        if not (0 < self.rider_pos.x < self.SCREEN_WIDTH and 0 < self.rider_pos.y < self.SCREEN_HEIGHT):
            # sfx: crash_fall
            return True, -10.0

        # Check for being stationary
        if self.rider_vel.length() < 0.1:
            self.stationary_counter += 1
        else:
            self.stationary_counter = 0

        if self.stationary_counter >= self.STATIONARY_LIMIT:
            # sfx: stuck_sound
            return True, -10.0

        # Check for max steps (truncation is handled in step)
        if self.steps >= self.MAX_STEPS - 1:
            return False, -5.0 # Return terminated=False, truncation will handle episode end

        # Check for victory
        if self.rider_pos.x >= self.finish_line_x:
            # sfx: victory_fanfare
            return True, 100.0

        return False, 0.0

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
            "rider_pos": (self.rider_pos.x, self.rider_pos.y),
            "rider_vel": (self.rider_vel.x, self.rider_vel.y),
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw terrain
        for line in self.terrain:
            pygame.draw.aaline(self.screen, self.COLOR_TERRAIN, line[0], line[1], 5)

        # Draw player lines
        for line in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_PLAYER_LINE, line[0], line[1], 2)

        # Draw checkpoints
        for cp in self.checkpoints:
            color = self.COLOR_CHECKPOINT if not cp["reached"] else self.COLOR_GRID
            pos = (int(cp["pos"].x), int(cp["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, color)

        # Draw finish line
        for y in range(0, self.SCREEN_HEIGHT, 20):
            color1 = self.COLOR_FINISH if (y // 20) % 2 == 0 else self.COLOR_BG
            color2 = self.COLOR_BG if (y // 20) % 2 == 0 else self.COLOR_FINISH
            pygame.draw.rect(self.screen, color1, (self.finish_line_x, y, 10, 20))
            pygame.draw.rect(self.screen, color2, (self.finish_line_x + 10, y, 10, 20))

        # Draw trajectory prediction
        if self.rider_vel.length() > 0.1:
            start_p = self.rider_pos
            end_p = start_p + self.rider_vel.normalize() * self.rider_vel.length() * 5
            pygame.draw.aaline(self.screen, self.COLOR_TRAJECTORY, tuple(start_p), tuple(end_p))

        # Draw rider
        pos_x, pos_y = int(self.rider_pos.x), int(self.rider_pos.y)
        pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, self.RIDER_RADIUS, self.COLOR_RIDER)

        # Sled shape
        if self.rider_vel.length() > 0.5:
            direction = self.rider_vel.normalize()
            p1 = self.rider_pos - direction * self.RIDER_RADIUS
            p2 = self.rider_pos + direction.rotate(-135) * self.RIDER_RADIUS
            p3 = self.rider_pos + direction.rotate(135) * self.RIDER_RADIUS
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_RIDER)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_RIDER)

    def _render_ui(self):
        speed = self.rider_vel.length() * 10
        speed_text = self.font_ui.render(f"Speed: {speed:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (10, 10))

        dist = self.rider_pos.x / self.finish_line_x * 100 if self.finish_line_x != 0 else 0
        dist_text = self.font_ui.render(f"Progress: {max(0, dist):.0f}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(dist_text, (self.SCREEN_WIDTH - dist_text.get_width() - 10, 10))

        steps_text = self.font_ui.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (10, self.SCREEN_HEIGHT - steps_text.get_height() - 10))

        score_text = self.font_ui.render(f"Score: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text,
                         (self.SCREEN_WIDTH - score_text.get_width() - 10, self.SCREEN_HEIGHT - score_text.get_height() - 10))

        if self.game_over:
            outcome_text = ""
            if self.rider_pos.x >= self.finish_line_x:
                outcome_text = "FINISH!"
            else:
                outcome_text = "CRASHED"

            text_surf = self.font_title.render(outcome_text, True, self.COLOR_PLAYER_LINE)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Beginning implementation validation...")
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    # For this to work, you must comment out the line: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # As that line prevents a window from being created.
    
    # To run in headed mode, comment out the following line:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # And then uncomment the pygame.display lines below.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup Pygame window for human play
    # This requires the "dummy" video driver to be disabled
    if os.environ.get("SDL_VIDEODRIVER") != "dummy":
        pygame.display.set_caption("Line Rider Gym")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    else:
        screen = None
        print("\nRunning in headless mode. No window will be displayed.")
        print("To play manually, comment out the 'os.environ' line at the top of the file.\n")


    terminated = False
    truncated = False
    running = True
    while running:
        action = [0, 0, 0] # Default is no-op
        
        if screen:
            # --- Action selection ---
            movement, space, shift = 0, 0, 0

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]

            # --- Event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Press 'R' to reset
                        obs, info = env.reset()
                        terminated = False
                        truncated = False
        else: # In headless mode, just take a few random steps for demonstration
            if env.steps > 10:
                running = False
            action = env.action_space.sample()

        # --- Step the environment ---
        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        if screen:
            # The observation is already a rendered frame
            # We just need to blit it to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30)  # Limit to 30 FPS for human play
        
        if (terminated or truncated) and screen:
            print(f"Episode finished. Terminated: {terminated}, Truncated: {truncated}. Score: {info['score']:.2f}")
            # In a real scenario, you might wait for a reset key press
            # For this example, we'll just let the loop continue until 'R' is pressed or the window is closed.
            pass


    env.close()