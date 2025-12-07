import gymnasium as gym
import os
import pygame
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:21:14.160205
# Source Brief: brief_00370.md
# Brief Index: 370
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control the speed of two drones along their separate paths to make them "
        "arrive at the central target at the same time."
    )
    user_guide = (
        "Controls: Use ↑ to increase and ↓ to decrease the blue drone's speed. "
        "Use ← to increase and → to decrease the orange drone's speed. "
        "Press space to attempt synchronization."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_DRONE_1 = (0, 192, 255)
    COLOR_DRONE_1_GLOW = (0, 192, 255, 50)
    COLOR_DRONE_2 = (255, 128, 0)
    COLOR_DRONE_2_GLOW = (255, 128, 0, 50)
    COLOR_TARGET = (255, 255, 255)
    COLOR_PATH = (100, 110, 130)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SUCCESS = (100, 255, 100)
    COLOR_FAILURE = (255, 100, 100)

    # Game Mechanics
    MIN_SPEED = 0.2
    MAX_SPEED = 2.0
    SPEED_ADJUST = 0.05
    SYNC_TOLERANCE_SECONDS = 0.1
    SUCCESSES_PER_LEVEL = 3

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
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_feedback = pygame.font.SysFont("Consolas", 32, bold=True)

        self.level = 1
        self.successful_syncs_at_level = 0

        # This will be initialized in reset()
        self.drones = []
        self.target_pos = (0, 0)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_predicted_time_diff = 0.0
        self.sync_feedback = None  # ('success'/'failure', time_diff, timer)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sync_feedback = None

        self._generate_paths()

        self.drones = [
            self._create_drone_state(self.path1, 0.5),
            self._create_drone_state(self.path2, 0.6),
        ]

        self.last_predicted_time_diff = self._calculate_predicted_time_diff()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        reward = 0
        terminated = False

        # --- Action Handling ---
        # Drone 1 Speed
        if movement == 1:  # Up
            self.drones[0]["speed"] = min(
                self.MAX_SPEED, self.drones[0]["speed"] + self.SPEED_ADJUST
            )
        elif movement == 2:  # Down
            self.drones[0]["speed"] = max(
                self.MIN_SPEED, self.drones[0]["speed"] - self.SPEED_ADJUST
            )
        # Drone 2 Speed
        if movement == 3:  # Left
            self.drones[1]["speed"] = min(
                self.MAX_SPEED, self.drones[1]["speed"] + self.SPEED_ADJUST
            )
        elif movement == 4:  # Right
            self.drones[1]["speed"] = max(
                self.MIN_SPEED, self.drones[1]["speed"] - self.SPEED_ADJUST
            )

        # --- Game Logic Update ---
        self._update_drone_state(0)
        self._update_drone_state(1)
        self.steps += 1

        # --- Reward Calculation ---
        current_predicted_time_diff = self._calculate_predicted_time_diff()
        if current_predicted_time_diff < self.last_predicted_time_diff:
            reward += 0.01  # Small reward for getting closer to sync
        self.last_predicted_time_diff = current_predicted_time_diff

        # --- Termination Check ---
        if space_pressed:
            # Player commits to a synchronization attempt
            arrival_diff = self._calculate_predicted_time_diff()
            terminated = True
            self.game_over = True

            if arrival_diff <= self.SYNC_TOLERANCE_SECONDS:
                # Success
                reward += 100
                self.score += 100
                self.successful_syncs_at_level += 1
                self.sync_feedback = ("success", arrival_diff, self.FPS * 2)  # Show for 2 seconds
                if self.successful_syncs_at_level >= self.SUCCESSES_PER_LEVEL:
                    self.level += 1
                    self.successful_syncs_at_level = 0
            else:
                # Failure
                # No negative reward, just the end of the episode
                self.sync_feedback = ("failure", arrival_diff, self.FPS * 2)

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.sync_feedback = ("timeout", 0, self.FPS * 2)

        truncated = False
        return self._get_observation(), reward, terminated, truncated, self._get_info()

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
            "level": self.level,
            "time_difference": self.last_predicted_time_diff,
        }

    def close(self):
        pygame.quit()

    # --- Game Logic Helpers ---
    def _create_drone_state(self, path, initial_speed):
        return {
            "path": path,
            "path_lengths": [
                math.hypot(
                    path[i][0] - path[(i + 1) % len(path)][0],
                    path[i][1] - path[(i + 1) % len(path)][1],
                )
                for i in range(len(path))
            ],
            "segment_idx": 0,
            "segment_progress": 0.0,
            "speed": initial_speed,
        }

    def _generate_paths(self):
        center_x, center_y = self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2
        self.target_pos = (center_x, center_y)

        num_points = 2 + self.level
        base_radius = 100 + (self.level - 1) * 30

        # Path 1
        self.path1 = [self.target_pos]
        radius1 = base_radius * self.np_random.uniform(0.9, 1.1)
        angle_step1 = 2 * math.pi / num_points
        for i in range(1, num_points):
            angle = angle_step1 * i + self.np_random.uniform(-0.1, 0.1)
            x = center_x + radius1 * math.cos(angle)
            y = center_y + radius1 * math.sin(angle)
            self.path1.append((x, y))

        # Path 2
        self.path2 = [self.target_pos]
        radius2 = base_radius * self.np_random.uniform(0.9, 1.1)
        angle_step2 = 2 * math.pi / num_points
        for i in range(1, num_points):
            angle = (
                angle_step2 * i + math.pi / num_points + self.np_random.uniform(-0.1, 0.1)
            )  # Offset angles
            x = center_x + radius2 * math.cos(angle)
            y = center_y + radius2 * math.sin(angle)
            self.path2.append((x, y))

    def _update_drone_state(self, drone_idx):
        drone = self.drones[drone_idx]
        travel_dist = drone["speed"] * (self.SCREEN_WIDTH / 640.0)  # Scale speed

        while travel_dist > 0:
            segment_len = drone["path_lengths"][drone["segment_idx"]]
            if segment_len == 0:  # Avoid division by zero on zero-length segments
                drone["segment_idx"] = (drone["segment_idx"] + 1) % len(drone["path"])
                continue

            dist_to_end_of_segment = (1.0 - drone["segment_progress"]) * segment_len

            if travel_dist >= dist_to_end_of_segment:
                travel_dist -= dist_to_end_of_segment
                drone["segment_progress"] = 0.0
                drone["segment_idx"] = (drone["segment_idx"] + 1) % len(drone["path"])
            else:
                drone["segment_progress"] += travel_dist / segment_len
                travel_dist = 0

    def _get_distance_to_target(self, drone_idx):
        drone = self.drones[drone_idx]
        total_dist = 0.0

        # Distance on current segment
        current_len = drone["path_lengths"][drone["segment_idx"]]
        total_dist += (1.0 - drone["segment_progress"]) * current_len

        # Sum of subsequent segments until target (index 0)
        num_segments = len(drone["path"])
        current_idx = (drone["segment_idx"] + 1) % num_segments
        while current_idx != 0:
            total_dist += drone["path_lengths"][current_idx]
            current_idx = (current_idx + 1) % num_segments

        return total_dist

    def _calculate_predicted_time_diff(self):
        dist1 = self._get_distance_to_target(0)
        dist2 = self._get_distance_to_target(1)

        time1 = dist1 / self.drones[0]["speed"] if self.drones[0]["speed"] > 0 else float("inf")
        time2 = dist2 / self.drones[1]["speed"] if self.drones[1]["speed"] > 0 else float("inf")

        return abs(time1 - time2) / self.FPS  # Convert from steps to seconds

    # --- Rendering Helpers ---
    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw paths and drones
        self._draw_path(self.drones[0], self.COLOR_PATH)
        self._draw_path(self.drones[1], self.COLOR_PATH)
        self._draw_drone(0, self.COLOR_DRONE_1, self.COLOR_DRONE_1_GLOW)
        self._draw_drone(1, self.COLOR_DRONE_2, self.COLOR_DRONE_2_GLOW)

        # Draw target
        pygame.gfxdraw.aacircle(
            self.screen, int(self.target_pos[0]), int(self.target_pos[1]), 12, self.COLOR_TARGET
        )
        pygame.gfxdraw.aacircle(
            self.screen, int(self.target_pos[0]), int(self.target_pos[1]), 13, self.COLOR_TARGET
        )

        # Draw sync feedback
        if self.sync_feedback:
            status, diff, timer = self.sync_feedback
            if timer > 0:
                alpha = int(255 * (timer / (self.FPS * 2)))
                radius = int(20 + 40 * (1 - timer / (self.FPS * 2)))
                color = self.COLOR_SUCCESS if status == "success" else self.COLOR_FAILURE
                if status == "timeout":
                    color = self.COLOR_TEXT

                s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(
                    s,
                    int(self.target_pos[0]),
                    int(self.target_pos[1]),
                    radius,
                    color + (alpha // 4,),
                )
                pygame.gfxdraw.aacircle(
                    s, int(self.target_pos[0]), int(self.target_pos[1]), radius, color + (alpha,)
                )
                self.screen.blit(s, (0, 0))

                self.sync_feedback = (status, diff, timer - 1)

    def _draw_path(self, drone, color):
        path = drone["path"]
        if len(path) > 1:
            pygame.draw.aalines(self.screen, color, True, path, 1)

    def _draw_drone(self, drone_idx, color, glow_color):
        drone = self.drones[drone_idx]
        p1 = drone["path"][drone["segment_idx"]]
        p2 = drone["path"][(drone["segment_idx"] + 1) % len(drone["path"])]

        # Linear interpolation for position
        x = p1[0] + (p2[0] - p1[0]) * drone["segment_progress"]
        y = p1[1] + (p2[1] - p1[1]) * drone["segment_progress"]
        pos = (int(x), int(y))

        # Draw glow
        glow_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surface, 20, 20, 18, glow_color)
        self.screen.blit(glow_surface, (pos[0] - 20, pos[1] - 20))

        # Draw drone
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, color)

    def _render_ui(self):
        # Top-left info
        self._draw_text(f"LEVEL: {self.level}", (10, 10), self.font_main, self.COLOR_TEXT)
        self._draw_text(f"SCORE: {self.score}", (10, 35), self.font_main, self.COLOR_TEXT)

        # Top-right info
        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        steps_size = self.font_main.size(steps_text)
        self._draw_text(
            steps_text, (self.SCREEN_WIDTH - steps_size[0] - 10, 10), self.font_main, self.COLOR_TEXT
        )

        # Bottom-center info
        time_diff_text = f"Predicted Time Diff: {self.last_predicted_time_diff:.3f}s"
        time_diff_size = self.font_main.size(time_diff_text)
        self._draw_text(
            time_diff_text,
            (self.SCREEN_WIDTH / 2 - time_diff_size[0] / 2, self.SCREEN_HEIGHT - 35),
            self.font_main,
            self.COLOR_TEXT,
        )

        # Drone-specific info
        self._draw_text(
            f"D1 Speed: {self.drones[0]['speed']:.2f}",
            (10, self.SCREEN_HEIGHT - 30),
            self.font_small,
            self.COLOR_DRONE_1,
        )
        d2_text = f"D2 Speed: {self.drones[1]['speed']:.2f}"
        d2_size = self.font_small.size(d2_text)
        self._draw_text(
            d2_text,
            (self.SCREEN_WIDTH - d2_size[0] - 10, self.SCREEN_HEIGHT - 30),
            self.font_small,
            self.COLOR_DRONE_2,
        )

        # Final feedback text
        if self.sync_feedback and self.sync_feedback[2] > 0:
            status, diff, _ = self.sync_feedback
            if status == "success":
                text = f"SYNCHRONIZED! ({diff:.3f}s)"
                color = self.COLOR_SUCCESS
            elif status == "failure":
                text = f"MISSED! ({diff:.3f}s)"
                color = self.COLOR_FAILURE
            else:
                text = "TIMEOUT"
                color = self.COLOR_TEXT

            text_surf = self.font_feedback.render(text, True, color)
            self.screen.blit(
                text_surf,
                (
                    self.SCREEN_WIDTH / 2 - text_surf.get_width() / 2,
                    self.SCREEN_HEIGHT / 2 - text_surf.get_height() / 2 - 50,
                ),
            )

    def _draw_text(self, text, pos, font, color):
        surface = font.render(text, True, color)
        self.screen.blit(surface, pos)

    def validate_implementation(self):
        # This method is for internal validation and can be removed for production
        print("Validating implementation...")
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


if __name__ == "__main__":
    # The main loop is for manual play and debugging
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    env.reset()
    env.validate_implementation() # Run validation

    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False

    # Pygame setup for rendering
    pygame.display.set_caption("Drone Synchronization")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    movement_action = 0  # 0=none, 1=up, 2=down, 3=left, 4=right
    space_action = 0

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset Environment")
    print("Q: Quit")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                # Player input mapping
                if event.key == pygame.K_UP:
                    movement_action = 1
                elif event.key == pygame.K_DOWN:
                    movement_action = 2
                elif event.key == pygame.K_LEFT:
                    movement_action = 3
                elif event.key == pygame.K_RIGHT:
                    movement_action = 4
                if event.key == pygame.K_SPACE:
                    space_action = 1
            if event.type == pygame.KEYUP:
                if event.key in [
                    pygame.K_UP,
                    pygame.K_DOWN,
                    pygame.K_LEFT,
                    pygame.K_RIGHT,
                ]:
                    movement_action = 0
                if event.key == pygame.K_SPACE:
                    space_action = 0

        if not done:
            action = [movement_action, space_action, 0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # After a step, we must reset the space action if it was a one-shot press
            if space_action == 1:
                space_action = 0

            if done:
                print(f"Episode Finished. Score: {info['score']}, Steps: {info['steps']}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    env.close()