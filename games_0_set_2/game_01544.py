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
    user_guide = "Controls: ←→ to steer. ↑ or Space to accelerate. ↓ or Shift to brake."

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced, side-view arcade racer. Navigate the twisting track, hit checkpoints, and complete 3 laps before time runs out."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_LAPS = 3
    MAX_STEPS = FPS * 180  # 180 seconds total time limit

    # --- Colors ---
    COLOR_BG = (10, 10, 20)
    COLOR_TRACK = (200, 200, 220)
    COLOR_CHECKPOINT = (0, 255, 100)
    COLOR_FINISH_LINE_A = (255, 255, 255)
    COLOR_FINISH_LINE_B = (0, 0, 0)
    COLOR_CAR = (255, 50, 50)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_SPARK = (255, 200, 100)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)

        # Track Generation
        self.track_length = 30000  # Total pixel length of one lap
        self.num_checkpoints = 10
        self.track_points = []
        self._generate_track()

        # State variables initialized in reset()
        self.car_pos_x = 0
        self.car_speed = 0
        self.car_steer_angle = 0
        self.track_scroll_y = 0
        self.current_lap = 0
        self.next_checkpoint_index = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lap_start_step = 0
        self.lap_times = []
        self.sparks = []
        self.speed_lines = []
        self.CAR_Y_ON_SCREEN = self.SCREEN_HEIGHT * 0.8

        # Initialize state
        # self.reset() # This is called by the test harness, not needed here

    def _generate_track(self):
        self.track_points = []
        base_width = 150
        num_points = self.track_length // 10
        for i in range(num_points):
            y = i * 10
            # Use a combination of sine waves for a more interesting track
            x_offset1 = 150 * math.sin(i / (num_points / 12) * 2 * math.pi)
            x_offset2 = 100 * math.sin(i / (num_points / 30) * 2 * math.pi)
            center_x = self.SCREEN_WIDTH / 2 + x_offset1 + x_offset2
            self.track_points.append((-y, center_x, base_width))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.car_pos_x = self.SCREEN_WIDTH / 2
        self.car_speed = 0
        self.car_steer_angle = 0
        # FIX: The car's world Y coordinate is (CAR_Y_ON_SCREEN - track_scroll_y).
        # The track starts at world Y=0. To place the car at the start,
        # track_scroll_y must be initialized to CAR_Y_ON_SCREEN.
        self.track_scroll_y = self.CAR_Y_ON_SCREEN
        self.current_lap = 1
        self.next_checkpoint_index = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lap_start_step = 0
        self.lap_times = []
        self.sparks = []
        self.speed_lines = [
            (random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT))
            for _ in range(50)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = False

        if not self.game_over:
            self._update_player(action)
            self._update_game_state()
            reward, terminated = self._calculate_reward_and_termination()
            self.score += reward
            self._update_effects()

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not self.game_over:
            terminated = True

        if terminated and not self.game_over:
            self.game_over = True

        # Truncation is not used in this game's logic, but the API requires it.
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Time limit is a terminal condition in this game

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        is_accelerating = movement == 1 or space_held
        is_braking = movement == 2 or shift_held

        if is_accelerating:
            self.car_speed += 0.5
        elif is_braking:
            self.car_speed -= 1.0
        else:
            self.car_speed *= 0.98  # Natural friction

        self.car_speed = max(0, min(self.car_speed, 20))

        steer_input = 0
        if movement == 3: steer_input = -1
        elif movement == 4: steer_input = 1

        turn_power = 3.5 * (1 - self.car_speed / 25)  # Slower turns at high speed
        turn_power = max(1.0, turn_power)

        self.car_pos_x += steer_input * turn_power
        self.car_pos_x = np.clip(self.car_pos_x, 0, self.SCREEN_WIDTH)

        target_steer_angle = steer_input * -15 * (self.car_speed / 20)
        self.car_steer_angle = self.car_steer_angle * 0.8 + target_steer_angle * 0.2

    def _update_game_state(self):
        self.track_scroll_y += self.car_speed

        # Check for lap completion
        if self.track_scroll_y >= self.track_length + self.CAR_Y_ON_SCREEN:
            self.track_scroll_y -= self.track_length
            if self.current_lap <= self.MAX_LAPS:
                lap_duration = (self.steps - self.lap_start_step) / self.FPS
                self.lap_times.append(lap_duration)
                self.lap_start_step = self.steps
                self.current_lap += 1
                self.next_checkpoint_index = 1

    def _calculate_reward_and_termination(self):
        car_world_y = self.CAR_Y_ON_SCREEN - self.track_scroll_y
        track_center_x, track_width = self._get_track_properties_at(car_world_y)

        if self.current_lap == 2: track_width *= 0.9
        elif self.current_lap >= 3: track_width *= 0.75

        track_left_edge = track_center_x - track_width / 2
        track_right_edge = track_center_x + track_width / 2

        reward = 0
        terminated = False

        is_off_track = self.car_pos_x < track_left_edge or self.car_pos_x > track_right_edge

        if is_off_track:
            reward = -100
            terminated = True
            for _ in range(30):
                self.sparks.append([
                    [self.car_pos_x, self.CAR_Y_ON_SCREEN],
                    [random.uniform(-4, 4), random.uniform(-4, 4)],
                    random.randint(20, 40)])
            return reward, terminated

        reward += 0.1  # On-track survival bonus

        dist_from_center = abs(self.car_pos_x - track_center_x)
        normalized_dist = dist_from_center / (track_width / 2)
        reward -= 0.1 * normalized_dist ** 2

        checkpoint_y_interval = self.track_length / self.num_checkpoints
        next_checkpoint_world_y = -(self.next_checkpoint_index * checkpoint_y_interval)

        car_world_y_start_of_frame = self.CAR_Y_ON_SCREEN - (self.track_scroll_y - self.car_speed)

        if car_world_y_start_of_frame > next_checkpoint_world_y and car_world_y <= next_checkpoint_world_y:
            if self.next_checkpoint_index < self.num_checkpoints:
                reward += 1.0
                self.next_checkpoint_index += 1

        # Check for crossing the finish line
        finish_line_world_y = 0
        if (car_world_y_start_of_frame > finish_line_world_y and car_world_y <= finish_line_world_y and self.current_lap > 1):
             reward += 10.0

        if self.current_lap > self.MAX_LAPS:
            time_bonus = 50 * max(0, self.MAX_STEPS - self.steps) / self.MAX_STEPS
            reward += 50 + time_bonus
            terminated = True

        return reward, terminated

    def _get_track_properties_at(self, world_y):
        lookup_y = -world_y % self.track_length
        idx = int(lookup_y / 10) % len(self.track_points)

        p1 = self.track_points[idx]
        p2 = self.track_points[(idx + 1) % len(self.track_points)]

        interp_factor = (lookup_y % 10) / 10.0

        center_x = p1[1] + (p2[1] - p1[1]) * interp_factor
        width = p1[2] + (p2[2] - p1[2]) * interp_factor

        return center_x, width

    def _update_effects(self):
        for spark in self.sparks:
            spark[0][0] += spark[1][0]
            spark[0][1] += spark[1][1]
            spark[1][1] += 0.2
            spark[2] -= 1
        self.sparks = [s for s in self.sparks if s[2] > 0]

        for i in range(len(self.speed_lines)):
            x, y = self.speed_lines[i]
            y += self.car_speed / 2
            if y > self.SCREEN_HEIGHT:
                y = 0
                x = random.uniform(0, self.SCREEN_WIDTH)
            self.speed_lines[i] = (x, y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_effects()
        self._render_track()
        self._render_player()

    def _render_track(self):
        # Render track segments
        for i in range(len(self.track_points)):
            p1 = self.track_points[i]
            p2 = self.track_points[(i + 1) % len(self.track_points)]

            y1_on_screen = p1[0] + self.track_scroll_y
            y2_on_screen = p2[0] + self.track_scroll_y
            if i == len(self.track_points) - 1: # Handle wrap-around for drawing
                y2_on_screen = p2[0] + self.track_scroll_y + self.track_length


            if not (min(y1_on_screen, y2_on_screen) > self.SCREEN_HEIGHT or max(y1_on_screen, y2_on_screen) < 0):
                track_width_1, track_width_2 = p1[2], p2[2]
                if self.current_lap == 2:
                    track_width_1, track_width_2 = track_width_1 * 0.9, track_width_2 * 0.9
                elif self.current_lap >= 3:
                    track_width_1, track_width_2 = track_width_1 * 0.75, track_width_2 * 0.75

                pygame.draw.aaline(self.screen, self.COLOR_TRACK, (p1[1] - track_width_1 / 2, y1_on_screen), (p2[1] - track_width_2 / 2, y2_on_screen))
                pygame.draw.aaline(self.screen, self.COLOR_TRACK, (p1[1] + track_width_1 / 2, y1_on_screen), (p2[1] + track_width_2 / 2, y2_on_screen))

        # Render checkpoints and finish line
        for i in range(self.num_checkpoints):
            checkpoint_world_y = -(i * (self.track_length / self.num_checkpoints))
            checkpoint_y_on_screen = checkpoint_world_y + self.track_scroll_y

            if 0 < checkpoint_y_on_screen < self.SCREEN_HEIGHT:
                center_x, width = self._get_track_properties_at(checkpoint_world_y)
                if self.current_lap == 2: width *= 0.9
                elif self.current_lap >= 3: width *= 0.75
                left = center_x - width / 2

                if i == 0: # Finish line
                    for j in range(10):
                        color = self.COLOR_FINISH_LINE_A if j % 2 == 0 else self.COLOR_FINISH_LINE_B
                        pygame.draw.rect(self.screen, color, (left + j * width/10, checkpoint_y_on_screen - 5, math.ceil(width/10), 10))
                else: # Checkpoint
                    pygame.draw.rect(self.screen, self.COLOR_CHECKPOINT, (left, checkpoint_y_on_screen - 2, width, 4))

    def _render_effects(self):
        if self.car_speed > 5:
            for x, y in self.speed_lines:
                length = self.car_speed * 1.5
                alpha = int(150 * (self.car_speed / 20))
                line_surf = pygame.Surface((2, max(1, length)), pygame.SRCALPHA)
                line_surf.fill((255, 255, 255, alpha))
                self.screen.blit(line_surf, (x, y))

        for spark in self.sparks:
            pos = (int(spark[0][0]), int(spark[0][1]))
            size = max(1, int(spark[2] / 8))
            pygame.draw.circle(self.screen, self.COLOR_SPARK, pos, size)

    def _render_player(self):
        car_width, car_height = 20, 35
        car_surf = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, self.COLOR_CAR, (0, 0, car_width, car_height), border_radius=4)
        pygame.draw.rect(car_surf, (255, 255, 200), (2, 2, 5, 7), border_radius=2)
        pygame.draw.rect(car_surf, (255, 255, 200), (car_width - 7, 2, 5, 7), border_radius=2)

        rotated_car = pygame.transform.rotate(car_surf, self.car_steer_angle)
        car_rect = rotated_car.get_rect(center=(self.car_pos_x, self.CAR_Y_ON_SCREEN))

        self.screen.blit(rotated_car, car_rect.topleft)

    def _render_ui(self):
        lap_text = f"LAP: {min(self.current_lap, self.MAX_LAPS)}/{self.MAX_LAPS}"
        lap_surf = self.font_ui.render(lap_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_surf, (10, 10))

        time_left = max(0, self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"TIME: {time_left:.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 35))

        speed_text = f"{int(self.car_speed * 10):03d} KPH"
        speed_surf = self.font_ui.render(speed_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_surf, (self.SCREEN_WIDTH - speed_surf.get_width() - 10, self.SCREEN_HEIGHT - speed_surf.get_height() - 10))

        if self.game_over:
            msg = "RACE COMPLETE!" if self.current_lap > self.MAX_LAPS else "GAME OVER"
            msg_surf = self.font_big.render(msg, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(msg_surf, msg_rect)

            if self.current_lap > self.MAX_LAPS:
                total_time = sum(self.lap_times)
                time_msg = f"Total Time: {total_time:.2f}s"
                time_surf = self.font_ui.render(time_msg, True, self.COLOR_UI_TEXT)
                time_rect = time_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30))
                self.screen.blit(time_surf, time_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.current_lap,
            "speed": self.car_speed,
            "lap_times": self.lap_times,
        }

    def close(self):
        pygame.quit()