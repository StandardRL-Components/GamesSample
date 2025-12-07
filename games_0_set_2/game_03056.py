import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Use arrow keys to aim your track drawing. Hold Space to draw longer segments. Hold Shift to let the rider fall without drawing."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw a physics-based track to guide your sledder to the finish line before time runs out. Master the terrain to get the best score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WORLD_WIDTH = 100.0  # meters
        self.WORLD_HEIGHT = self.SCREEN_HEIGHT / self.SCREEN_WIDTH * self.WORLD_WIDTH

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 50)

        # Game constants
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.TIME_LIMIT_SECONDS = 30
        self.GRAVITY = 0.15
        self.FRICTION = 0.99
        self.RIDER_RADIUS = 3
        self.TRAIL_LENGTH = 50
        self.MAX_TRACK_SEGMENTS = 200

        # Colors
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_TERRAIN = (60, 65, 80)
        self.COLOR_RIDER = (255, 255, 255)
        self.COLOR_RIDER_GLOW = (200, 200, 255)
        self.COLOR_TRACK = (255, 50, 50)
        self.COLOR_FINISH = (50, 255, 50)
        self.COLOR_TRAIL = (100, 150, 255)
        self.COLOR_UI = (220, 220, 240)

        # Initialize state variables
        self.rider_pos = None
        self.rider_vel = None
        self.player_track = None
        self.rider_trail = None
        self.terrain_map = None
        self.last_track_point = None
        self.terrain_roughness = None
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.game_won = False
        self.last_rider_x = 0

        # Initialize state
        # self.reset() is called by the wrapper, but we can call it for standalone use.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS

        self.rider_pos = pygame.Vector2(5, self.WORLD_HEIGHT - 15)
        self.rider_vel = pygame.Vector2(0, 0)
        self.last_rider_x = self.rider_pos.x

        self.rider_trail = deque(maxlen=self.TRAIL_LENGTH)
        self.player_track = deque(maxlen=self.MAX_TRACK_SEGMENTS)

        self.terrain_roughness = 0.05
        self._generate_terrain()

        start_ground_y = self._get_ground_height_and_slope(self.rider_pos.x)[0]
        self.rider_pos.y = start_ground_y + self.RIDER_RADIUS
        
        # FIX: Initialize the drawing cursor at the rider's position, not below it.
        self.last_track_point = self.rider_pos.copy()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        if self.game_over:
            # Although not strictly required by the API, it's good practice
            # to return the last observation and a final info dict.
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Game Logic ---
        self._handle_drawing(action)
        self._update_physics()

        # --- State Updates ---
        self.steps += 1
        self.time_remaining -= 1
        if self.steps % 500 == 0:
            self.terrain_roughness += 0.01

        # --- Reward Calculation ---
        reward = self._calculate_reward()
        self.score += reward

        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS # Truncation for step limit

        if terminated or truncated:
            self.game_over = True
            if self.game_won:
                reward += 60  # Combined reward for finishing
            elif not truncated: # Don't penalize for truncation
                reward -= 10  # Penalty for crashing or timeout
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_drawing(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if shift_held:
            # Sound: Woosh (freefall)
            return  # Do not draw anything

        # Determine segment length
        length = 5.0 if space_held else 2.5

        # Determine angle based on movement action
        # Angles in radians, 0 is right, pi/2 is down
        angle_map = {
            0: 0.0,  # None -> Horizontal
            1: -math.pi / 4,  # Up -> Angled up
            2: math.pi / 4,  # Down -> Angled down
            3: -math.pi / 6,  # Left -> Angled slightly up-back (less common)
            4: math.pi / 6,  # Right -> Angled slightly down-forward
        }
        angle = angle_map[movement]

        # Create new segment
        start_point = self.last_track_point
        end_point = start_point + pygame.Vector2(length, 0).rotate_rad(angle)

        # Clamp to world bounds
        end_point.x = max(0, min(self.WORLD_WIDTH, end_point.x))
        end_point.y = max(0, min(self.WORLD_HEIGHT, end_point.y))

        self.player_track.append((start_point, end_point))
        self.last_track_point = end_point
        # Sound: Ink draw sound

    def _update_physics(self):
        # Add current position to trail
        self.rider_trail.append(self.rider_pos.copy())

        # Get ground properties under the rider
        ground_y, ground_slope = self._get_ground_height_and_slope(self.rider_pos.x)

        on_ground = self.rider_pos.y <= ground_y + 0.1

        if on_ground:
            # Snap to ground
            self.rider_pos.y = ground_y

            # Gravity along slope
            force = self.GRAVITY * math.sin(ground_slope)
            self.rider_vel.x += force

            # Apply friction
            self.rider_vel.x *= self.FRICTION

            # Align velocity vector to the slope to prevent bouncing
            vel_mag = self.rider_vel.length()
            if vel_mag > 0:
                slope_vec = pygame.Vector2(math.cos(ground_slope), math.sin(ground_slope))
                self.rider_vel = slope_vec * vel_mag * math.copysign(1, self.rider_vel.x)

            # Sound: Sled sliding on snow
        else:
            # Freefall
            self.rider_vel.y += self.GRAVITY

        # Update position
        self.rider_pos += self.rider_vel

    def _get_ground_height_and_slope(self, x_pos):
        # Terrain ground
        terrain_y, terrain_slope = self._get_terrain_properties(x_pos)

        # Player track ground
        track_y, track_slope = self._get_player_track_properties(x_pos)

        # FIX: Player track takes precedence if it's higher or at the same level.
        # Changed > to >= to ensure the drawn track is preferred for stability.
        if track_y is not None and track_y >= terrain_y:
            return track_y, track_slope

        return terrain_y, terrain_slope

    def _get_terrain_properties(self, x_pos):
        screen_x = int(x_pos / self.WORLD_WIDTH * self.SCREEN_WIDTH)
        if not (0 <= screen_x < self.SCREEN_WIDTH - 1):
            return 0, 0

        y1_screen = self.terrain_map[screen_x]
        y2_screen = self.terrain_map[screen_x + 1]

        y1_world = (self.SCREEN_HEIGHT - y1_screen) / self.SCREEN_HEIGHT * self.WORLD_HEIGHT
        y2_world = (self.SCREEN_HEIGHT - y2_screen) / self.SCREEN_HEIGHT * self.WORLD_HEIGHT

        dx_world = self.WORLD_WIDTH / self.SCREEN_WIDTH
        slope = math.atan2(y2_world - y1_world, dx_world)

        return y1_world, slope

    def _get_player_track_properties(self, x_pos):
        highest_y = -1
        best_slope = 0

        for start, end in self.player_track:
            if min(start.x, end.x) <= x_pos <= max(start.x, end.x):
                # Interpolate to find y at x_pos
                if abs(end.x - start.x) < 1e-6:
                    if end.x == start.x: # Vertical line
                        y = max(start.y, end.y)
                    else: # Nearly vertical
                        continue
                else:
                    t = (x_pos - start.x) / (end.x - start.x)
                    y = start.y + t * (end.y - start.y)

                if y > highest_y:
                    highest_y = y
                    dx = end.x - start.x
                    dy = end.y - start.y
                    best_slope = math.atan2(dy, dx)

        if highest_y == -1:
            return None, None
        return highest_y, best_slope

    def _calculate_reward(self):
        # Reward for forward progress
        progress = self.rider_pos.x - self.last_rider_x
        self.last_rider_x = self.rider_pos.x
        reward = progress * 0.1  # 10 reward for crossing the whole screen

        # Penalty for time passing
        reward -= 0.01

        return reward

    def _check_termination(self):
        # Reached finish line
        if self.rider_pos.x >= self.WORLD_WIDTH - 5:
            self.game_won = True
            return True

        # Crashed (out of bounds)
        if not (0 < self.rider_pos.y < self.WORLD_HEIGHT):
            return True
        if self.rider_pos.x < 0:
            return True

        # Time ran out
        if self.time_remaining <= 0:
            return True

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        # Render grid
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Render terrain
        terrain_points = [(i, y) for i, y in enumerate(self.terrain_map)]
        terrain_points.extend([(self.SCREEN_WIDTH - 1, self.SCREEN_HEIGHT), (0, self.SCREEN_HEIGHT)])
        pygame.draw.polygon(self.screen, self.COLOR_TERRAIN, terrain_points)

        # Render finish line
        finish_x_screen = self._world_to_screen_x(self.WORLD_WIDTH - 5)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_x_screen, 0), (finish_x_screen, self.SCREEN_HEIGHT), 3)

        # Render player-drawn track
        for start, end in self.player_track:
            p1 = self._world_to_screen(start)
            p2 = self._world_to_screen(end)
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 2)

        # Render drawing cursor
        cursor_pos = self._world_to_screen(self.last_track_point)
        pygame.draw.circle(self.screen, self.COLOR_TRACK, cursor_pos, 3, 1)

        # Render rider trail
        if len(self.rider_trail) > 1:
            trail_points = [self._world_to_screen(p) for p in self.rider_trail]
            pygame.draw.aalines(self.screen, self.COLOR_TRAIL, False, trail_points)

        # Render rider
        rider_screen_pos = self._world_to_screen(self.rider_pos)
        glow_radius = int(self.RIDER_RADIUS * self._get_screen_scale() * 2.5)

        # Glow effect
        if glow_radius > 0:
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_RIDER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (rider_screen_pos[0] - glow_radius, rider_screen_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Rider body
        pygame.gfxdraw.filled_circle(self.screen, rider_screen_pos[0], rider_screen_pos[1], int(self.RIDER_RADIUS * self._get_screen_scale()), self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_screen_pos[0], rider_screen_pos[1], int(self.RIDER_RADIUS * self._get_screen_scale()), self.COLOR_RIDER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_str = f"{self.time_remaining // self.FPS:02d}:{self.time_remaining % self.FPS * 100 // self.FPS:02d}"
        time_text = self.font_ui.render(f"Time: {time_str}", True, self.COLOR_UI)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        if self.game_over:
            msg = "FINISH!" if self.game_won else "GAME OVER"
            color = self.COLOR_FINISH if self.game_won else self.COLOR_TRACK
            msg_text = self.font_msg.render(msg, True, color)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def _generate_terrain(self):
        self.terrain_map = np.zeros(self.SCREEN_WIDTH)
        base_y = self.SCREEN_HEIGHT * 0.75

        # Use np_random for reproducibility
        freq1 = self.np_random.uniform(0.5, 1.5)
        freq2 = self.np_random.uniform(2.0, 3.0)
        amp1 = self.np_random.uniform(20, 40)
        amp2 = self.np_random.uniform(10, 20)
        rough_amp = self.SCREEN_HEIGHT * self.terrain_roughness

        for i in range(self.SCREEN_WIDTH):
            x = i / self.SCREEN_WIDTH
            wave1 = amp1 * math.sin(x * 2 * math.pi * freq1)
            wave2 = amp2 * math.sin(x * 2 * math.pi * freq2)
            downward_slope = i * 0.3
            noise = (self.np_random.integers(0, 2 * 100) / 100 - 1.0) * rough_amp # Use .integers

            self.terrain_map[i] = int(base_y + wave1 + wave2 - downward_slope + noise)

        self.terrain_map = np.clip(self.terrain_map, self.SCREEN_HEIGHT * 0.4, self.SCREEN_HEIGHT - 1)

    def _world_to_screen(self, pos):
        scale = self._get_screen_scale()
        x = int(pos.x * scale)
        y = int(self.SCREEN_HEIGHT - (pos.y * scale))
        return x, y

    def _world_to_screen_x(self, x_world):
        return int(x_world * self._get_screen_scale())

    def _get_screen_scale(self):
        return self.SCREEN_WIDTH / self.WORLD_WIDTH

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "rider_pos": (self.rider_pos.x, self.rider_pos.y),
            "game_over": self.game_over
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the game directly to test it
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)

    # --- Manual Play Setup ---
    pygame.display.set_caption("Sled Rider")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    # Game loop for manual play
    while running:
        # Action mapping from keyboard to MultiDiscrete
        keys = pygame.key.get_pressed()

        movement = 0  # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset(seed=42)
                total_reward = 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset(seed=42)
            total_reward = 0

        clock.tick(env.FPS)

    env.close()