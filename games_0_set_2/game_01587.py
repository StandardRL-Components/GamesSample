import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import os
import pygame


# Set Pygame to run in a headless mode, which is required for Gymnasium environments
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to set the angle of the next track piece. "
        "Hold Space for a longer piece. Hold Shift for a steep 'booster' ramp."
    )

    game_description = (
        "Draw a track for your sled in real-time! Balance speed and safety to "
        "navigate the course and reach the finish line before time runs out."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH = 3200
    FINISH_LINE_X = 3000
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_TRACK = (200, 200, 210)
    COLOR_SLED = (220, 50, 50)
    COLOR_SLED_GLOW = (255, 100, 100)
    COLOR_START = (80, 200, 120)
    COLOR_FINISH = (200, 80, 120)
    COLOR_SPARK = (240, 240, 255)
    COLOR_SPEED_LINE = (100, 150, 255, 150)  # RGBA
    COLOR_UI_TEXT = (230, 230, 230)

    # Physics
    GRAVITY = 0.2
    FRICTION = 0.005
    AIR_RESISTANCE = 0.995
    BOUNCE_DAMPENING = 0.7

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.sled_pos = np.array([0.0, 0.0])
        self.sled_vel = np.array([0.0, 0.0])
        self.track_segments = collections.deque()
        self.last_track_end_point = np.array([0.0, 0.0])
        self.camera_pos = np.array([0.0, 0.0])
        self.sparks = []
        self.speed_lines = []

        self.steps = 0
        self.score = 0
        self.time_stationary = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_stationary = 0
        self.game_over = False

        self.sled_pos = np.array([150.0, 200.0])
        self.sled_vel = np.array([2.0, 0.0])

        self.track_segments.clear()
        initial_segment = (np.array([0.0, 220.0]), np.array([200.0, 220.0]))
        self.track_segments.append(initial_segment)
        self.last_track_end_point = initial_segment[1].copy()

        self.sparks.clear()
        self.speed_lines.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # Although the episode is over, we return the last observation.
            # A reward of 0 is given for any action taken after termination.
            # terminated is True, truncated is False (as it's a terminal state).
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Action: Draw Track ---
        self._handle_action(action)

        # --- 2. Update Physics & Game State ---
        old_pos_x = self.sled_pos[0]
        self._update_physics()
        self._update_particles()

        self.steps += 1

        # --- 3. Calculate Reward ---
        reward = 0.0
        # Reward for forward progress
        progress = self.sled_pos[0] - old_pos_x
        reward += progress * 0.1

        # Penalty for being stationary
        if np.linalg.norm(self.sled_vel) < 0.5:
            self.time_stationary += 1
            reward -= 0.05
        else:
            self.time_stationary = 0

        # --- 4. Check Termination ---
        terminated = False
        truncated = False
        if self.sled_pos[0] >= self.FINISH_LINE_X:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.sled_pos[1] > self.SCREEN_HEIGHT * 2 or self.time_stationary > 100:
            reward -= 10
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            # Truncated, not terminated, as it's a time limit, not a failure state
            truncated = True
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        # The action from MultiDiscrete is a numpy array.
        # It's safer to cast numpy integer types to native Python types
        # before using them in conditional logic or as dictionary keys.
        movement = int(action[0])
        space_held = bool(action[1])
        shift_held = bool(action[2])

        start_point = self.last_track_end_point.copy()

        if shift_held:  # Booster ramp
            length = 30
            angle_deg = -60
        else:
            length = 100 if space_held else 50
            angles = {0: 0, 1: -15, 2: 15, 3: -35, 4: 35}  # none, up, down, left(steep up), right(steep down)
            angle_deg = angles[movement]

        angle_rad = math.radians(angle_deg)
        end_point = start_point + np.array([length * math.cos(angle_rad), length * math.sin(angle_rad)])

        new_segment = (start_point, end_point)
        self.track_segments.append(new_segment)
        self.last_track_end_point = end_point

        # Limit total track segments to avoid performance issues
        while len(self.track_segments) > 100:
            self.track_segments.popleft()

        # Clean up segments far behind the sled
        while self.track_segments and self.track_segments[0][1][0] < self.sled_pos[0] - self.SCREEN_WIDTH:
            self.track_segments.popleft()

    def _update_physics(self):
        # Apply gravity and air resistance
        self.sled_vel[1] += self.GRAVITY
        self.sled_vel *= self.AIR_RESISTANCE

        predicted_pos = self.sled_pos + self.sled_vel

        on_ground = False
        for p1, p2 in self.track_segments:
            # Broad phase check
            if not (min(p1[0], p2[0]) - 5 < predicted_pos[0] < max(p1[0], p2[0]) + 5):
                continue

            # Find closest point on line segment to the sled
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue

            t = np.dot(predicted_pos - p1, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)
            closest_point = p1 + t * line_vec

            dist_vec = predicted_pos - closest_point
            dist = np.linalg.norm(dist_vec)

            sled_radius = 5  # Half of sled size
            if dist < sled_radius:
                on_ground = True

                # Collision response
                overlap = sled_radius - dist
                # Avoid division by zero if dist is zero
                if dist > 1e-6:
                    self.sled_pos += (dist_vec / dist) * overlap

                # Normal of the line segment
                norm_val = np.linalg.norm(line_vec)
                if norm_val < 1e-6: continue
                normal = np.array([line_vec[1], -line_vec[0]]) / norm_val

                # Make sure normal points "up" relative to sled
                if np.dot(normal, self.sled_vel) > 0:
                    normal *= -1

                # Reflect velocity
                self.sled_vel = self.sled_vel - (1 + self.BOUNCE_DAMPENING) * np.dot(self.sled_vel, normal) * normal

                # Apply friction
                speed = np.linalg.norm(self.sled_vel)
                if speed > 0:
                    friction_force = self.sled_vel / speed * self.FRICTION * 10  # Scaled for effect
                    self.sled_vel -= friction_force

                # Add sparks
                if random.random() < 0.8:  # sound: sled_grind.wav
                    for _ in range(3):
                        spark_vel = normal * random.uniform(1, 3) + (self.sled_vel * 0.2)
                        self.sparks.append({
                            "pos": self.sled_pos.copy(),
                            "vel": spark_vel + np.random.uniform(-0.5, 0.5, 2),
                            "life": random.randint(10, 20)
                        })
                break

        if not on_ground:
            self.sled_pos = predicted_pos

    def _update_particles(self):
        # Update sparks
        self.sparks = [s for s in self.sparks if s['life'] > 1]
        for spark in self.sparks:
            spark['pos'] += spark['vel']
            spark['life'] -= 1

        # Update speed lines
        self.speed_lines = [line for line in self.speed_lines if line['life'] > 1]
        for line in self.speed_lines:
            line['pos'][0] -= line['speed']
            line['life'] -= 1

        # Add new speed lines if moving fast
        speed = np.linalg.norm(self.sled_vel)
        if speed > 10:
            for _ in range(int(speed / 5)):
                self.speed_lines.append({
                    "pos": self.sled_pos + np.random.uniform(-10, 10, 2),
                    "len": random.uniform(speed * 0.5, speed * 1.5),
                    "speed": speed * 0.1,
                    "life": 15
                })

    def _get_observation(self):
        # Update camera to follow sled
        self.camera_pos[0] = self.sled_pos[0] - self.SCREEN_WIDTH / 4
        self.camera_pos[1] = self.sled_pos[1] - self.SCREEN_HEIGHT / 2
        self.camera_pos[0] = max(0, min(self.camera_pos[0], self.WORLD_WIDTH - self.SCREEN_WIDTH))

        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        cam_x, cam_y = self.camera_pos

        # Draw start/finish lines
        start_screen_x = int(50 - cam_x)
        pygame.draw.line(self.screen, self.COLOR_START, (start_screen_x, 0), (start_screen_x, self.SCREEN_HEIGHT), 3)
        finish_screen_x = int(self.FINISH_LINE_X - cam_x)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_screen_x, 0), (finish_screen_x, self.SCREEN_HEIGHT), 3)

        # Draw track
        for p1, p2 in self.track_segments:
            sp1 = (int(p1[0] - cam_x), int(p1[1] - cam_y))
            sp2 = (int(p2[0] - cam_x), int(p2[1] - cam_y))
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, sp1, sp2, 3)

        # Draw speed lines
        for line in self.speed_lines:
            start_pos = (int(line['pos'][0] - cam_x), int(line['pos'][1] - cam_y))
            end_pos = (int(line['pos'][0] - line['len'] - cam_x), int(line['pos'][1] - cam_y))
            # Pygame draw functions don't handle alpha well, so we create a surface
            line_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(line_surf, self.COLOR_SPEED_LINE, start_pos, end_pos, 2)
            self.screen.blit(line_surf, (0,0))


        # Draw sparks
        for spark in self.sparks:
            pos = (int(spark['pos'][0] - cam_x), int(spark['pos'][1] - cam_y))
            if 0 <= pos[0] < self.SCREEN_WIDTH and 0 <= pos[1] < self.SCREEN_HEIGHT:
                pygame.gfxdraw.pixel(self.screen, pos[0], pos[1], self.COLOR_SPARK)

        # Draw sled
        sled_screen_pos = (int(self.sled_pos[0] - cam_x), int(self.sled_pos[1] - cam_y))
        sled_rect = pygame.Rect(sled_screen_pos[0] - 5, sled_screen_pos[1] - 5, 10, 10)

        # Glow effect
        glow_radius = 10
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_SLED_GLOW + (50,), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (sled_rect.centerx - glow_radius, sled_rect.centery - glow_radius),
                         special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_SLED, sled_rect)

    def _render_ui(self):
        speed = np.linalg.norm(self.sled_vel) * 10  # Scale for display
        speed_text = self.font_small.render(f"SPEED: {speed:.0f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (10, 10))

        time_elapsed = self.steps / 60.0  # Assuming 60fps for display
        time_text = self.font_small.render(f"TIME: {time_elapsed:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        score_text = self.font_small.render(f"SCORE: {self.score:.0f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 35))

        if self.game_over:
            if self.sled_pos[0] >= self.FINISH_LINE_X:
                msg = "FINISH!"
            else:
                msg = "CRASHED"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sled_pos": self.sled_pos.tolist(),
            "sled_vel": self.sled_vel.tolist(),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires pygame to be installed with display support.
    # The environment itself runs headlessly.
    try:
        import pygame
        os.environ.pop("SDL_VIDEODRIVER", None)
        pygame.init()
        pygame.font.init()

        env = GameEnv()
        obs, info = env.reset()
        done = False

        # Pygame setup for human play
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Sled Rider")
        clock = pygame.time.Clock()

        movement = 0
        space_held = 0
        shift_held = 0

        print("--- Human Controls ---")
        print(GameEnv.user_guide)
        print("Press R to reset, Q to quit.")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False

            if not done:
                keys = pygame.key.get_pressed()

                # Map keys to MultiDiscrete action space
                # Movement
                if keys[pygame.K_UP]:
                    movement = 1
                elif keys[pygame.K_DOWN]:
                    movement = 2
                elif keys[pygame.K_LEFT]:
                    movement = 3
                elif keys[pygame.K_RIGHT]:
                    movement = 4
                else:
                    movement = 0

                # Space and Shift
                space_held = 1 if keys[pygame.K_SPACE] else 0
                shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

                action = [movement, space_held, shift_held]

                obs, reward, terminated, truncated, info = env.step(action)

                # Render the observation from the environment to the screen
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()

                if terminated or truncated:
                    print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
                    done = True  # Wait for reset

            clock.tick(60)  # Run at 60 FPS for smooth human play

        env.close()
    except ImportError:
        print("Pygame is not installed. Cannot run human-playable demo.")
    except pygame.error as e:
        print(f"Could not initialize display for human-playable demo: {e}")
        print("Please ensure you have a display environment set up (e.g., X11, Wayland, etc.).")
        print("The environment itself is running headlessly and is not affected.")