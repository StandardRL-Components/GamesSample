import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set the SDL video driver to "dummy" to run Pygame headlessly
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the drawing cursor. "
        "Hold [SPACE] to draw a line. Release to finalize. "
        "Press [SHIFT] to restart the sled at the cost of one life."
    )

    game_description = (
        "A physics-based puzzle game. Draw lines to guide the sled from the "
        "green start line to the blue finish line. Manage your limited restarts!"
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30

    # Physics
    GRAVITY = 0.4
    SLED_RADIUS = 7
    CURSOR_SPEED = 8
    FRICTION = 0.02
    RESTITUTION = 0.2
    MAX_LINE_LENGTH = 250
    STUCK_THRESHOLD_VEL = 0.1
    STUCK_TIME_LIMIT = 120  # 4 seconds at 30fps

    # Colors
    COLOR_BG = (245, 245, 250)
    COLOR_GRID = (225, 225, 230)
    COLOR_TRACK = (20, 20, 20)
    COLOR_SLED = (231, 76, 60)
    COLOR_SLED_OUTLINE = (192, 57, 43)
    COLOR_CURSOR = (52, 152, 219, 200)
    COLOR_GHOST_LINE = (41, 128, 185, 220)
    COLOR_START = (46, 204, 113)
    COLOR_FINISH = (52, 152, 219)
    COLOR_UI_TEXT = (44, 62, 80)
    COLOR_PARTICLE = (149, 165, 166)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        # This sets the video mode, which is required for operations like .convert_alpha()
        # even in headless mode.
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_msg = pygame.font.SysFont("Arial", 48, bold=True)

        self.sled_pos = pygame.math.Vector2(0, 0)
        self.sled_vel = pygame.math.Vector2(0, 0)
        self.on_ground = False
        self.track_lines = []
        self.cursor_pos = pygame.math.Vector2(0, 0)
        self.is_drawing = False
        self.line_start_pos = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_elapsed = 0.0
        self.restarts_remaining = 0
        self.stuck_timer = 0
        self.particles = []
        self.np_random = None

        self.start_x = 60
        self.finish_x = self.WIDTH - 60

        # Don't call reset in __init__ if it depends on a seeded RNG
        # self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)

        start_y = self.HEIGHT / 2 - 50
        self.start_pos = pygame.math.Vector2(self.start_x, start_y)
        self.sled_pos = pygame.math.Vector2(self.start_pos)
        self.sled_vel = pygame.math.Vector2(0, 0)
        self.on_ground = False

        self.track_lines = []
        # Add an initial flat ground line for stability
        floor_y = self.HEIGHT - 40
        self.track_lines.append(
            (pygame.math.Vector2(0, floor_y), pygame.math.Vector2(self.WIDTH, floor_y))
        )

        self.cursor_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.is_drawing = False
        self.line_start_pos = None
        self.prev_space_held = False
        self.prev_shift_held = False

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_elapsed = 0.0
        self.restarts_remaining = 3
        self.stuck_timer = 0

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._handle_input(movement, space_held, shift_held)
        self._update_physics()
        self._update_particles()

        self.steps += 1
        self.time_elapsed += 1 / self.FPS

        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        self.game_over = terminated

        if self.auto_advance:
            self.clock.tick(self.FPS)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        if space_held and not self.prev_space_held:
            self.is_drawing = True
            self.line_start_pos = pygame.math.Vector2(self.cursor_pos)

        if not space_held and self.prev_space_held and self.is_drawing:
            self.is_drawing = False
            end_pos = pygame.math.Vector2(self.cursor_pos)
            line_vec = end_pos - self.line_start_pos
            if line_vec.length_squared() > 0:
                if line_vec.length() > self.MAX_LINE_LENGTH:
                    end_pos = self.line_start_pos + line_vec.normalize() * self.MAX_LINE_LENGTH
                if (end_pos - self.line_start_pos).length() > 2:
                    self.track_lines.append((self.line_start_pos, end_pos))

        if shift_held and not self.prev_shift_held:
            self._restart_sled()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _restart_sled(self):
        if self.restarts_remaining > 0:
            self.restarts_remaining -= 1
            self.score -= 5.0
            self.sled_pos = pygame.math.Vector2(self.start_pos)
            self.sled_vel = pygame.math.Vector2(0, 0)
            self.stuck_timer = 0
            self.on_ground = False

    def _update_physics(self):
        if self.game_over: return

        self.sled_vel.y += self.GRAVITY
        self.sled_pos += self.sled_vel

        self.on_ground = False
        resolution_vector = pygame.math.Vector2(0, 0)

        for p1, p2 in self.track_lines:
            line_vec = p2 - p1
            line_len_sq = line_vec.length_squared()
            if line_len_sq == 0: continue

            t = max(0, min(1, (self.sled_pos - p1).dot(line_vec) / line_len_sq))
            closest_point = p1 + t * line_vec
            dist_vec = self.sled_pos - closest_point

            if dist_vec.length_squared() < self.SLED_RADIUS ** 2:
                dist = dist_vec.length()
                penetration = self.SLED_RADIUS - dist
                if penetration > 0:
                    self.on_ground = True
                    normal = dist_vec.normalize()
                    resolution_vector += normal * penetration

                    vel_dot_normal = self.sled_vel.dot(normal)
                    if vel_dot_normal < 0:
                        self.sled_vel -= (1 + self.RESTITUTION) * vel_dot_normal * normal

                        tangent = pygame.math.Vector2(normal.y, -normal.x)
                        tangent_vel_comp = self.sled_vel.dot(tangent)
                        friction_force = -tangent_vel_comp * self.FRICTION
                        self.sled_vel += friction_force * tangent

                        if self.sled_vel.length() > 2:
                            self._create_particles(self.sled_pos, 3, -normal)

        self.sled_pos += resolution_vector

        if self.sled_vel.length() < self.STUCK_THRESHOLD_VEL:
            self.stuck_timer += 1
        else:
            self.stuck_timer = 0

    def _calculate_reward(self):
        reward = 0.0
        if self.sled_pos.x > self.start_x:
            reward += 0.1 * np.clip(self.sled_vel.x / 10.0, 0, 1)
        else:
            reward -= 0.01

        if self.stuck_timer > 30: # Penalize being stuck
            reward -= 0.05

        return reward

    def _check_termination(self):
        if self.sled_pos.x >= self.finish_x:
            self.score += 100.0
            return True

        crashed = not (
            -self.SLED_RADIUS < self.sled_pos.x < self.WIDTH + self.SLED_RADIUS and
            -self.SLED_RADIUS < self.sled_pos.y < self.HEIGHT + self.SLED_RADIUS
        )
        stuck_out = self.stuck_timer > self.STUCK_TIME_LIMIT

        if crashed or stuck_out:
            if self.restarts_remaining <= 0:
                self.score -= 10.0
                return True
            else:
                self._restart_sled()
                return False

        if self.steps >= 5000:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_zones()
        self._render_track()
        self._render_particles()
        self._render_sled()
        self._render_cursor_and_ghost_line()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time": self.time_elapsed,
            "restarts": self.restarts_remaining,
            "sled_vx": self.sled_vel.x,
            "sled_vy": self.sled_vel.y,
        }

    def _render_grid(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_zones(self):
        pygame.draw.line(self.screen, self.COLOR_START, (self.start_x, 0), (self.start_x, self.HEIGHT), 3)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_x, 0), (self.finish_x, self.HEIGHT), 3)

    def _render_track(self):
        for p1, p2 in self.track_lines:
            pygame.draw.aaline(self.screen, self.COLOR_TRACK, p1, p2, 3)

    def _render_sled(self):
        pos = (int(self.sled_pos.x), int(self.sled_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.SLED_RADIUS, self.COLOR_SLED)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.SLED_RADIUS, self.COLOR_SLED_OUTLINE)

    def _render_cursor_and_ghost_line(self):
        cursor_pos_int = (int(self.cursor_pos.x), int(self.cursor_pos.y))
        s = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_CURSOR, (10, 10), 10, 2)
        self.screen.blit(s, (cursor_pos_int[0] - 10, cursor_pos_int[1] - 10))

        if self.is_drawing and self.line_start_pos:
            start_pos = (int(self.line_start_pos.x), int(self.line_start_pos.y))
            end_pos = cursor_pos_int
            line_vec = self.cursor_pos - self.line_start_pos
            if line_vec.length_squared() > 0:
                if line_vec.length() > self.MAX_LINE_LENGTH:
                    end_pos_vec = self.line_start_pos + line_vec.normalize() * self.MAX_LINE_LENGTH
                    end_pos = (int(end_pos_vec.x), int(end_pos_vec.y))

            temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(temp_surf, self.COLOR_GHOST_LINE, start_pos, end_pos, 4)
            self.screen.blit(temp_surf, (0,0))

    def _create_particles(self, pos, count, normal):
        for _ in range(count):
            if self.np_random is None: self.np_random, _ = gym.utils.seeding.np_random()
            angle = math.atan2(normal.y, normal.x) + (self.np_random.random() - 0.5) * math.pi / 2
            speed = self.np_random.random() * 2 + 1
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.math.Vector2(pos),
                "vel": vel,
                "life": self.np_random.integers(10, 25),
                "size": self.np_random.random() * 2 + 2,
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["life"] -= 1
            p["size"] -= 0.1

    def _render_particles(self):
        for p in self.particles:
            if p["size"] > 0:
                pos = (int(p["pos"].x), int(p["pos"].y))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["size"]), self.COLOR_PARTICLE)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        time_text = self.font_ui.render(f"TIME: {self.time_elapsed:.1f}s", True, self.COLOR_UI_TEXT)
        restarts_text = self.font_ui.render(f"RESTARTS: {self.restarts_remaining}", True, self.COLOR_UI_TEXT)
        speed_text = self.font_ui.render(f"SPEED: {self.sled_vel.length():.1f}", True, self.COLOR_UI_TEXT)

        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (10, 30))
        self.screen.blit(restarts_text, (10, 50))
        self.screen.blit(speed_text, (self.WIDTH - 120, 10))

        if self.game_over:
            msg = "FINISH!" if self.sled_pos.x >= self.finish_x else "GAME OVER"
            color = self.COLOR_FINISH if msg == "FINISH!" else self.COLOR_SLED
            msg_text = self.font_msg.render(msg, True, color)
            text_rect = msg_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It requires a graphical environment.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS", etc.
    
    try:
        import pygame
        print(GameEnv.user_guide)
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        # The main 'screen' is now the display screen from env
        screen = env.screen
        pygame.display.set_caption("Line Rider Gym")
        
        terminated = False
        total_reward = 0
        
        movement_action = 0
        space_action = 0
        shift_action = 0

        running = True
        while running:
            # Pygame event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Get keyboard state for continuous actions
            keys = pygame.key.get_pressed()
            
            movement_action = 0 # No movement
            if keys[pygame.K_UP]: movement_action = 1
            elif keys[pygame.K_DOWN]: movement_action = 2
            elif keys[pygame.K_LEFT]: movement_action = 3
            elif keys[pygame.K_RIGHT]: movement_action = 4
            
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement_action, space_action, shift_action]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
                total_reward = 0
                obs, info = env.reset()
                pygame.time.wait(2000) # Pause for 2 seconds on game over
        
        env.close()

    except (ImportError, pygame.error) as e:
        print(f"Could not run manual play mode: {e}")
        print("Running a short automated test instead.")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        env = GameEnv()
        obs, info = env.reset()
        for i in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished after {i} steps. Score: {info['score']:.2f}")
                obs, info = env.reset()
        env.close()
        print("Automated test complete.")