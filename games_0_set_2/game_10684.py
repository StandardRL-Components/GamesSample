import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Guide a squadron of drones through a winding tunnel, collecting orbs and avoiding obstacles."
    )
    user_guide = "Use ← and → arrow keys to steer the squadron left and right."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 2700  # 45 seconds at 60 FPS
        self.TUNNEL_LENGTH = 8000
        self.SAFE_START_LENGTH = self.HEIGHT * 2

        # Drone constants
        self.NUM_DRONES = 5
        self.DRONE_SIZE = 12
        self.DRONE_GLOW_RADIUS = 15
        self.DRONE_OSC_AMPLITUDE = 25
        self.DRONE_OSC_PERIOD = 90
        self.DRONE_NUDGE_SPEED = 3
        self.DRONE_SCREEN_Y = self.HEIGHT * 0.8

        # World constants
        self.SCROLL_SPEED = self.TUNNEL_LENGTH / self.MAX_STEPS
        self.TUNNEL_BASE_WIDTH = 180
        self.TUNNEL_WAVE_AMP = 50
        self.TUNNEL_WAVE_FREQ = 0.001
        self.NUM_OBSTACLES = 25
        self.NUM_ORBS = 40
        self.ORB_RADIUS = 8
        self.FINISH_LINE_Y = self.TUNNEL_LENGTH

        # Colors
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_WALL = (60, 80, 100)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_ORB = (50, 150, 255)
        self.COLOR_FINISH = (50, 255, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.DRONE_COLORS = [
            (255, 0, 128), (255, 128, 0), (255, 255, 0), (0, 255, 128), (0, 128, 255)
        ]
        self.STAR_COLORS = [(50, 60, 70), (80, 90, 100), (40, 50, 60)]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_status = pygame.font.Font(None, 48)

        # Initialize state variables
        self.drones = []
        self.obstacles = []
        self.orbs = []
        self.walls = {}
        self.stars = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.drones_finished = 0
        self.world_y_offset = 0.0
        self.squadron_center_x = self.WIDTH / 2

        # Initialize drones
        self.drones = []
        for i in range(self.NUM_DRONES):
            self.drones.append({
                "alive": True,
                "pos": pygame.Vector2(0, self.DRONE_SCREEN_Y),
                "color": self.DRONE_COLORS[i],
                "phase": (i / self.NUM_DRONES) * 2 * math.pi
            })

        # Procedurally generate level
        self._generate_tunnel()
        self._generate_obstacles()
        self._generate_orbs()
        self._generate_stars()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- 1. Handle Action ---
        movement = action[0]
        nudge = 0
        if movement == 3:  # Left
            nudge = -self.DRONE_NUDGE_SPEED
        elif movement == 4:  # Right
            nudge = self.DRONE_NUDGE_SPEED

        # --- 2. Update Game State ---
        self.world_y_offset += self.SCROLL_SPEED
        self.squadron_center_x += nudge
        self.squadron_center_x = np.clip(self.squadron_center_x, 0, self.WIDTH)

        self._update_drones()

        # --- 3. Collisions and Events ---
        collision_reward, orb_reward = self._handle_collisions()
        reward += collision_reward + orb_reward

        # --- 4. Check Termination Conditions ---
        terminated = self.game_over  # Set to true if a collision happened
        truncated = False

        # Win condition
        if self.drones_finished == self.NUM_DRONES:
            reward += 100
            terminated = True
            self.game_over = True

        # Timeout condition
        if not terminated and self.steps >= self.MAX_STEPS:
            reward -= 10
            truncated = True
            self.game_over = True

        # Survival reward
        if not terminated and not truncated:
            reward += 0.1

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_drones(self):
        for drone in self.drones:
            if not drone["alive"]:
                continue

            oscillation = self.DRONE_OSC_AMPLITUDE * math.sin(
                2 * math.pi * self.steps / self.DRONE_OSC_PERIOD + drone["phase"]
            )
            drone["pos"].x = self.squadron_center_x + oscillation

            # Check if drone crossed finish line
            if self.world_y_offset + (self.HEIGHT - self.DRONE_SCREEN_Y) > self.FINISH_LINE_Y:
                drone["alive"] = False  # Mark as finished
                self.drones_finished += 1

    def _handle_collisions(self):
        collision_reward = 0
        orb_reward = 0

        # Get tunnel wall positions at drone's y-level
        current_y = int(self.world_y_offset + (self.HEIGHT - self.DRONE_SCREEN_Y))
        if 0 <= current_y < self.TUNNEL_LENGTH:
            left_wall_x, right_wall_x = self.walls[current_y]
        else:
            left_wall_x, right_wall_x = -self.WIDTH, self.WIDTH * 2 # Wide boundaries outside tunnel

        for drone in self.drones:
            if not drone["alive"]:
                continue

            drone_rect = pygame.Rect(
                drone["pos"].x - self.DRONE_SIZE / 2,
                drone["pos"].y - self.DRONE_SIZE / 2,
                self.DRONE_SIZE, self.DRONE_SIZE
            )

            # Wall collision
            if not (left_wall_x < drone_rect.centerx < right_wall_x):
                drone["alive"] = False
                self.game_over = True
                collision_reward = -100
                break

            # Obstacle collision
            for obs_rect in self.obstacles:
                # Transform world obstacle rect to screen rect
                screen_obs_rect = obs_rect.move(0, -self.world_y_offset)
                if drone_rect.colliderect(screen_obs_rect):
                    drone["alive"] = False
                    self.game_over = True
                    collision_reward = -100
                    break
            if self.game_over: break

            # Orb collection
            for orb in self.orbs[:]:
                orb_pos_screen = (orb["pos"][0], orb["pos"][1] - self.world_y_offset)
                if drone_rect.collidepoint(orb_pos_screen):
                    self.orbs.remove(orb)
                    self.score += 1
                    orb_reward += 1

        return collision_reward, orb_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_world()
        self._render_drones()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.stars:
            # Parallax effect: slower stars are "further away"
            star_y = (star['y'] - self.world_y_offset * star['p']) % self.HEIGHT
            pygame.draw.circle(self.screen, star['c'], (int(star['x']), int(star_y)), star['r'])

    def _render_world(self):
        # Render tunnel walls
        visible_y_start = int(self.world_y_offset)
        visible_y_end = int(self.world_y_offset + self.HEIGHT)

        left_points, right_points = [], []
        for y_world in range(visible_y_start, visible_y_end + 1):
            if y_world in self.walls:
                y_screen = y_world - self.world_y_offset
                left_x, right_x = self.walls[y_world]
                left_points.append((left_x, y_screen))
                right_points.append((right_x, y_screen))

        if len(left_points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_WALL, False, left_points, 3)
            pygame.draw.lines(self.screen, self.COLOR_WALL, False, right_points, 3)

        # Render finish line
        finish_y_screen = self.FINISH_LINE_Y - self.world_y_offset
        if 0 < finish_y_screen < self.HEIGHT:
            left_x, right_x = self.walls.get(self.FINISH_LINE_Y - 1, (0, self.WIDTH))
            pygame.draw.line(self.screen, self.COLOR_FINISH, (left_x, finish_y_screen), (right_x, finish_y_screen), 5)

        # Render obstacles
        for obs_rect in self.obstacles:
            screen_rect = obs_rect.move(0, -self.world_y_offset)
            if self.screen.get_rect().colliderect(screen_rect):
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)
                pygame.draw.rect(self.screen, (255, 150, 150), screen_rect, 2)

        # Render orbs
        for orb in self.orbs:
            y_screen = orb["pos"][1] - self.world_y_offset
            if 0 < y_screen < self.HEIGHT:
                pulse = (math.sin(self.steps * 0.1 + orb["phase"]) + 1) / 2
                radius = self.ORB_RADIUS * (0.8 + 0.4 * pulse)
                color = self.COLOR_ORB

                # Glow effect for orbs
                glow_radius = int(radius * 1.8)
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*color, 50))
                self.screen.blit(glow_surf, (orb["pos"][0] - glow_radius, y_screen - glow_radius))

                pygame.gfxdraw.filled_circle(self.screen, int(orb["pos"][0]), int(y_screen), int(radius), color)
                pygame.gfxdraw.aacircle(self.screen, int(orb["pos"][0]), int(y_screen), int(radius), color)

    def _render_drones(self):
        for drone in self.drones:
            if drone["alive"]:
                # Glow effect
                glow_surf = pygame.Surface((self.DRONE_GLOW_RADIUS * 2, self.DRONE_GLOW_RADIUS * 2), pygame.SRCALPHA)
                glow_color = (*drone["color"], 80)
                pygame.gfxdraw.filled_circle(glow_surf, self.DRONE_GLOW_RADIUS, self.DRONE_GLOW_RADIUS,
                                              self.DRONE_GLOW_RADIUS, glow_color)
                self.screen.blit(glow_surf, (drone["pos"].x - self.DRONE_GLOW_RADIUS, drone["pos"].y - self.DRONE_GLOW_RADIUS))

                # Drone body
                rect = pygame.Rect(
                    drone["pos"].x - self.DRONE_SIZE / 2,
                    drone["pos"].y - self.DRONE_SIZE / 2,
                    self.DRONE_SIZE, self.DRONE_SIZE
                )
                pygame.draw.rect(self.screen, drone["color"], rect)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"ORBS: {self.score}/{self.NUM_ORBS}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}s", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Drones remaining
        alive_drones = sum(1 for d in self.drones if d["alive"])
        status_text = f"DRONES: {alive_drones}/{self.NUM_DRONES}"
        if self.drones_finished > 0:
            status_text += f" | FINISHED: {self.drones_finished}"

        drone_text = self.font_ui.render(status_text, True, self.COLOR_TEXT)
        self.screen.blit(drone_text, (self.WIDTH / 2 - drone_text.get_width() / 2, self.HEIGHT - 35))

        # Game over / Win message
        if self.game_over:
            if self.drones_finished == self.NUM_DRONES:
                msg = "MISSION COMPLETE"
                color = self.COLOR_FINISH
            else:
                msg = "MISSION FAILED"
                color = self.COLOR_OBSTACLE

            end_text = self.font_status.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "drones_alive": sum(1 for d in self.drones if d["alive"]),
            "drones_finished": self.drones_finished
        }

    def _generate_tunnel(self):
        self.walls = {}
        for y in range(self.TUNNEL_LENGTH):
            if y < self.SAFE_START_LENGTH:
                # Straight and wide section for a safe start
                center_x = self.WIDTH / 2
                width = self.TUNNEL_BASE_WIDTH + self.TUNNEL_WAVE_AMP
            else:
                # Original wavy tunnel, adjusted to start after the safe zone
                adjusted_y = y - self.SAFE_START_LENGTH
                center_x = self.WIDTH / 2 + self.TUNNEL_WAVE_AMP * math.sin(self.TUNNEL_WAVE_FREQ * adjusted_y)
                width = self.TUNNEL_BASE_WIDTH + self.TUNNEL_WAVE_AMP / 2 * math.sin(
                    self.TUNNEL_WAVE_FREQ * 2 * adjusted_y)
            self.walls[y] = (center_x - width / 2, center_x + width / 2)

    def _generate_obstacles(self):
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            y = self.np_random.uniform(self.SAFE_START_LENGTH, self.TUNNEL_LENGTH - self.HEIGHT)
            left_wall, right_wall = self.walls[int(y)]

            width = self.np_random.uniform(30, 80)
            height = self.np_random.uniform(20, 50)

            # Ensure obstacle is fully within the tunnel
            x_min = left_wall + 5
            x_max = right_wall - width - 5
            if x_min >= x_max: continue  # Tunnel too narrow here

            x = self.np_random.uniform(x_min, x_max)
            self.obstacles.append(pygame.Rect(x, y, width, height))

    def _generate_orbs(self):
        self.orbs = []
        for _ in range(self.NUM_ORBS):
            placed = False
            for _ in range(100): # Max 100 attempts to place an orb
                y = self.np_random.uniform(self.SAFE_START_LENGTH, self.TUNNEL_LENGTH - self.HEIGHT)
                left_wall, right_wall = self.walls[int(y)]

                # Ensure orb is within walls and not too close
                x_min = left_wall + self.ORB_RADIUS + 10
                x_max = right_wall - self.ORB_RADIUS - 10
                if x_min >= x_max: continue

                x = self.np_random.uniform(x_min, x_max)
                orb_pos = (x, y)
                orb_rect = pygame.Rect(x - self.ORB_RADIUS, y - self.ORB_RADIUS, self.ORB_RADIUS * 2, self.ORB_RADIUS * 2)

                # Check for overlap with obstacles
                if not any(obs.colliderect(orb_rect) for obs in self.obstacles):
                    self.orbs.append({"pos": orb_pos, "phase": self.np_random.uniform(0, 2 * math.pi)})
                    placed = True
                    break

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            color_idx = self.np_random.integers(0, len(self.STAR_COLORS))
            self.stars.append({
                'x': self.np_random.uniform(0, self.WIDTH),
                'y': self.np_random.uniform(0, self.HEIGHT),
                'r': self.np_random.choice([1, 2]),
                'p': self.np_random.uniform(0.1, 0.5),  # parallax factor
                'c': self.STAR_COLORS[color_idx]
            })

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # You might need to comment out the `os.environ` line at the top
    # to enable visual rendering.
    env = GameEnv()
    obs, info = env.reset()
    done = False

    # Create a display surface if not running in a truly headless environment
    try:
        display_surf = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Drone Squadron")
        can_render = True
    except pygame.error:
        can_render = False

    # Game loop
    running = True
    while running:
        action = [0, 0, 0]  # Default action: no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        if keys[pygame.K_r]:  # Reset on 'R' key
            obs, info = env.reset()

        if not env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}")


        if can_render:
            # The observation is (H, W, C). Pygame needs (W, H, C) for a surface.
            # And it's rotated. So we need to process it for display.
            draw_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_surf.blit(draw_surf, (0, 0))
            pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()