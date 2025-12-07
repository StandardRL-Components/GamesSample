import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:19:32.203793
# Source Brief: brief_00942.md
# Brief Index: 942
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a morphing, rotating polygon.
    The goal is to survive for 120 seconds by avoiding other rotating polygons.
    High-speed collisions grant a speed boost but still end the game.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a morphing, rotating polygon and survive for 120 seconds by avoiding other geometric shapes. "
        "Change your shape to navigate tight spaces."
    )
    user_guide = (
        "Use ↑/↓ arrows to change your rotation speed. Press space to add sides to your polygon and shift to remove them."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60.0
    GAME_DURATION_SECONDS = 120
    MAX_STEPS = int(GAME_DURATION_SECONDS * FPS)

    # Colors
    COLOR_BG = (15, 19, 26)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    OBSTACLE_COLORS = [(255, 80, 80), (255, 150, 80), (255, 220, 80)]
    OBSTACLE_GLOW_COLORS = [(200, 50, 50), (200, 120, 50), (200, 180, 50)]
    COLOR_PARTICLE = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (40, 45, 60, 150)

    # Player settings
    PLAYER_RADIUS = 35
    PLAYER_ROT_SPEED_CHANGE = 0.2  # rot/sec per step
    PLAYER_MAX_ROT_SPEED = 15.0 # rot/sec
    PLAYER_MIN_SIDES = 3
    PLAYER_MAX_SIDES = 8

    # Obstacle settings
    OBSTACLE_SPAWN_RATE_SECS = 1.5
    OBSTACLE_MIN_SPEED = 0.5  # rot/sec
    OBSTACLE_MAX_SPEED = 2.0  # rot/sec
    OBSTACLE_MIN_RADIUS = 20
    OBSTACLE_MAX_RADIUS = 50
    OBSTACLE_MIN_SIDES = 3
    OBSTACLE_MAX_SIDES = 6
    
    # Physics settings
    HIGH_SPEED_COLLISION_THRESHOLD = 5.0 # rot/sec
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_sides = pygame.font.SysFont("Consolas", 28, bold=True)
        
        # Game state variables are initialized in reset()
        self.player = {}
        self.obstacles = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.game_time = 0.0
        self.score = 0
        self.terminated = False
        self.win = False
        self.last_collision_type = None
        
        self.prev_space_pressed = False
        self.prev_shift_pressed = False

        self.obstacle_spawn_timer = 0.0
        
        self._create_stars()
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player = {
            "pos": pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2),
            "radius": self.PLAYER_RADIUS,
            "sides": self.PLAYER_MIN_SIDES,
            "rot_speed": 1.0,  # rotations per second
            "angle": 0.0,
        }

        self.obstacles.clear()
        self.particles.clear()
        
        self.steps = 0
        self.game_time = 0.0
        self.score = 0
        self.terminated = False
        self.win = False
        self.last_collision_type = None
        
        self.prev_space_pressed = False
        self.prev_shift_pressed = False
        
        self.obstacle_spawn_timer = 0.0

        if not self.stars:
            self._create_stars()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.dt = 1.0 / self.FPS
        self.game_time += self.dt
        self.steps += 1
        self.last_collision_type = None

        self._handle_input(action)
        self._update_player()
        self._update_obstacles()
        self._update_particles()
        
        self._check_collisions()
        self.terminated = self._check_termination()

        reward = self._calculate_reward()
        self.score += reward

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            self.terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_action, shift_action = action

        # Action 0: Movement (controls rotation speed)
        if movement == 1:  # Up
            self.player["rot_speed"] += self.PLAYER_ROT_SPEED_CHANGE
        elif movement == 2:  # Down
            self.player["rot_speed"] -= self.PLAYER_ROT_SPEED_CHANGE
        
        self.player["rot_speed"] = np.clip(self.player["rot_speed"], -self.PLAYER_MAX_ROT_SPEED, self.PLAYER_MAX_ROT_SPEED)
        
        # Action 1: Space (increase sides) - triggers on press (0->1)
        space_pressed = space_action == 1
        if space_pressed and not self.prev_space_pressed:
            # sfx: UI_MORPH_UP
            self.player["sides"] += 1
            if self.player["sides"] > self.PLAYER_MAX_SIDES:
                self.player["sides"] = self.PLAYER_MIN_SIDES
        self.prev_space_pressed = space_pressed

        # Action 2: Shift (decrease sides) - triggers on press (0->1)
        shift_pressed = shift_action == 1
        if shift_pressed and not self.prev_shift_pressed:
            # sfx: UI_MORPH_DOWN
            self.player["sides"] -= 1
            if self.player["sides"] < self.PLAYER_MIN_SIDES:
                self.player["sides"] = self.PLAYER_MAX_SIDES
        self.prev_shift_pressed = shift_pressed

    def _update_player(self):
        self.player["angle"] = (self.player["angle"] + self.player["rot_speed"] * 360 * self.dt) % 360

    def _update_obstacles(self):
        self.obstacle_spawn_timer += self.dt
        if self.obstacle_spawn_timer >= self.OBSTACLE_SPAWN_RATE_SECS:
            self.obstacle_spawn_timer = 0
            self._spawn_obstacle()

        for obs in self.obstacles[:]:
            obs["pos"] += obs["vel"] * self.dt
            obs["angle"] = (obs["angle"] + obs["rot_speed"] * 360 * self.dt) % 360
            
            # Remove if off-screen
            if not (-self.OBSTACLE_MAX_RADIUS < obs["pos"].x < self.SCREEN_WIDTH + self.OBSTACLE_MAX_RADIUS and \
                    -self.OBSTACLE_MAX_RADIUS < obs["pos"].y < self.SCREEN_HEIGHT + self.OBSTACLE_MAX_RADIUS):
                self.obstacles.remove(obs)
    
    def _spawn_obstacle(self):
        edge = self.np_random.integers(0, 4)
        if edge == 0: # Top
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -self.OBSTACLE_MAX_RADIUS)
        elif edge == 1: # Bottom
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.OBSTACLE_MAX_RADIUS)
        elif edge == 2: # Left
            pos = pygame.math.Vector2(-self.OBSTACLE_MAX_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        else: # Right
            pos = pygame.math.Vector2(self.SCREEN_WIDTH + self.OBSTACLE_MAX_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))

        target_point = pygame.math.Vector2(
            self.np_random.uniform(self.SCREEN_WIDTH * 0.3, self.SCREEN_WIDTH * 0.7),
            self.np_random.uniform(self.SCREEN_HEIGHT * 0.3, self.SCREEN_HEIGHT * 0.7)
        )
        velocity = (target_point - pos).normalize() * self.np_random.uniform(40, 80)
        
        color_index = self.np_random.integers(0, len(self.OBSTACLE_COLORS))

        self.obstacles.append({
            "pos": pos,
            "vel": velocity,
            "radius": self.np_random.uniform(self.OBSTACLE_MIN_RADIUS, self.OBSTACLE_MAX_RADIUS),
            "sides": self.np_random.integers(self.OBSTACLE_MIN_SIDES, self.OBSTACLE_MAX_SIDES + 1),
            "rot_speed": self.np_random.uniform(self.OBSTACLE_MIN_SPEED, self.OBSTACLE_MAX_SPEED) * self.np_random.choice([-1, 1]),
            "angle": self.np_random.uniform(0, 360),
            "color": self.OBSTACLE_COLORS[color_index],
            "glow_color": self.OBSTACLE_GLOW_COLORS[color_index]
        })
        
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"] * self.dt
            p["life"] -= self.dt
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_collisions(self):
        player_verts = self._get_polygon_vertices(self.player["pos"], self.player["sides"], self.player["radius"], self.player["angle"])
        
        for obs in self.obstacles:
            # Broad-phase check
            dist_sq = (self.player["pos"] - obs["pos"]).length_squared()
            if dist_sq > (self.player["radius"] + obs["radius"])**2:
                continue
                
            # Narrow-phase check (SAT)
            obs_verts = self._get_polygon_vertices(obs["pos"], obs["sides"], obs["radius"], obs["angle"])
            if self._check_sat_collision(player_verts, obs_verts):
                # sfx: COLLISION_HIGH or COLLISION_LOW
                if abs(self.player["rot_speed"]) > self.HIGH_SPEED_COLLISION_THRESHOLD:
                    self.last_collision_type = "high_speed"
                    # sfx: BOOST
                    self.player["rot_speed"] += 2.0 * np.sign(self.player["rot_speed"]) if self.player["rot_speed"] != 0 else 2.0
                else:
                    self.last_collision_type = "low_speed"
                    self.player["rot_speed"] *= 0.5

                collision_point = (self.player["pos"] + obs["pos"]) / 2
                self._create_particle_explosion(collision_point, 50)
                return

    def _check_sat_collision(self, verts1, verts2):
        axes = self._get_axes(verts1) + self._get_axes(verts2)
        for axis in axes:
            p1 = self._project_polygon(axis, verts1)
            p2 = self._project_polygon(axis, verts2)
            if not (p1[1] >= p2[0] and p2[1] >= p1[0]):
                return False # Found a separating axis
        return True # No separating axis found

    def _get_axes(self, vertices):
        axes = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            edge = p2 - p1
            normal = pygame.math.Vector2(-edge.y, edge.x).normalize()
            axes.append(normal)
        return axes

    def _project_polygon(self, axis, vertices):
        min_proj = vertices[0].dot(axis)
        max_proj = min_proj
        for i in range(1, len(vertices)):
            p = vertices[i].dot(axis)
            if p < min_proj:
                min_proj = p
            elif p > max_proj:
                max_proj = p
        return (min_proj, max_proj)

    def _calculate_reward(self):
        if self.last_collision_type == "high_speed":
            return 10.0
        if self.last_collision_type == "low_speed":
            return -5.0
        if self.win:
            return 100.0
        return 0.01 # Survival reward

    def _check_termination(self):
        if self.last_collision_type is not None:
            return True
        if self.game_time >= self.GAME_DURATION_SECONDS:
            self.win = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_obstacles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.stars:
            alpha = 50 + 40 * math.sin(self.game_time * star[3] + star[4])
            color = (*self.COLOR_UI_TEXT[:3], max(0, min(255, alpha)))
            pygame.gfxdraw.filled_circle(self.screen, int(star[0]), int(star[1]), int(star[2]), color)

    def _render_player(self):
        points = self._get_polygon_vertices(self.player["pos"], self.player["sides"], self.player["radius"], self.player["angle"])
        self._draw_glowing_polygon(self.screen, self.COLOR_PLAYER, points, self.COLOR_PLAYER_GLOW, 5, 8)

    def _render_obstacles(self):
        for obs in self.obstacles:
            points = self._get_polygon_vertices(obs["pos"], obs["sides"], obs["radius"], obs["angle"])
            self._draw_glowing_polygon(self.screen, obs["color"], points, obs["glow_color"], 3, 5)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / p["max_life"]))))
            color = (*self.COLOR_PARTICLE, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color)

    def _render_ui(self):
        # Speed display
        speed_text = f"SPEED: {self.player['rot_speed']:.1f} r/s"
        speed_surf = self.font_ui.render(speed_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_surf, (10, 10))

        # Timer display
        time_left = max(0, self.GAME_DURATION_SECONDS - self.game_time)
        timer_text = f"TIME: {time_left:.1f}s"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 10, 10))
        
        # Sides display
        sides_text = f'SIDES: {self.player["sides"]}'
        sides_surf = self.font_sides.render(sides_text, True, self.COLOR_UI_TEXT)
        ui_bg_rect = pygame.Rect(0, 0, sides_surf.get_width() + 20, sides_surf.get_height() + 10)
        ui_bg_rect.centerx = self.SCREEN_WIDTH / 2
        ui_bg_rect.bottom = self.SCREEN_HEIGHT - 10
        
        bg_surface = pygame.Surface(ui_bg_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(bg_surface, self.COLOR_UI_BG, bg_surface.get_rect(), border_radius=5)
        self.screen.blit(bg_surface, ui_bg_rect.topleft)
        
        sides_rect = sides_surf.get_rect(center=ui_bg_rect.center)
        self.screen.blit(sides_surf, sides_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "game_time": self.game_time,
            "player_rot_speed": self.player["rot_speed"],
            "player_sides": self.player["sides"],
            "win": self.win,
        }

    def _get_polygon_vertices(self, pos, sides, radius, angle_deg):
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(sides):
            theta = angle_rad + 2 * math.pi * i / sides
            x = pos.x + radius * math.cos(theta)
            y = pos.y + radius * math.sin(theta)
            vertices.append(pygame.math.Vector2(x, y))
        return vertices

    def _draw_glowing_polygon(self, surface, color, points, glow_color, glow_layers, glow_spread):
        # Ensure we have points to draw
        if not points:
            return
            
        # Create integer points for drawing
        int_points = [(int(p.x), int(p.y)) for p in points]
        
        # Draw glow layers
        for i in range(glow_layers, 0, -1):
            glow_alpha = 60 * (1 - i / glow_layers)
            temp_glow_color = (*glow_color, int(glow_alpha))
            
            # This is a simple way to expand the polygon for a glow effect
            # A more robust method would use geometry shaders or polygon offsetting
            expanded_points = []
            center = sum(points, pygame.math.Vector2()) / len(points)
            for p in points:
                direction = (p - center).normalize() if (p-center).length() > 0 else pygame.math.Vector2(0,0)
                expanded_points.append(p + direction * i * (glow_spread / glow_layers))
            
            int_expanded_points = [(int(p.x), int(p.y)) for p in expanded_points]
            if len(int_expanded_points) > 2:
                pygame.gfxdraw.filled_polygon(surface, int_expanded_points, temp_glow_color)
                pygame.gfxdraw.aapolygon(surface, int_expanded_points, temp_glow_color)
        
        # Draw main polygon
        if len(int_points) > 2:
            pygame.gfxdraw.filled_polygon(surface, int_points, color)
            pygame.gfxdraw.aapolygon(surface, int_points, color)

    def _create_particle_explosion(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(50, 150)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.uniform(0.5, 1.2)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": life,
                "max_life": life,
                "radius": self.np_random.uniform(1, 4)
            })
            
    def _create_stars(self):
        self.stars.clear()
        for _ in range(100):
            self.stars.append((
                self.np_random.uniform(0, self.SCREEN_WIDTH), # x
                self.np_random.uniform(0, self.SCREEN_HEIGHT),# y
                self.np_random.uniform(0.5, 1.5), # radius
                self.np_random.uniform(0.5, 1.5), # speed
                self.np_random.uniform(0, 2 * math.pi) # phase
            ))

    def close(self):
        pygame.quit()


# --- Example Usage ---
if __name__ == "__main__":
    # The original code had a validation function that is not standard for gym envs
    # and a main loop that requires a display. We keep the main loop for human play.
    
    # To run with a display, comment out the os.environ line at the top.
    # os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup for human play
    try:
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Polygon Survival")
        clock = pygame.time.Clock()
        running = True
        total_reward = 0
    except pygame.error:
        print("Pygame display not available. Running in headless mode.")
        running = False


    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        
        mov = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: mov = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: mov = 2
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Survived: {info['game_time']:.2f}s, Win: {info['win']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.FPS)

    env.close()