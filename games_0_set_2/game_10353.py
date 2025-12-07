import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:15:34.164275
# Source Brief: brief_00353.md
# Brief Index: 353
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper function to generate polygon vertices
def _generate_polygon_vertices(shape, size):
    """Generates vertices for a regular polygon centered at (0,0)."""
    if shape == "triangle":
        num_vertices = 3
        angle_offset = -math.pi / 2 # Point the triangle upwards
    elif shape == "square":
        num_vertices = 4
        angle_offset = math.pi / 4 # Align with axes
    elif shape == "hexagon":
        num_vertices = 6
        angle_offset = 0
    else:
        return []

    vertices = []
    for i in range(num_vertices):
        angle = 2 * math.pi * i / num_vertices + angle_offset
        x = size * math.cos(angle)
        y = size * math.sin(angle)
        vertices.append((x, y))
    return vertices

# Helper function to draw a rotated and translated polygon
def _draw_polygon(surface, vertices, position, angle_deg, color, width=0):
    """Draws a polygon with given transformation."""
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    transformed_points = []
    for x, y in vertices:
        # Rotate
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        # Translate
        tx = int(rx + position[0])
        ty = int(ry + position[1])
        transformed_points.append((tx, ty))

    if len(transformed_points) > 1:
        pygame.draw.polygon(surface, color, transformed_points, width)

# Helper for a visually appealing glow effect
def _draw_glowing_polygon(surface, vertices, position, angle_deg, base_color, glow_color, glow_steps=5):
    """Draws a polygon with a soft glow effect."""
    for i in range(glow_steps, 0, -1):
        glow_size_increase = i * 2
        alpha = int(100 / glow_steps * (1 - (i-1) / glow_steps))
        current_glow_color = (*glow_color, alpha)
        
        temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        
        glow_vertices = [(v[0] * (1 + glow_size_increase / 100), v[1] * (1 + glow_size_increase / 100)) for v in vertices]
        _draw_polygon(temp_surface, glow_vertices, position, angle_deg, current_glow_color)
        surface.blit(temp_surface, (0, 0))

    # Draw the main polygon on top
    _draw_polygon(surface, vertices, position, angle_deg, base_color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Align mystical glyphs to their ancient carvings. Move, rotate, and resize shapes to solve the temple's puzzles and unlock its secrets."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the glyph. Hold space to rotate. Hold shift and press ↑/↓ to resize the glyph."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # --- Visual & Gameplay Constants ---
        self.COLOR_BG = (25, 20, 30)
        self.COLOR_STONE_DARK = (45, 40, 50)
        self.COLOR_STONE_LIGHT = (65, 60, 70)
        self.COLOR_CARVING = (90, 80, 100)
        self.COLOR_CARVING_PULSE = (150, 140, 180)
        self.COLOR_GLYPH_ACTIVE = (0, 255, 255) # Cyan
        self.COLOR_GLYPH_INACTIVE = (180, 180, 180)
        self.COLOR_GLOW_ACTIVE = (0, 150, 255)
        self.COLOR_GLYPH_PLACED = (255, 220, 100) # Gold
        self.COLOR_GLOW_PLACED = (255, 180, 0)
        self.COLOR_UI = (220, 220, 240)

        self.MAX_STEPS = 1000
        self.MOVE_SPEED = 4.0
        self.ROTATE_SPEED = 3.0
        self.RESIZE_SPEED = 1.0
        self.SNAP_DISTANCE_POS = 20.0
        self.SNAP_DISTANCE_ROT = 15.0
        self.SNAP_DISTANCE_SIZE = 10.0
        
        self._define_temples()
        self.background_surface = self._create_background()
        
        # Use a numpy random generator
        self.np_random = None

    def _define_temples(self):
        self.temples = [
            [{"shape": "square", "size": 40, "pos": (320, 200), "rot": 0}],
            [
                {"shape": "triangle", "size": 50, "pos": (200, 200), "rot": 30},
                {"shape": "square", "size": 35, "pos": (440, 200), "rot": -15},
            ],
            [
                {"shape": "hexagon", "size": 40, "pos": (150, 200), "rot": 0},
                {"shape": "square", "size": 30, "pos": (320, 150), "rot": 45},
                {"shape": "triangle", "size": 45, "pos": (490, 250), "rot": 180},
            ],
            [
                {"shape": "hexagon", "size": 35, "pos": (120, 130), "rot": 15},
                {"shape": "hexagon", "size": 35, "pos": (120, 270), "rot": -15},
                {"shape": "square", "size": 50, "pos": (320, 200), "rot": 0},
                {"shape": "triangle", "size": 40, "pos": (520, 150), "rot": 90},
                {"shape": "triangle", "size": 40, "pos": (520, 250), "rot": -90},
            ]
        ]

    def _create_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        bg.fill(self.COLOR_BG)
        # Use a fixed seed for the background so it's consistent
        rng = np.random.default_rng(12345)
        for _ in range(3000):
            x = rng.integers(0, self.WIDTH + 1)
            y = rng.integers(0, self.HEIGHT + 1)
            size = rng.integers(1, 4)
            color_idx = rng.integers(0, 2)
            color = [self.COLOR_STONE_DARK, self.COLOR_STONE_LIGHT][color_idx]
            pygame.draw.rect(bg, color, (x, y, size, size))
        return bg

    def _setup_temple(self, level):
        self.temple_level = level
        if self.temple_level >= len(self.temples):
            self.game_over = True
            return

        self.carvings = []
        self.glyphs = []
        
        temple_spec = self.temples[self.temple_level]
        for i, spec in enumerate(temple_spec):
            self.carvings.append({
                "id": i, **spec, "is_filled": False, 
                "vertices": _generate_polygon_vertices(spec["shape"], spec["size"])
            })
            
            rand_x = self.np_random.uniform(50, self.WIDTH - 50)
            rand_y = self.np_random.uniform(50, self.HEIGHT - 50)
            rand_rot = self.np_random.uniform(0, 360)
            rand_size = spec["size"] * self.np_random.uniform(0.5, 1.5)
            
            self.glyphs.append({
                "target_id": i, **spec, "pos": (rand_x, rand_y), "rot": rand_rot, 
                "size": rand_size, "is_placed": False,
                "vertices": _generate_polygon_vertices(spec["shape"], 1)
            })
        
        self._select_next_active_glyph()

    def _select_next_active_glyph(self):
        unplaced_indices = [i for i, g in enumerate(self.glyphs) if not g["is_placed"]]
        if not unplaced_indices:
            self.active_glyph_idx = -1
        else:
            self.active_glyph_idx = unplaced_indices[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        
        self._setup_temple(0)
        
        return self._get_observation(), self._get_info()
    
    def _get_active_glyph(self):
        if self.active_glyph_idx != -1 and not self.game_over:
            return self.glyphs[self.active_glyph_idx]
        return None

    def _calculate_error(self, glyph, target):
        if not glyph or not target: return float('inf')
        
        pos_err = math.hypot(glyph["pos"][0] - target["pos"][0], glyph["pos"][1] - target["pos"][1])
        size_err = abs(glyph["size"] - target["size"]) * 2
        
        rot_diff = abs(glyph["rot"] - target["rot"]) % 360
        rot_err = min(rot_diff, 360 - rot_diff)
        
        return pos_err + size_err + rot_err

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        active_glyph = self._get_active_glyph()
        
        if active_glyph:
            target_carving = self.carvings[active_glyph["target_id"]]
            prev_error = self._calculate_error(active_glyph, target_carving)

            pos_x, pos_y = active_glyph["pos"]
            size = active_glyph["size"]
            
            if shift_held: # Resize mode
                if movement == 1: size += self.RESIZE_SPEED # Up
                elif movement == 2: size -= self.RESIZE_SPEED # Down
                active_glyph["size"] = max(5.0, size)
            else: # Movement mode
                if movement == 1: pos_y -= self.MOVE_SPEED # Up
                elif movement == 2: pos_y += self.MOVE_SPEED # Down
                elif movement == 3: pos_x -= self.MOVE_SPEED # Left
                elif movement == 4: pos_x += self.MOVE_SPEED # Right
                active_glyph["pos"] = (
                    max(0, min(self.WIDTH, pos_x)),
                    max(0, min(self.HEIGHT, pos_y))
                )

            if space_held: # Rotation
                active_glyph["rot"] = (active_glyph["rot"] + self.ROTATE_SPEED) % 360
            
            new_error = self._calculate_error(active_glyph, target_carving)
            reward += (prev_error - new_error) * 0.01
            
            pos_dist = math.hypot(active_glyph["pos"][0] - target_carving["pos"][0], active_glyph["pos"][1] - target_carving["pos"][1])
            size_diff = abs(active_glyph["size"] - target_carving["size"])
            rot_diff = abs(active_glyph["rot"] - target_carving["rot"]) % 360
            rot_dist = min(rot_diff, 360 - rot_diff)
            
            if (pos_dist < self.SNAP_DISTANCE_POS and 
                size_diff < self.SNAP_DISTANCE_SIZE and 
                rot_dist < self.SNAP_DISTANCE_ROT):
                
                active_glyph["pos"] = target_carving["pos"]
                active_glyph["size"] = target_carving["size"]
                active_glyph["rot"] = target_carving["rot"]
                active_glyph["is_placed"] = True
                target_carving["is_filled"] = True
                
                self.score += 5
                reward += 5
                self._spawn_particles(active_glyph["pos"][0], active_glyph["pos"][1], 50, self.COLOR_GLYPH_PLACED)
                
                self._select_next_active_glyph()
        
        self._update_particles()
        
        if all(c["is_filled"] for c in self.carvings):
            self.score += 50
            reward += 50
            self._setup_temple(self.temple_level + 1)
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False # This game does not have a truncation condition separate from termination

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] = (p["pos"][0] + p["vel"][0], p["pos"][1] + p["vel"][1])
            p["vel"] = (p["vel"][0], p["vel"][1] + 0.1) # Gravity
            p["lifespan"] -= 1

    def _spawn_particles(self, x, y, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": [x, y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": self.np_random.integers(20, 41),
                "color": color,
                "size": self.np_random.integers(1, 4)
            })

    def _render_background(self):
        self.screen.blit(self.background_surface, (0, 0))

    def _render_carvings(self):
        for c in self.carvings:
            width = 2 if not c["is_filled"] else 0
            color = self.COLOR_CARVING
            if c["is_filled"]:
                pulse = (math.sin(self.steps * 0.1) + 1) / 2
                color = (
                    int(self.COLOR_GLYPH_PLACED[0] * 0.3 + self.COLOR_CARVING_PULSE[0] * 0.7 * pulse),
                    int(self.COLOR_GLYPH_PLACED[1] * 0.3 + self.COLOR_CARVING_PULSE[1] * 0.7 * pulse),
                    int(self.COLOR_GLYPH_PLACED[2] * 0.3 + self.COLOR_CARVING_PULSE[2] * 0.7 * pulse)
                )
            _draw_polygon(self.screen, c["vertices"], c["pos"], c["rot"], color, width)

    def _render_glyphs(self):
        for i, g in enumerate(self.glyphs):
            if i != self.active_glyph_idx:
                color = self.COLOR_GLYPH_PLACED if g["is_placed"] else self.COLOR_GLYPH_INACTIVE
                glow_color = self.COLOR_GLOW_PLACED if g["is_placed"] else self.COLOR_GLYPH_INACTIVE
                scaled_vertices = [(v[0] * g["size"], v[1] * g["size"]) for v in g["vertices"]]
                _draw_glowing_polygon(self.screen, scaled_vertices, g["pos"], g["rot"], color, glow_color, glow_steps=3)
        
        active_glyph = self._get_active_glyph()
        if active_glyph:
            scaled_vertices = [(v[0] * active_glyph["size"], v[1] * active_glyph["size"]) for v in active_glyph["vertices"]]
            _draw_glowing_polygon(self.screen, scaled_vertices, active_glyph["pos"], active_glyph["rot"], self.COLOR_GLYPH_ACTIVE, self.COLOR_GLOW_ACTIVE, glow_steps=7)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 40.0))))
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p["color"], alpha), (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])))

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        temple_text = self.font.render(f"TEMPLE: {self.temple_level + 1}", True, self.COLOR_UI)
        steps_text = self.font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI)
        
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        self.screen.blit(temple_text, (10, 10))
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 35))
        
        if self.game_over:
            if self.temple_level >= len(self.temples):
                msg = "ALL TEMPLES UNLOCKED"
            else:
                msg = "TIME HAS RUN OUT"
            
            end_text = self.font.render(msg, True, self.COLOR_GLYPH_PLACED)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_observation(self):
        self._render_background()
        self._render_carvings()
        self._render_glyphs()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "temple": self.temple_level,
            "glyphs_placed": sum(1 for g in self.glyphs if g["is_placed"]),
            "total_glyphs": len(self.glyphs)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and testing
    # It will not be run by the evaluation server
    # but is useful for development.
    
    # Check if a display is available, otherwise create a dummy one
    if "SDL_VIDEODRIVER" not in os.environ:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        # For local testing, you might want a real display
        # os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Glyph Temple")
        is_headless = False
    except pygame.error:
        print("No display available. Running in headless mode.")
        is_headless = True
        # For headless, we don't need a real screen
        screen = None

    clock = pygame.time.Clock()
    running = True
    
    while running:
        movement = 0
        space_held = 0
        shift_held = 0
        
        # Only process Pygame events if a display is available
        if not is_headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        else:
            # In headless mode, you might want to send random or scripted actions
            action = env.action_space.sample()
            movement, space_held, shift_held = action[0], action[1], action[2]
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if not is_headless:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            if not is_headless:
                pygame.time.wait(2000)
            obs, info = env.reset(seed=42) # Reset with a seed for consistency

        clock.tick(30)
        
    env.close()