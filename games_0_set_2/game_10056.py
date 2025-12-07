import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle class for visual effects."""
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.initial_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.vel *= 0.98  # Damping

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.initial_lifespan))
            current_radius = int(self.radius * (self.lifespan / self.initial_lifespan))
            if current_radius > 0:
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, self.color + (alpha,), (current_radius, current_radius), current_radius)
                surface.blit(temp_surf, (int(self.pos.x - current_radius), int(self.pos.y - current_radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Weave threads of starlight to recreate celestial constellations before time runs out."
    user_guide = "Controls: Use arrow keys to move between stars. Press space to select a star and again on another to weave a thread. Press shift once per level for a time boost."
    auto_advance = True

    # --- Colors and Visual Constants ---
    COLOR_BG = (10, 5, 30)
    COLOR_BG_STARS = (50, 40, 80)
    COLOR_GUIDE = (40, 30, 70, 100)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TIMER_WARN = (255, 100, 100)
    ELEMENT_COLORS = {
        "fire": (255, 80, 20),
        "water": (60, 150, 255),
        "earth": (80, 220, 120),
        "light": (250, 250, 180),
        "dark": (180, 80, 255)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width = 640
        self.screen_height = 400
        self.fps = 30

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_star_idx = 0
        self.selected_star_idx = None
        self.woven_threads = set()
        self.time_remaining = 0
        self.time_limit = 0
        self.current_constellation_index = 0
        self.constellations = []
        self.stars = []
        self.bg_stars = []
        self.particles = []
        self.time_manipulation_used = False
        self.last_space_held = 0
        self.last_shift_held = 0

        self._setup_constellations()
        self.reset()
        
    def _setup_constellations(self):
        self.constellations = [
            {
                "name": "The Chalice",
                "stars": [
                    {"pos": (150, 200), "type": "water"}, {"pos": (200, 120), "type": "water"},
                    {"pos": (250, 200), "type": "water"}, {"pos": (200, 280), "type": "light"}
                ],
                "connections": {(0, 1), (1, 2), (0, 3), (2, 3)}
            },
            {
                "name": "The Crimson Arrow",
                "stars": [
                    {"pos": (450, 100), "type": "fire"}, {"pos": (400, 200), "type": "fire"},
                    {"pos": (500, 200), "type": "fire"}, {"pos": (450, 300), "type": "dark"},
                    {"pos": (350, 200), "type": "fire"}
                ],
                "connections": {(0, 1), (0, 2), (1, 3), (2, 3), (1,4)}
            },
            {
                "name": "Veridian Crown",
                "stars": [
                    {"pos": (100, 100), "type": "earth"}, {"pos": (200, 80), "type": "light"},
                    {"pos": (300, 100), "type": "earth"}, {"pos": (320, 200), "type": "earth"},
                    {"pos": (200, 250), "type": "dark"}, {"pos": (80, 200), "type": "earth"}
                ],
                "connections": {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)}
            }
        ]

    def _setup_level(self):
        level_data = self.constellations[self.current_constellation_index]
        self.stars = [
            {
                "pos": pygame.Vector2(s["pos"]),
                "color": self.ELEMENT_COLORS[s["type"]],
                "radius": 12,
                "type": s["type"]
            } for s in level_data["stars"]
        ]
        self.target_connections = level_data["connections"]
        self.woven_threads = set()
        self.selected_star_idx = None
        self.cursor_star_idx = 0
        self.time_limit = max(15, 60 - self.current_constellation_index * 5)
        self.time_remaining = self.time_limit * self.fps
        self.time_manipulation_used = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.last_space_held = 0
        self.last_shift_held = 0
        self.current_constellation_index = 0
        
        # Generate static background stars
        self.bg_stars = [(random.randint(0, self.screen_width), random.randint(0, self.screen_height), random.randint(1, 2)) for _ in range(100)]
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Handle Input ---
        movement, space_held, shift_held = action
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        reward += self._handle_input(movement, space_press, shift_press)
        
        # --- Update Game State ---
        self.time_remaining -= 1
        reward -= 0.1 / self.fps # Small penalty for time passing

        self._update_particles()
        
        # --- Check for Termination ---
        terminated = False
        truncated = False
        if self._check_constellation_complete():
            # SFX: Level Complete
            reward += 100
            self.score += 100
            self._create_burst_particles(self.screen.get_rect().center, self.COLOR_TEXT, 100, 10)
            self.current_constellation_index += 1
            if self.current_constellation_index >= len(self.constellations):
                terminated = True # Game won
            else:
                self._setup_level() # Next level

        if self.time_remaining <= 0:
            # SFX: Game Over
            if not terminated: # Avoid double penalty if won on last step
                reward -= 50
            terminated = True
        
        if self.steps >= 2000:
            truncated = True
        
        self.game_over = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _handle_input(self, movement, space_press, shift_press):
        reward = 0
        
        # --- Cursor Movement ---
        if movement != 0: # 1=up, 2=down, 3=left, 4=right
            current_pos = self.stars[self.cursor_star_idx]["pos"]
            best_target = -1
            min_dist_sq = float('inf')

            for i, star in enumerate(self.stars):
                if i == self.cursor_star_idx: continue
                
                vec = star["pos"] - current_pos
                if vec.length_squared() == 0: continue

                # Check if target is in the right direction
                is_correct_direction = False
                if movement == 1 and vec.y < 0 and abs(vec.y) > abs(vec.x): is_correct_direction = True # Up
                elif movement == 2 and vec.y > 0 and abs(vec.y) > abs(vec.x): is_correct_direction = True # Down
                elif movement == 3 and vec.x < 0 and abs(vec.x) > abs(vec.y): is_correct_direction = True # Left
                elif movement == 4 and vec.x > 0 and abs(vec.x) > abs(vec.y): is_correct_direction = True # Right
                
                if is_correct_direction and vec.length_squared() < min_dist_sq:
                    min_dist_sq = vec.length_squared()
                    best_target = i
            
            if best_target != -1:
                self.cursor_star_idx = best_target
                # SFX: Cursor Move

        # --- Space Press (Select/Weave) ---
        if space_press:
            if self.selected_star_idx is None:
                self.selected_star_idx = self.cursor_star_idx
                # SFX: Select Star
                self._create_burst_particles(self.stars[self.cursor_star_idx]["pos"], self.stars[self.cursor_star_idx]["color"], 20, 2)
            else:
                if self.selected_star_idx != self.cursor_star_idx:
                    # Create thread
                    idx1, idx2 = sorted((self.selected_star_idx, self.cursor_star_idx))
                    if (idx1, idx2) not in self.woven_threads:
                        self.woven_threads.add((idx1, idx2))
                        reward += 1
                        self.score += 1
                        # SFX: Weave Thread
                        self._create_thread_particles(self.stars[idx1]["pos"], self.stars[idx2]["pos"], self.stars[idx1]["color"])
                    self.selected_star_idx = None
                else: # Clicked same star again
                    self.selected_star_idx = None # Cancel selection

        # --- Shift Press (Time Manipulation) ---
        if shift_press and not self.time_manipulation_used:
            self.time_manipulation_used = True
            self.time_remaining += 5 * self.fps # Add 5 seconds
            self.time_remaining = min(self.time_remaining, self.time_limit * self.fps)
            reward += 5
            self.score += 5
            # SFX: Time Warp
            self._create_burst_particles(self.screen.get_rect().center, self.COLOR_CURSOR, 50, 5, is_ripple=True)

        return reward

    def _check_constellation_complete(self):
        return self.woven_threads.issuperset(self.target_connections)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_constellation_guide()
        self._render_threads()
        self._render_stars()
        self._render_cursor()
        if self.selected_star_idx is not None:
            self._render_active_thread()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "constellation": self.current_constellation_index + 1,
            "time_left_seconds": max(0, self.time_remaining / self.fps)
        }

    # --- Rendering Methods ---
    def _draw_glow_circle(self, surface, pos, radius, color, glow_layers=5, glow_factor=2.0):
        for i in range(glow_layers, 0, -1):
            alpha = int(255 / (glow_layers * 2) * (1 - i / glow_layers))
            current_radius = int(radius + (i * glow_factor))
            pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), current_radius, color + (alpha,))

    def _render_background(self):
        for x, y, r in self.bg_stars:
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_BG_STARS)
    
    def _render_constellation_guide(self):
        for idx1, idx2 in self.target_connections:
            pos1 = self.stars[idx1]["pos"]
            pos2 = self.stars[idx2]["pos"]
            pygame.draw.aaline(self.screen, self.COLOR_GUIDE, pos1, pos2)

    def _render_threads(self):
        for idx1, idx2 in self.woven_threads:
            is_correct = (idx1, idx2) in self.target_connections
            color = (255, 255, 255) if is_correct else (100, 100, 150)
            width = 3 if is_correct else 1
            pos1 = self.stars[idx1]["pos"]
            pos2 = self.stars[idx2]["pos"]
            pygame.draw.line(self.screen, color, pos1, pos2, width)

    def _render_stars(self):
        for i, star in enumerate(self.stars):
            pos_int = (int(star["pos"].x), int(star["pos"].y))
            self._draw_glow_circle(self.screen, pos_int, star["radius"], star["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], star["radius"], star["color"])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], star["radius"], (255, 255, 255))
    
    def _render_cursor(self):
        cursor_pos = self.stars[self.cursor_star_idx]["pos"]
        radius = self.stars[self.cursor_star_idx]["radius"] + 5
        angle = (self.steps * 4) % 360
        for i in range(4):
            a = math.radians(angle + i * 90)
            start = cursor_pos + pygame.Vector2(math.cos(a), math.sin(a)) * radius
            end = cursor_pos + pygame.Vector2(math.cos(a), math.sin(a)) * (radius + 5)
            pygame.draw.aaline(self.screen, self.COLOR_CURSOR, start, end)

    def _render_active_thread(self):
        start_pos = self.stars[self.selected_star_idx]["pos"]
        end_pos = self.stars[self.cursor_star_idx]["pos"]
        pygame.draw.aaline(self.screen, self.COLOR_CURSOR, start_pos, end_pos, 2)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Score
        score_surf = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Constellation Name
        name = self.constellations[self.current_constellation_index]["name"]
        name_surf = self.font_large.render(name, True, self.COLOR_TEXT)
        self.screen.blit(name_surf, (self.screen_width / 2 - name_surf.get_width() / 2, 10))

        # Timer
        time_sec = max(0, self.time_remaining / self.fps)
        timer_color = self.COLOR_TEXT if time_sec > 10 else self.COLOR_TIMER_WARN
        timer_surf = self.font_medium.render(f"TIME: {time_sec:.1f}", True, timer_color)
        self.screen.blit(timer_surf, (self.screen_width - timer_surf.get_width() - 10, 10))

    # --- Particle Helpers ---
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
    
    def _create_burst_particles(self, pos, color, count, speed_scale, is_ripple=False):
        for _ in range(count):
            if is_ripple:
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, 1) * speed_scale
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * radius
            else:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, speed_scale)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            
            p = Particle(pos, vel, random.randint(2, 5), color, random.randint(20, 40))
            self.particles.append(p)
    
    def _create_thread_particles(self, start_pos, end_pos, color):
        direction = (end_pos - start_pos).normalize()
        for i in range(15):
            # Spawn particles along the line
            frac = i / 14.0
            pos = start_pos.lerp(end_pos, frac)
            vel = direction * 2 + pygame.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
            p = Particle(pos, vel, 3, color, 20)
            self.particles.append(p)

    def close(self):
        pygame.quit()

# Example usage for interactive play
if __name__ == '__main__':
    # Set a real video driver for interactive play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for rendering the environment ---
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Cosmic Weaver")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Action state
    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_held = 0
    shift_held = 0

    print("--- Cosmic Weaver ---")
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("  Q: Quit")

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        # --- Get Player Input ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # --- Step the Environment ---
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render the observation ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode Finished!")
            print(f"Final Score: {info['score']}")
            print(f"Total Reward: {total_reward:.2f}")
            print("Resetting environment...")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before reset

        clock.tick(env.fps)

    env.close()