
# Generated: 2025-08-27T19:31:33.546632
# Source Brief: brief_02176.md
# Brief Index: 2176

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, angle, speed, color, size, gravity=0):
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.size = size
        self.gravity = gravity
        self.life = random.uniform(20, 40)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.size = max(0, self.size - 0.1)
        self.life -= 1
        return self.life > 0 and self.size > 0

    def draw(self, surface, world_to_iso_func):
        iso_pos = world_to_iso_func(self.x, self.y)
        pygame.draw.circle(surface, self.color, (int(iso_pos[0]), int(iso_pos[1])), int(self.size))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑/↓ to cycle through crystals. Hold Space to rotate clockwise, Shift to rotate counter-clockwise."
    )

    game_description = (
        "Rotate crystals in an isometric cavern to redirect a light beam. Illuminate all 12 crystals before time runs out to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds

        # World space dimensions (for physics)
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 1000, 1000
        self.CRYSTAL_COUNT = 12
        self.CRYSTAL_RADIUS = 30
        self.ROTATION_SPEED = 2.0  # degrees per frame

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_TILE = (30, 35, 50)
        self.COLOR_BEAM = (255, 255, 100)
        self.COLOR_BEAM_CORE = (255, 255, 255)
        self.COLOR_CRYSTAL_UNLIT = (50, 100, 200)
        self.COLOR_CRYSTAL_LIT = (100, 255, 255)
        self.COLOR_CRYSTAL_LIT_GLOW = (100, 255, 255, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_SELECTION = (255, 255, 255)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = 0
        self.crystals = []
        self.light_source_pos = (0, 0)
        self.light_source_angle = 0.0
        self.light_path = []
        self.lit_crystal_count = 0
        self.selected_crystal_idx = 0
        self.prev_action = np.zeros(self.action_space.shape)
        self.particles = []
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.MAX_STEPS
        self.selected_crystal_idx = 0
        self.prev_action = np.zeros(self.action_space.shape)
        self.particles.clear()

        # Initialize crystal positions and angles
        self.crystals.clear()
        grid_size = 4
        spacing = self.WORLD_WIDTH / (grid_size + 1)
        for i in range(self.CRYSTAL_COUNT):
            row = (i % grid_size) + 1
            col = (i // grid_size) + 1
            x = col * spacing + self.np_random.uniform(-spacing/4, spacing/4)
            y = row * spacing + self.np_random.uniform(-spacing/4, spacing/4)
            angle = self.np_random.uniform(0, 360)
            self.crystals.append({"pos": (x, y), "angle": angle, "is_lit": False, "target_angle": angle})
        
        # Position light source
        self.light_source_pos = (self.np_random.uniform(100, 200), -50)
        self.light_source_angle = self.np_random.uniform(75, 105)

        self.lit_crystal_count = self._update_light_path()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Actions ---
        action_taken = False

        # Cycle selection on key press (not hold)
        if movement == 1 and self.prev_action[0] != 1: # Up
            self.selected_crystal_idx = (self.selected_crystal_idx + 1) % self.CRYSTAL_COUNT
        elif movement == 2 and self.prev_action[0] != 2: # Down
            self.selected_crystal_idx = (self.selected_crystal_idx - 1 + self.CRYSTAL_COUNT) % self.CRYSTAL_COUNT

        # Rotate selected crystal
        selected_crystal = self.crystals[self.selected_crystal_idx]
        if space_held:
            selected_crystal["target_angle"] += self.ROTATION_SPEED
            action_taken = True
        if shift_held:
            selected_crystal["target_angle"] -= self.ROTATION_SPEED
            action_taken = True
        
        self.prev_action = action

        # --- Update Game State ---
        self.steps += 1
        self.timer -= 1
        
        # Animate crystal rotation
        for crystal in self.crystals:
            diff = (crystal["target_angle"] - crystal["angle"] + 180) % 360 - 180
            if abs(diff) > 0.1:
                crystal["angle"] += diff * 0.2
                action_taken = True # Keep recalculating path while animating

        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # --- Calculate Reward & Update Light Path ---
        old_lit_count = self.lit_crystal_count
        reward = -0.001 # Small penalty for time passing

        if action_taken:
            self.lit_crystal_count = self._update_light_path()
            newly_lit = self.lit_crystal_count - old_lit_count
            if newly_lit > 0:
                reward += newly_lit * 1.0 # Reward for lighting a new crystal
            elif newly_lit < 0:
                reward += newly_lit * 0.5 # Penalty for un-lighting a crystal
        
        self.score += reward

        # --- Check Termination ---
        terminated = False
        if self.lit_crystal_count == self.CRYSTAL_COUNT:
            self.score += 100.0
            terminated = True
            # sfx: victory fanfare
        elif self.timer <= 0:
            self.score -= 10.0
            terminated = True
            # sfx: failure sound
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_light_path(self):
        self.light_path.clear()
        for crystal in self.crystals:
            crystal["is_lit"] = False
        
        lit_indices = set()
        current_pos = self.light_source_pos
        current_angle_deg = self.light_source_angle
        max_bounces = self.CRYSTAL_COUNT + 2

        for _ in range(max_bounces):
            intersection = self._find_beam_intersection(current_pos, current_angle_deg, lit_indices)
            
            if intersection is None:
                break

            intersect_pos, hit_object, hit_type = intersection
            self.light_path.append((current_pos, intersect_pos))
            
            if hit_type == "wall":
                break
            
            if hit_type == "crystal":
                crystal_idx = hit_object
                lit_indices.add(crystal_idx)
                
                crystal = self.crystals[crystal_idx]
                
                # Reflection logic
                incident_angle_rad = math.radians(current_angle_deg)
                crystal_normal_rad = math.radians(crystal["angle"] + 90)
                
                reflected_angle_rad = 2 * crystal_normal_rad - incident_angle_rad
                
                current_pos = intersect_pos
                current_angle_deg = math.degrees(reflected_angle_rad)
                # sfx: crystal ping
        
        for i in lit_indices:
            self.crystals[i]["is_lit"] = True
            
        return len(lit_indices)

    def _find_beam_intersection(self, start_pos, angle_deg, ignore_crystals):
        min_dist_sq = float('inf')
        closest_hit = None
        
        rad = math.radians(angle_deg)
        ray_dir = (math.cos(rad), math.sin(rad))

        # Check wall intersections
        # Top wall (y=0)
        if ray_dir[1] > 0:
            t = (0 - start_pos[1]) / ray_dir[1]
            if t > 1e-6:
                x = start_pos[0] + t * ray_dir[0]
                if 0 <= x <= self.WORLD_WIDTH:
                    dist_sq = t*t
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_hit = ((x, 0), None, "wall")
        # Bottom wall (y=WORLD_HEIGHT)
        if ray_dir[1] < 0:
            t = (self.WORLD_HEIGHT - start_pos[1]) / ray_dir[1]
            if t > 1e-6:
                x = start_pos[0] + t * ray_dir[0]
                if 0 <= x <= self.WORLD_WIDTH:
                    dist_sq = t*t
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_hit = ((x, self.WORLD_HEIGHT), None, "wall")
        # Left wall (x=0)
        if ray_dir[0] > 0:
            t = (0 - start_pos[0]) / ray_dir[0]
            if t > 1e-6:
                y = start_pos[1] + t * ray_dir[1]
                if 0 <= y <= self.WORLD_HEIGHT:
                    dist_sq = t*t
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_hit = ((0, y), None, "wall")
        # Right wall (x=WORLD_WIDTH)
        if ray_dir[0] < 0:
            t = (self.WORLD_WIDTH - start_pos[0]) / ray_dir[0]
            if t > 1e-6:
                y = start_pos[1] + t * ray_dir[1]
                if 0 <= y <= self.WORLD_HEIGHT:
                    dist_sq = t*t
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_hit = ((self.WORLD_WIDTH, y), None, "wall")

        # Check crystal intersections
        for i, crystal in enumerate(self.crystals):
            if i in ignore_crystals:
                continue

            oc = (start_pos[0] - crystal["pos"][0], start_pos[1] - crystal["pos"][1])
            b = 2 * (oc[0] * ray_dir[0] + oc[1] * ray_dir[1])
            c = oc[0]**2 + oc[1]**2 - self.CRYSTAL_RADIUS**2
            
            discriminant = b**2 - 4*c
            if discriminant >= 0:
                sqrt_d = math.sqrt(discriminant)
                t1 = (-b - sqrt_d) / 2
                t2 = (-b + sqrt_d) / 2
                
                if t1 > 1e-6:
                    dist_sq = t1*t1
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        intersect_pos = (start_pos[0] + t1 * ray_dir[0], start_pos[1] + t1 * ray_dir[1])
                        closest_hit = (intersect_pos, i, "crystal")
        
        return closest_hit

    def _world_to_iso(self, x, y):
        iso_x = self.WIDTH / 2 + (x - y) * 0.3
        iso_y = 50 + (x + y) * 0.15
        return iso_x, iso_y

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
            "time_left": self.timer / self.FPS,
            "lit_crystals": self.lit_crystal_count
        }

    def _render_game(self):
        self._render_background()
        self._render_light_beam()
        self._render_particles()
        self._render_crystals()

    def _render_background(self):
        tile_w, tile_h = 60, 30
        for x in range(0, self.WORLD_WIDTH, int(tile_w / 0.6)):
            for y in range(0, self.WORLD_HEIGHT, int(tile_h / 0.3)):
                iso_pos = self._world_to_iso(x, y)
                points = [
                    self._world_to_iso(x, y),
                    self._world_to_iso(x + tile_w, y),
                    self._world_to_iso(x + tile_w, y + tile_h),
                    self._world_to_iso(x, y + tile_h),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TILE)

    def _render_light_beam(self):
        if not self.light_path:
            return
        
        for start, end in self.light_path:
            iso_start = self._world_to_iso(start[0], start[1])
            iso_end = self._world_to_iso(end[0], end[1])
            
            pygame.draw.line(self.screen, self.COLOR_BEAM, iso_start, iso_end, 5)
            pygame.draw.line(self.screen, self.COLOR_BEAM_CORE, iso_start, iso_end, 1)

            # Spawn particles
            if self.np_random.random() < 0.8:
                dist = math.hypot(end[0] - start[0], end[1] - start[1])
                if dist > 0:
                    for _ in range(int(dist / 50)):
                        p = self.np_random.random()
                        px = start[0] * (1-p) + end[0] * p
                        py = start[1] * (1-p) + end[1] * p
                        self.particles.append(Particle(px, py, 0, 0, self.COLOR_BEAM, 2))


    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen, self._world_to_iso)

    def _render_crystals(self):
        for i, crystal in enumerate(self.crystals):
            iso_pos = self._world_to_iso(crystal["pos"][0], crystal["pos"][1])
            iso_radius = self.CRYSTAL_RADIUS * 0.3
            
            # Draw glow for lit crystals
            if crystal["is_lit"]:
                glow_radius = int(iso_radius * (1.5 + 0.2 * math.sin(self.steps * 0.1)))
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, self.COLOR_CRYSTAL_LIT_GLOW, (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surf, (iso_pos[0] - glow_radius, iso_pos[1] - glow_radius))
            
            # Draw crystal body (hexagon)
            color = self.COLOR_CRYSTAL_LIT if crystal["is_lit"] else self.COLOR_CRYSTAL_UNLIT
            points = []
            for j in range(6):
                angle_rad = math.radians(60 * j + 30)
                x = iso_pos[0] + iso_radius * math.cos(angle_rad)
                y = iso_pos[1] + iso_radius * math.sin(angle_rad)
                points.append((int(x), int(y)))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            
            # Draw selection indicator
            if i == self.selected_crystal_idx:
                pulse = 1.0 + 0.1 * math.sin(self.steps * 0.15)
                sel_radius = int(iso_radius * 1.2 * pulse)
                pygame.gfxdraw.aacircle(self.screen, int(iso_pos[0]), int(iso_pos[1]), sel_radius, self.COLOR_SELECTION)

            # Draw rotation indicator line
            angle_rad = math.radians(crystal["angle"])
            line_end_x = iso_pos[0] + (iso_radius * 0.8) * math.cos(angle_rad)
            line_end_y = iso_pos[1] + (iso_radius * 0.8) * math.sin(angle_rad)
            pygame.draw.line(self.screen, self.COLOR_BG, iso_pos, (line_end_x, line_end_y), 2)


    def _render_ui(self):
        # Lit crystal count
        lit_text = f"Lit: {self.lit_crystal_count} / {self.CRYSTAL_COUNT}"
        text_surf = self.font_large.render(lit_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (20, 15))

        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_text = f"Time: {time_left:.1f}"
        text_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(text_surf, text_rect)

        # Victory/Loss message
        if self.game_over:
            if self.lit_crystal_count == self.CRYSTAL_COUNT:
                message = "SUCCESS"
                color = self.COLOR_CRYSTAL_LIT
            else:
                message = "TIME OUT"
                color = (200, 50, 50)
            
            msg_surf = self.font_large.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            bg_rect = msg_rect.inflate(20, 20)
            pygame.draw.rect(self.screen, self.COLOR_BG, bg_rect, border_radius=5)
            self.screen.blit(msg_surf, msg_rect)


    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a display for human play
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crystal Cavern")
    
    running = True
    total_reward = 0.0
    
    # Main game loop for human play
    while running:
        # Action defaults
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3 # Not used by game logic but valid action
        elif keys[pygame.K_RIGHT]: movement = 4 # Not used by game logic but valid action
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = np.array([movement, space, shift])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0.0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            print("Press 'R' to reset.")

        env.clock.tick(env.FPS)
        
    env.close()