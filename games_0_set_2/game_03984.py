
# Generated: 2025-08-28T01:02:55.269714
# Source Brief: brief_03984.md
# Brief Index: 3984

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper for particles
Particle = namedtuple("Particle", ["pos", "vel", "life", "color", "radius"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle through crystal types. "
        "Press Space to place the selected crystal."
    )

    game_description = (
        "A strategic puzzle game. Place reflective crystals to guide a light beam through a cavern "
        "and illuminate all the gems before you run out of time or crystals."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.TILE_W, self.TILE_H = 20, 10
        self.ORIGIN_X, self.ORIGIN_Y = self.WIDTH // 2, 60

        self.MAX_STEPS = 6000
        self.STARTING_CRYSTALS = 20
        self.NUM_GEMS = 10
        self.NUM_WALL_CLUSTERS = 8
        self.MAX_BEAM_REFLECTIONS = 30
        self.BEAM_MAX_LEN = 1000

        # --- Colors ---
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_WALL = (40, 30, 60)
        self.COLOR_WALL_ACCENT = (60, 50, 90)
        self.COLOR_GRID = (25, 20, 40)
        self.COLOR_LIGHT_BEAM = (255, 255, 255)
        self.COLOR_LIGHT_GLOW = (100, 200, 255)
        self.COLOR_GEM_UNLIT = (100, 100, 120)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 30)

        self.CRYSTAL_SPECS = [
            {"angle": 45, "color": (255, 0, 255), "glow": (150, 0, 150)}, # Magenta
            {"angle": -45, "color": (255, 255, 0), "glow": (150, 150, 0)}, # Yellow
            {"angle": 90, "color": (0, 255, 255), "glow": (0, 150, 150)}, # Cyan
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.grid = None
        self.gems = None
        self.light_source = None
        self.placed_crystals = None
        self.light_paths = None
        self.lit_gem_indices = None
        self.crystals_remaining = 0
        self.cursor_pos = None
        self.selected_crystal_type = 0
        self.space_was_held = False
        self.shift_was_held = False
        self.particles = []
        self.last_reward = 0
        self.termination_reason = ""

        # Run validation
        # self.validate_implementation() # Commented out for submission as per instructions

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_reward = 0
        self.termination_reason = ""

        self.crystals_remaining = self.STARTING_CRYSTALS
        self.cursor_pos = np.array([self.GRID_W // 2, self.GRID_H // 2])
        self.selected_crystal_type = 0
        self.placed_crystals = []
        self.particles = []
        
        self.space_was_held = True # Prevent action on first frame
        self.shift_was_held = True

        self._generate_level()
        self._recalculate_all_light()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_action, shift_action = action
        space_held = space_action == 1
        shift_held = shift_action == 1

        reward = 0
        action_taken = False

        # --- Handle Input ---
        space_pressed = space_held and not self.space_was_held
        shift_pressed = shift_held and not self.shift_was_held

        # Move cursor
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 2 and self.cursor_pos[1] < self.GRID_H - 1: self.cursor_pos[1] += 1
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 4 and self.cursor_pos[0] < self.GRID_W - 1: self.cursor_pos[0] += 1

        # Cycle crystal type
        if shift_pressed:
            self.selected_crystal_type = (self.selected_crystal_type + 1) % len(self.CRYSTAL_SPECS)
            # sfx: crystal_cycle.wav
            self._spawn_particles(self._to_iso(*self.cursor_pos), 5, self.CRYSTAL_SPECS[self.selected_crystal_type]['color'])

        # Place crystal
        if space_pressed and self._is_valid_placement(self.cursor_pos[0], self.cursor_pos[1]):
            action_taken = True
            crystal_spec = self.CRYSTAL_SPECS[self.selected_crystal_type]
            self.placed_crystals.append({
                "pos": tuple(self.cursor_pos),
                "type": self.selected_crystal_type,
                "angle": crystal_spec["angle"]
            })
            self.crystals_remaining -= 1
            # sfx: crystal_place.wav
            self._spawn_particles(self._to_iso(*self.cursor_pos), 20, crystal_spec['color'])

        self.space_was_held = space_held
        self.shift_was_held = shift_held
        
        # --- Update Game State ---
        if action_taken:
            prev_lit_count = len(self.lit_gem_indices)
            self._recalculate_all_light()
            newly_lit_count = len(self.lit_gem_indices) - prev_lit_count

            if newly_lit_count > 0:
                reward += newly_lit_count * 0.1
                # sfx: gem_lit.wav

        self.steps += 1
        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        gems_lit_count = len(self.lit_gem_indices)

        if gems_lit_count == self.NUM_GEMS:
            terminated = True
            reward += 100 + 1 # Win bonus + final gem bonus
            self.game_over = True
            self.termination_reason = "All Gems Illuminated!"
            # sfx: win_jingle.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward = -100 # Time out penalty
            self.game_over = True
            self.termination_reason = "Time Limit Reached"
            # sfx: lose_sound.wav
        elif action_taken and self.crystals_remaining < 0: # Should be 0, but < is safer
             # Check if won on last crystal
            if gems_lit_count != self.NUM_GEMS:
                terminated = True
                reward = -100 # Out of crystals penalty
                self.game_over = True
                self.termination_reason = "Out of Crystals"
                # sfx: lose_sound.wav

        self.score += reward
        self.last_reward = reward
        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "crystals_remaining": self.crystals_remaining,
            "gems_lit": len(self.lit_gem_indices),
        }

    def _to_iso(self, gx, gy):
        sx = self.ORIGIN_X + (gx - gy) * self.TILE_W
        sy = self.ORIGIN_Y + (gx + gy) * self.TILE_H
        return int(sx), int(sy)

    def _generate_level(self):
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=int)
        
        # Place light source
        source_y = self.np_random.integers(self.GRID_H // 4, 3 * self.GRID_H // 4)
        self.light_source = {"pos": (0, source_y), "angle": 0}

        # Place walls
        for _ in range(self.NUM_WALL_CLUSTERS):
            w = self.np_random.integers(1, 4)
            h = self.np_random.integers(1, 4)
            x = self.np_random.integers(2, self.GRID_W - w - 2)
            y = self.np_random.integers(0, self.GRID_H - h)
            self.grid[x:x+w, y:y+h] = 1

        # Place gems, ensuring not on walls or too close to edge
        self.gems = []
        attempts = 0
        while len(self.gems) < self.NUM_GEMS and attempts < 1000:
            gx = self.np_random.integers(1, self.GRID_W - 1)
            gy = self.np_random.integers(1, self.GRID_H - 1)
            if self.grid[gx, gy] == 0:
                is_duplicate = any(g['pos'] == (gx, gy) for g in self.gems)
                if not is_duplicate:
                    self.gems.append({"pos": (gx, gy), "is_lit": False})
            attempts += 1
        
        # Clear space around light source
        self.grid[self.light_source['pos'][0], self.light_source['pos'][1]] = 0
        self.grid[self.light_source['pos'][0]+1, self.light_source['pos'][1]] = 0


    def _recalculate_all_light(self):
        self.light_paths = []
        self.lit_gem_indices = set()
        
        start_pos_world = self._to_iso(*self.light_source['pos'])
        
        beams_to_process = [(start_pos_world, self.light_source['angle'], self.MAX_BEAM_REFLECTIONS)]

        processed_beam_starts = set()

        while beams_to_process:
            start_p, angle, reflections_left = beams_to_process.pop(0)

            if reflections_left <= 0:
                continue

            # Avoid re-processing beams from the exact same spot and angle
            key = (start_p, angle)
            if key in processed_beam_starts:
                continue
            processed_beam_starts.add(key)
            
            rad = math.radians(angle)
            direction = pygame.Vector2(math.cos(rad), math.sin(rad))
            
            end_p = start_p
            for i in range(self.BEAM_MAX_LEN):
                end_p += direction
                
                # Check for gem intersections along the path
                for idx, gem in enumerate(self.gems):
                    gem_p = self._to_iso(*gem['pos'])
                    if pygame.Vector2(gem_p).distance_to(end_p) < 8:
                        self.lit_gem_indices.add(idx)

                # Check for crystal intersections
                hit_crystal = False
                for crystal in self.placed_crystals:
                    crystal_p = self._to_iso(*crystal['pos'])
                    if pygame.Vector2(crystal_p).distance_to(end_p) < 10:
                        new_angle = (angle + crystal['angle']) % 360
                        beams_to_process.append((crystal_p, new_angle, reflections_left - 1))
                        hit_crystal = True
                        break
                if hit_crystal:
                    break

                # Check for wall/boundary intersections
                gx, gy = -1, -1
                try: # Inverse ISO is tricky, so we approximate by checking nearby grid cells
                    min_dist = float('inf')
                    for ix in range(self.GRID_W):
                        for iy in range(self.GRID_H):
                            iso_p = self._to_iso(ix, iy)
                            dist = pygame.Vector2(iso_p).distance_to(end_p)
                            if dist < min_dist:
                                min_dist = dist
                                gx, gy = ix, iy
                except:
                    pass
                
                if not (0 <= gx < self.GRID_W and 0 <= gy < self.GRID_H):
                    break # Out of bounds
                if self.grid[gx, gy] == 1:
                    break # Hit wall

            self.light_paths.append((start_p, end_p))

        for i, gem in enumerate(self.gems):
            gem['is_lit'] = i in self.lit_gem_indices

    def _is_valid_placement(self, gx, gy):
        if not (0 <= gx < self.GRID_W and 0 <= gy < self.GRID_H):
            return False
        if self.grid[gx, gy] == 1:
            return False
        if self.crystals_remaining <= 0:
            return False
        if any(c['pos'] == (gx, gy) for c in self.placed_crystals):
            return False
        if any(g['pos'] == (gx, gy) for g in self.gems):
            return False
        return True

    def _render_game(self):
        # Render grid floor
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                p1 = self._to_iso(x, y)
                p2 = self._to_iso(x + 1, y)
                p3 = self._to_iso(x + 1, y + 1)
                p4 = self._to_iso(x, y + 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

        # Render walls
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self.grid[x, y] == 1:
                    p = self._to_iso(x, y)
                    iso_rect = [
                        self._to_iso(x, y),
                        self._to_iso(x + 1, y),
                        self._to_iso(x + 1, y + 1),
                        self._to_iso(x, y + 1)
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, iso_rect, self.COLOR_WALL)
                    pygame.gfxdraw.aapolygon(self.screen, iso_rect, self.COLOR_WALL_ACCENT)
        
        # Render gems
        for idx, gem in enumerate(self.gems):
            p = self._to_iso(*gem['pos'])
            is_lit = idx in self.lit_gem_indices
            gem_color = self.CRYSTAL_SPECS[idx % len(self.CRYSTAL_SPECS)]['color'] if is_lit else self.COLOR_GEM_UNLIT
            
            if is_lit:
                pygame.gfxdraw.filled_circle(self.screen, p[0], p[1], 10, (*gem_color, 50))
                pygame.gfxdraw.filled_circle(self.screen, p[0], p[1], 7, (*gem_color, 100))
            
            points = [
                (p[0], p[1] - 8), (p[0] + 6, p[1]),
                (p[0], p[1] + 8), (p[0] - 6, p[1])
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, gem_color)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TEXT)

        # Render placed crystals
        for crystal in self.placed_crystals:
            p = self._to_iso(*crystal['pos'])
            spec = self.CRYSTAL_SPECS[crystal['type']]
            pygame.gfxdraw.filled_circle(self.screen, p[0], p[1], 12, (*spec['glow'], 60))
            pygame.gfxdraw.filled_circle(self.screen, p[0], p[1], 8, spec['color'])
            pygame.gfxdraw.aacircle(self.screen, p[0], p[1], 8, self.COLOR_TEXT)

        # Render light source
        p = self._to_iso(*self.light_source['pos'])
        pygame.gfxdraw.filled_circle(self.screen, p[0], p[1], 10, self.COLOR_LIGHT_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, p[0], p[1], 6, self.COLOR_LIGHT_BEAM)

        # Render light paths
        for start_p, end_p in self.light_paths:
            pygame.draw.line(self.screen, self.COLOR_LIGHT_GLOW, start_p, end_p, 5)
            pygame.draw.aaline(self.screen, self.COLOR_LIGHT_BEAM, start_p, end_p, 1)

        # Render cursor
        if not self.game_over:
            p = self._to_iso(*self.cursor_pos)
            spec = self.CRYSTAL_SPECS[self.selected_crystal_type]
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            
            if self._is_valid_placement(self.cursor_pos[0], self.cursor_pos[1]):
                color = spec['color']
                alpha = int(100 + pulse * 100)
            else:
                color = (255, 0, 0)
                alpha = int(50 + pulse * 50)
            
            pygame.gfxdraw.filled_circle(self.screen, p[0], p[1], 8, (*color, alpha))
            pygame.gfxdraw.aacircle(self.screen, p[0], p[1], 8, (*color, 200))
        
        self._render_particles()

    def _render_ui(self):
        # --- UI Panel ---
        ui_bg = pygame.Rect(self.WIDTH - 170, 10, 160, 95)
        pygame.draw.rect(self.screen, (*self.COLOR_BG, 200), ui_bg, border_radius=10)
        pygame.draw.rect(self.screen, self.COLOR_WALL_ACCENT, ui_bg, width=2, border_radius=10)

        # --- Text Rendering Helper ---
        def draw_text(text, pos, font, color=self.COLOR_TEXT, shadow_color=self.COLOR_TEXT_SHADOW):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + 1, pos[1] + 1))
            self.screen.blit(text_surf, pos)

        # --- UI Content ---
        time_left = max(0, self.MAX_STEPS - self.steps)
        draw_text(f"Time: {time_left / 100:.1f}s", (self.WIDTH - 160, 20), self.font_small)
        draw_text(f"Crystals: {self.crystals_remaining}", (self.WIDTH - 160, 45), self.font_small)
        draw_text(f"Gems: {len(self.lit_gem_indices)} / {self.NUM_GEMS}", (self.WIDTH - 160, 70), self.font_small)

        # --- Game Over Message ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            draw_text(self.termination_reason, (self.WIDTH//2 - self.font_large.size(self.termination_reason)[0]//2, self.HEIGHT//2 - 50), self.font_large)
            msg = "Episode Finished"
            draw_text(msg, (self.WIDTH//2 - self.font_small.size(msg)[0]//2, self.HEIGHT//2), self.font_small)

    def _spawn_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(20, 40)
            radius = self.np_random.uniform(2, 5)
            self.particles.append(Particle(pygame.Vector2(pos), vel, life, color, radius))

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            pos = p.pos + p.vel
            life = p.life - 1
            vel = p.vel * 0.95 # Damping
            if life > 0:
                new_particles.append(Particle(pos, vel, life, p.color, p.radius))
        self.particles = new_particles

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p.life / 40.0))
            color = (*p.color, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), int(p.radius), color)

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    env.reset()
    
    running = True
    game_over_pause = 0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step the Environment ---
        # For human play, we only step on an actual action
        if any(action) or not env.auto_advance:
            obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated and game_over_pause == 0:
            game_over_pause = 150 # Frames to wait before auto-reset
        
        if game_over_pause > 0:
            game_over_pause -= 1
            if game_over_pause == 0:
                print(f"Game Over. Final Score: {info['score']}. Resetting.")
                env.reset()

        # --- Pygame-specific event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Manual reset.")
                env.reset()
        
        # --- Rendering ---
        # The observation is already the rendered screen
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # If you had a display window, you'd blit it here.
        # For this script, we'll just control the loop rate.
        env.clock.tick(60) # Limit human play to 60fps

    pygame.quit()