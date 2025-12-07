import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:30:26.596743
# Source Brief: brief_01136.md
# Brief Index: 1136
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Light Beam Reflector
    
    A real-time puzzle game where the player places and rotates mirrors to guide a
    light beam. The objective is to charge all 5 energy cells by hitting them with
    the beam before the 30-second timer runs out. The game features visually
    appealing light effects and chain reactions.
    
    Action Space: MultiDiscrete([5, 2, 2])
    - action[0] (Movement):
        - 0: None
        - 1: Up
        - 2: Down
        - 3: Left
        - 4: Right
        If Shift is held, Left/Right rotates the last placed mirror.
        Otherwise, moves the player's placer cursor.
    - action[1] (Spacebar):
        - 0: Released
        - 1: Held
        On press (transition from 0 to 1):
        If Shift is held, removes the nearest mirror.
        Otherwise, places a new mirror at the placer's location.
    - action[2] (Shift):
        - 0: Released
        - 1: Held
        Modifies the behavior of Movement and Spacebar actions.
    
    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    An RGB image of the game screen.
    
    Reward Structure:
    - +100 for winning (charging all cells).
    - -100 for losing (timer runs out).
    - +1.0 for each cell fully charged.
    - +0.01 for each beam reflection (chain reaction).
    - A small continuous reward for increasing the total charge across all cells.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A real-time puzzle game where the player places and rotates mirrors to guide a "
        "light beam to charge all energy cells before the timer runs out."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to place a mirror. "
        "Hold shift and use ←→ to rotate a mirror, or press space to remove one."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TIME_LIMIT_SECONDS = 30
    FRAME_RATE = 30 # For physics and timer updates
    MAX_STEPS = TIME_LIMIT_SECONDS * FRAME_RATE

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_PLACER = (255, 255, 0, 200)
    COLOR_BEAM = (255, 255, 180)
    COLOR_BEAM_CORE = (255, 255, 255)
    COLOR_MIRROR = (200, 200, 220)
    COLOR_MIRROR_SELECTED = (100, 255, 255)
    COLOR_CELL_UNCHARGED = (70, 80, 100)
    COLOR_CELL_CHARGED = (0, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    
    # Game Parameters
    PLACER_SPEED = 7
    MIRROR_LENGTH = 50
    MIRROR_THICKNESS = 4
    MIRROR_ROTATION_SPEED = 4 # degrees per step
    MAX_MIRRORS = 7
    MAX_BEAM_BOUNCES = 15
    CELL_RADIUS = 20
    CELL_CHARGE_RATE = 0.01
    
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
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # State variables are initialized in reset()
        self.placer_pos = None
        self.mirrors = None
        self.cells = None
        self.beam_source = None
        self.beam_segments = None
        self.particles = None
        self.steps = None
        self.score = None
        self.time_remaining = None
        self.game_over = None
        self.win_condition = None
        self.prev_space_held = None
        self.last_selected_mirror_idx = None
        self.total_charge_last_step = None
        
        # Call reset to initialize the state
        # self.reset() # This is typically called by the environment runner, not in __init__

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.placer_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.mirrors = []
        
        # Initialize cells in fixed positions
        self.cells = [
            {"pos": pygame.math.Vector2(100, 100), "charge": 0.0, "is_charged": False},
            {"pos": pygame.math.Vector2(540, 300), "charge": 0.0, "is_charged": False},
            {"pos": pygame.math.Vector2(320, 350), "charge": 0.0, "is_charged": False},
            {"pos": pygame.math.Vector2(500, 80), "charge": 0.0, "is_charged": False},
            {"pos": pygame.math.Vector2(150, 250), "charge": 0.0, "is_charged": False},
        ]
        
        self.beam_source = pygame.math.Vector2(20, self.SCREEN_HEIGHT / 2)
        self.beam_segments = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.game_over = False
        self.win_condition = False
        
        self.prev_space_held = False
        self.last_selected_mirror_idx = -1
        self.total_charge_last_step = 0

        # Initial beam calculation
        self._update_beam()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.time_remaining -= 1 / self.FRAME_RATE

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        self._handle_input(movement, space_pressed, shift_held)

        # --- Update Game Logic ---
        beam_recalculated = self._update_beam()
        reward += self._update_cells()
        self._update_particles()
        
        # Add small reward for reflections (chain reactions)
        if beam_recalculated:
            reward += 0.01 * (len(self.beam_segments) - 1)

        # --- Check Termination ---
        num_charged_cells = sum(1 for cell in self.cells if cell["is_charged"])
        if num_charged_cells == len(self.cells):
            self.win_condition = True
            self.game_over = True
            reward += 100.0 # Win bonus
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            self.win_condition = False
            self.game_over = True
            reward -= 100.0 # Loss penalty

        self.score += reward
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_held):
        # Determine which mirror is selected (closest to placer)
        self.last_selected_mirror_idx = -1
        if self.mirrors:
            closest_dist = float('inf')
            for i, mirror in enumerate(self.mirrors):
                dist = self.placer_pos.distance_to(mirror["pos"])
                if dist < closest_dist and dist < self.MIRROR_LENGTH / 2 + 10:
                    closest_dist = dist
                    self.last_selected_mirror_idx = i

        if shift_held:
            # --- SHIFT-MODIFIED ACTIONS ---
            if self.last_selected_mirror_idx != -1:
                # Rotate selected mirror
                if movement == 3: # Left
                    self.mirrors[self.last_selected_mirror_idx]["angle"] -= self.MIRROR_ROTATION_SPEED
                elif movement == 4: # Right
                    self.mirrors[self.last_selected_mirror_idx]["angle"] += self.MIRROR_ROTATION_SPEED
            
            # Remove mirror
            if space_pressed and self.last_selected_mirror_idx != -1:
                self.mirrors.pop(self.last_selected_mirror_idx)
                self.last_selected_mirror_idx = -1
                # sfx: mirror_remove
        else:
            # --- NORMAL ACTIONS ---
            # Move placer
            if movement == 1: self.placer_pos.y -= self.PLACER_SPEED
            if movement == 2: self.placer_pos.y += self.PLACER_SPEED
            if movement == 3: self.placer_pos.x -= self.PLACER_SPEED
            if movement == 4: self.placer_pos.x += self.PLACER_SPEED
            
            # Clamp placer position to screen bounds
            self.placer_pos.x = np.clip(self.placer_pos.x, 0, self.SCREEN_WIDTH)
            self.placer_pos.y = np.clip(self.placer_pos.y, 0, self.SCREEN_HEIGHT)
            
            # Place mirror
            if space_pressed and len(self.mirrors) < self.MAX_MIRRORS:
                new_mirror = {
                    "pos": self.placer_pos.copy(),
                    "angle": 45, # Default angle
                }
                self.mirrors.append(new_mirror)
                self.last_selected_mirror_idx = len(self.mirrors) - 1
                # sfx: mirror_place

    def _update_beam(self):
        self.beam_segments = []
        self._trace_beam(self.beam_source, pygame.math.Vector2(1, 0), self.MAX_BEAM_BOUNCES)
        return True

    def _trace_beam(self, start_point, direction, bounces_left):
        if bounces_left <= 0:
            return

        direction.normalize_ip()
        closest_hit = None
        min_dist = float('inf')

        # Check for intersections with mirrors
        for i, mirror in enumerate(self.mirrors):
            p1 = mirror["pos"] + pygame.math.Vector2(self.MIRROR_LENGTH/2, 0).rotate(mirror["angle"])
            p2 = mirror["pos"] - pygame.math.Vector2(self.MIRROR_LENGTH/2, 0).rotate(mirror["angle"])
            
            hit_point = self._get_line_segment_intersection(start_point, direction, p1, p2)
            
            if hit_point:
                dist = start_point.distance_to(hit_point)
                if 0.01 < dist < min_dist:
                    min_dist = dist
                    normal = (p2 - p1).rotate(90).normalize()
                    closest_hit = {"type": "mirror", "point": hit_point, "normal": normal}

        # Check for intersections with screen boundaries
        # Top
        if direction.y < 0:
            t = (0 - start_point.y) / direction.y
            if t > 0.01 and t < min_dist: min_dist = t; closest_hit = {"type": "boundary", "point": start_point + t * direction}
        # Bottom
        if direction.y > 0:
            t = (self.SCREEN_HEIGHT - start_point.y) / direction.y
            if t > 0.01 and t < min_dist: min_dist = t; closest_hit = {"type": "boundary", "point": start_point + t * direction}
        # Left
        if direction.x < 0:
            t = (0 - start_point.x) / direction.x
            if t > 0.01 and t < min_dist: min_dist = t; closest_hit = {"type": "boundary", "point": start_point + t * direction}
        # Right
        if direction.x > 0:
            t = (self.SCREEN_WIDTH - start_point.x) / direction.x
            if t > 0.01 and t < min_dist: min_dist = t; closest_hit = {"type": "boundary", "point": start_point + t * direction}

        if closest_hit:
            end_point = closest_hit["point"]
            self.beam_segments.append((start_point, end_point))
            
            if closest_hit["type"] == "mirror":
                # sfx: beam_reflect
                reflected_dir = direction.reflect(closest_hit["normal"])
                self._trace_beam(end_point, reflected_dir, bounces_left - 1)
        else: # No hit, beam goes off-screen
            end_point = start_point + direction * (self.SCREEN_WIDTH * 2)
            self.beam_segments.append((start_point, end_point))

    def _update_cells(self):
        reward = 0
        current_total_charge = 0
        
        for cell in self.cells:
            if cell["is_charged"]:
                current_total_charge += 1.0
                continue
            
            is_hit = False
            for p1, p2 in self.beam_segments:
                # Check line segment-circle intersection
                d = p2 - p1
                if d.length() == 0: continue
                
                f = p1 - cell["pos"]
                a = d.dot(d)
                b = 2 * f.dot(d)
                c = f.dot(f) - self.CELL_RADIUS**2
                discriminant = b**2 - 4*a*c
                
                if discriminant >= 0:
                    discriminant = math.sqrt(discriminant)
                    t1 = (-b - discriminant) / (2*a)
                    t2 = (-b + discriminant) / (2*a)
                    if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                        is_hit = True
                        break
            
            if is_hit:
                # sfx: cell_charging
                cell["charge"] = min(1.0, cell["charge"] + self.CELL_CHARGE_RATE)
                if cell["charge"] >= 1.0:
                    cell["is_charged"] = True
                    reward += 1.0 # Reward for charging a cell
                    # sfx: cell_charged_fully
                    # Spawn burst of particles
                    for _ in range(30):
                        angle = random.uniform(0, 360)
                        speed = random.uniform(1, 4)
                        vel = pygame.math.Vector2(speed, 0).rotate(angle)
                        self.particles.append({
                            "pos": cell["pos"].copy(), "vel": vel,
                            "life": random.randint(20, 40), "color": self.COLOR_CELL_CHARGED
                        })
                else:
                    # Spawn a single particle while charging
                    angle = random.uniform(0, 360)
                    vel = pygame.math.Vector2(1, 0).rotate(angle)
                    self.particles.append({
                        "pos": cell["pos"] + vel * self.CELL_RADIUS, "vel": vel * 0.5,
                        "life": 15, "color": self.COLOR_CELL_CHARGED
                    })

            current_total_charge += cell["charge"]
            
        # Continuous reward for increasing total charge
        charge_delta = current_total_charge - self.total_charge_last_step
        if charge_delta > 0:
            reward += charge_delta * 0.1
        self.total_charge_last_step = current_total_charge
        
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Cells
        for cell in self.cells:
            pos = (int(cell["pos"].x), int(cell["pos"].y))
            # Glow effect
            for i in range(10, 0, -2):
                alpha = 80 * (1 - i / 10)
                if cell["is_charged"]:
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CELL_RADIUS + i, (*self.COLOR_CELL_CHARGED, int(alpha)))
            # Base circle
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.CELL_RADIUS, self.COLOR_CELL_UNCHARGED)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.CELL_RADIUS, self.COLOR_CELL_UNCHARGED)
            # Charge fill
            if cell["charge"] > 0:
                fill_radius = int(self.CELL_RADIUS * math.sqrt(cell["charge"]))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], fill_radius, self.COLOR_CELL_CHARGED)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], fill_radius, self.COLOR_CELL_CHARGED)

        # Render Beam
        for p1, p2 in self.beam_segments:
            pygame.draw.aaline(self.screen, self.COLOR_BEAM, p1, p2, 3)
            pygame.draw.aaline(self.screen, self.COLOR_BEAM_CORE, p1, p2, 1)

        # Render Mirrors
        for i, mirror in enumerate(self.mirrors):
            color = self.COLOR_MIRROR_SELECTED if i == self.last_selected_mirror_idx else self.COLOR_MIRROR
            half_len = pygame.math.Vector2(self.MIRROR_LENGTH/2, 0).rotate(mirror["angle"])
            p1 = mirror["pos"] - half_len
            p2 = mirror["pos"] + half_len
            pygame.draw.line(self.screen, color, p1, p2, self.MIRROR_THICKNESS)

        # Render Particles
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / 40))
            color = (*p["color"], alpha)
            pos = (int(p["pos"].x), int(p["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["life"] / 10), color)

        # Render Placer
        if not self.game_over:
            placer_pos_int = (int(self.placer_pos.x), int(self.placer_pos.y))
            size = 10 + 3 * math.sin(self.steps * 0.2)
            pygame.draw.line(self.screen, self.COLOR_PLACER, (placer_pos_int[0] - size, placer_pos_int[1]), (placer_pos_int[0] + size, placer_pos_int[1]), 1)
            pygame.draw.line(self.screen, self.COLOR_PLACER, (placer_pos_int[0], placer_pos_int[1] - size), (placer_pos_int[0], placer_pos_int[1] + size), 1)

    def _render_ui(self):
        # Render cell count
        charged_count = sum(1 for cell in self.cells if cell["is_charged"])
        cell_text = self.font_ui.render(f"Cells: {charged_count}/{len(self.cells)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(cell_text, (10, 10))
        
        # Render timer
        timer_text = self.font_ui.render(f"Time: {max(0, self.time_remaining):.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Render Game Over screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_condition:
                end_text = self.font_game_over.render("SUCCESS", True, self.COLOR_CELL_CHARGED)
            else:
                end_text = self.font_game_over.render("TIME OUT", True, (255, 50, 50))
                
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "mirrors_placed": len(self.mirrors),
            "cells_charged": sum(1 for cell in self.cells if cell["is_charged"]),
        }

    def _get_line_segment_intersection(self, p0, d0, p1, p2):
        d1 = p2 - p1
        if d1.cross(d0) == 0: return None # Parallel
        t = d1.cross(p0 - p1) / d1.cross(d0)
        u = -d0.cross(p1 - p0) / d0.cross(d1)
        if 0 < t and 0 <= u <= 1:
            return p0 + t * d0
        return None

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This part of the script will not run in the test environment,
    # but is useful for local testing. We need to unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    # --- Human Playable Demo ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Light Beam Reflector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Action Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered surface, so we just need to display it.
        # We need to transpose it back for pygame's display format.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart.")

        clock.tick(GameEnv.FRAME_RATE)
        
    env.close()