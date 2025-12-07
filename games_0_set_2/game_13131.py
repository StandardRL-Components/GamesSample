import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:09:09.226048
# Source Brief: brief_03131.md
# Brief Index: 3131
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a game where the player controls a 'boson',
    manipulating its size to affect gravity, firing 'anti-bosons' to
    terraform a circular board, and capturing control points while avoiding hazards.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a 'boson' to capture control points on a circular board. "
        "Fire 'anti-bosons' to terraform the area and toggle your size to manipulate gravity, all while avoiding deadly hazards."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to fire and shift to toggle your boson's size."
    )
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_BOARD = (25, 30, 45)
    COLOR_TERRA = (60, 65, 80)
    COLOR_BOSON = (0, 150, 255)
    COLOR_ANTIBOSON = (255, 80, 80)
    COLOR_HAZARD = (255, 180, 0)
    COLOR_CP_ACTIVE = (50, 255, 100)
    COLOR_CP_CAPTURED = (180, 180, 180)
    COLOR_TEXT = (220, 220, 220)

    # Game Parameters
    MAX_STEPS = 2000
    BOARD_RADIUS = 180
    BOSON_SPEED = 3.0
    BOSON_RADIUS_LARGE = 20
    BOSON_RADIUS_SMALL = 10
    ANTIBOSON_SPEED = 6.0
    ANTIBOSON_RADIUS = 5
    ANTIBOSON_COOLDOWN = 15 # steps
    TERRAFORM_RADIUS = 30
    HAZARD_RADIUS = 8
    HAZARD_INITIAL_SPEED = 0.5
    HAZARD_SPEED_INCREASE_INTERVAL = 200
    HAZARD_SPEED_INCREASE_AMOUNT = 0.05
    NUM_CONTROL_POINTS = 4
    NUM_HAZARDS = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Action and Observation Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_game_over = pygame.font.SysFont('Consolas', 50, bold=True)

        # Game State (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.board_center = pygame.math.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        
        self.boson_pos = pygame.math.Vector2(0, 0)
        self.boson_radius = 0
        self.boson_direction = pygame.math.Vector2(0, -1)
        self.is_boson_small = False
        
        self.anti_bosons = []
        self.terraformed_areas = []
        self.control_points = []
        self.hazards = []
        self.particles = []

        self.fire_cooldown_timer = 0
        self.prev_space_state = False
        self.prev_shift_state = False

        self.prev_dist_to_cp = float('inf')
        self.prev_dist_to_hazard = float('inf')
        
        # Initial call to reset to set up the first state
        # self.reset() is called by the wrapper/runner
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.boson_pos = self.board_center.copy()
        self.is_boson_small = False
        self.boson_radius = self.BOSON_RADIUS_LARGE
        self.boson_direction = pygame.math.Vector2(0, -1)
        
        self.anti_bosons.clear()
        self.terraformed_areas.clear()
        self.particles.clear()

        self.fire_cooldown_timer = 0
        self.prev_space_state = False
        self.prev_shift_state = False

        # Initialize Control Points
        self.control_points.clear()
        for i in range(self.NUM_CONTROL_POINTS):
            angle = 2 * math.pi * i / self.NUM_CONTROL_POINTS + (math.pi / 4)
            pos = self.board_center + pygame.math.Vector2(
                math.cos(angle) * self.BOARD_RADIUS * 0.75,
                math.sin(angle) * self.BOARD_RADIUS * 0.75
            )
            self.control_points.append({"pos": pos, "captured": False, "radius": 12})

        # Initialize Hazards
        self.hazards.clear()
        for _ in range(self.NUM_HAZARDS):
            pos = self.board_center + pygame.math.Vector2(
                random.uniform(-self.BOARD_RADIUS, self.BOARD_RADIUS),
                random.uniform(-self.BOARD_RADIUS, self.BOARD_RADIUS)
            )
            while pos.distance_to(self.boson_pos) < 100: # Don't spawn on player
                 pos = self.board_center + pygame.math.Vector2(
                    random.uniform(-self.BOARD_RADIUS, self.BOARD_RADIUS),
                    random.uniform(-self.BOARD_RADIUS, self.BOARD_RADIUS)
                )
            vel = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.HAZARD_INITIAL_SPEED
            self.hazards.append({"pos": pos, "vel": vel, "speed": self.HAZARD_INITIAL_SPEED})

        # Reset distance trackers for reward calculation
        self._update_dist_trackers()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        # --- 1. Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length() > 0:
            self.boson_direction = move_vec.normalize()
            self.boson_pos += self.boson_direction * self.BOSON_SPEED
            # Clamp position to board
            dist_from_center = self.boson_pos.distance_to(self.board_center)
            if dist_from_center > self.BOARD_RADIUS - self.boson_radius:
                self.boson_pos = self.board_center + (self.boson_pos - self.board_center).normalize() * (self.BOARD_RADIUS - self.boson_radius)

        # Fire Anti-Boson (on key press)
        if space_held and not self.prev_space_state and self.fire_cooldown_timer <= 0:
            # sfx: player_shoot.wav
            start_pos = self.boson_pos + self.boson_direction * (self.boson_radius + 5)
            self.anti_bosons.append({"pos": start_pos, "vel": self.boson_direction * self.ANTIBOSON_SPEED})
            self.fire_cooldown_timer = self.ANTIBOSON_COOLDOWN
        
        # Toggle Boson Size (on key press)
        if shift_held and not self.prev_shift_state:
            # sfx: player_resize.wav
            self.is_boson_small = not self.is_boson_small
            self.boson_radius = self.BOSON_RADIUS_SMALL if self.is_boson_small else self.BOSON_RADIUS_LARGE
            self._create_particles(self.boson_pos, 20, self.COLOR_BOSON, 2, 30)

        self.prev_space_state = space_held
        self.prev_shift_state = shift_held
        if self.fire_cooldown_timer > 0:
            self.fire_cooldown_timer -= 1

        # --- 2. Update Game State ---
        self._update_hazards()
        self._update_anti_bosons()
        self._update_particles()
        
        # --- 3. Check Interactions & Assign Rewards ---
        # Continuous rewards
        dist_cp, dist_hz = self._get_closest_distances()
        if dist_cp < self.prev_dist_to_cp: reward += 0.1
        if dist_hz < self.prev_dist_to_hazard: reward -= 0.5
        self.prev_dist_to_cp = dist_cp
        self.prev_dist_to_hazard = dist_hz

        # Boson-Hazard collision
        for hazard in self.hazards:
            if self.boson_pos.distance_to(hazard["pos"]) < self.boson_radius + self.HAZARD_RADIUS:
                # sfx: player_death.wav
                self.game_over = True
                reward += -10
                self._create_particles(self.boson_pos, 50, self.COLOR_BOSON, 4, 60)
                break
        
        # Control Point capture
        for cp in self.control_points:
            if not cp["captured"] and self.boson_pos.distance_to(cp["pos"]) < self.boson_radius + cp["radius"]:
                # sfx: capture_point.wav
                cp["captured"] = True
                reward += 5
                self.score += 1
                self._create_particles(cp["pos"], 30, self.COLOR_CP_ACTIVE, 3, 45)

        # --- 4. Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.game_over:
            terminated = True
        
        if all(cp["captured"] for cp in self.control_points):
            # sfx: victory.wav
            terminated = True
            self.game_over = True # To show game over screen
            reward += 100
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            terminated = True # Per Gymnasium API, terminated should be True if truncated is True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_dist_trackers(self):
        self.prev_dist_to_cp, self.prev_dist_to_hazard = self._get_closest_distances()

    def _get_closest_distances(self):
        # Closest uncaptured control point
        min_dist_cp = float('inf')
        uncaptured_cps = [cp for cp in self.control_points if not cp['captured']]
        if uncaptured_cps:
            min_dist_cp = min(self.boson_pos.distance_to(cp['pos']) for cp in uncaptured_cps)
        
        # Closest hazard
        min_dist_hz = float('inf')
        if self.hazards:
            min_dist_hz = min(self.boson_pos.distance_to(h['pos']) for h in self.hazards)
            
        return min_dist_cp, min_dist_hz

    def _update_hazards(self):
        # Difficulty scaling
        if self.steps > 0 and self.steps % self.HAZARD_SPEED_INCREASE_INTERVAL == 0:
            for h in self.hazards:
                h["speed"] += self.HAZARD_SPEED_INCREASE_AMOUNT
                h["vel"] = h["vel"].normalize() * h["speed"]

        for h in self.hazards:
            h["pos"] += h["vel"]
            dist_from_center = h["pos"].distance_to(self.board_center)
            if dist_from_center > self.BOARD_RADIUS - self.HAZARD_RADIUS:
                # Bounce off board edge
                normal = (h["pos"] - self.board_center).normalize()
                h["vel"] = h["vel"].reflect(normal)
                # Ensure it's inside after bounce
                h["pos"] = self.board_center + normal * (self.BOARD_RADIUS - self.HAZARD_RADIUS)

    def _update_anti_bosons(self):
        for ab in self.anti_bosons[:]:
            ab["pos"] += ab["vel"]
            # Check for terraforming impact
            if ab["pos"].distance_to(self.board_center) <= self.BOARD_RADIUS:
                if not any(ab["pos"].distance_to(pygame.math.Vector2(ta['x'], ta['y'])) < self.TERRAFORM_RADIUS for ta in self.terraformed_areas):
                    # sfx: terraform_impact.wav
                    self.terraformed_areas.append({'x': ab['pos'].x, 'y': ab['pos'].y, 'radius': self.TERRAFORM_RADIUS})
                    self._create_particles(ab["pos"], 15, self.COLOR_TERRA, 1.5, 25)
                    self.anti_bosons.remove(ab)
            # Remove if off-screen
            elif not self.screen.get_rect().collidepoint(ab["pos"].x, ab["pos"].y):
                self.anti_bosons.remove(ab)
    
    def _create_particles(self, pos, count, color, speed, lifetime):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(0.5, speed)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifetime": random.randint(lifetime // 2, lifetime),
                "max_lifetime": lifetime,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Draw main board
        pygame.gfxdraw.filled_circle(self.screen, int(self.board_center.x), int(self.board_center.y), self.BOARD_RADIUS, self.COLOR_BOARD)
        pygame.gfxdraw.aacircle(self.screen, int(self.board_center.x), int(self.board_center.y), self.BOARD_RADIUS, self.COLOR_BOARD)
        
        # Draw terraformed areas
        for ta in self.terraformed_areas:
            pygame.gfxdraw.filled_circle(self.screen, int(ta['x']), int(ta['y']), int(ta['radius']), self.COLOR_TERRA)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / p["max_lifetime"]))
            color = (*p["color"], alpha)
            size = int(3 * (p["lifetime"] / p["max_lifetime"]))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), size)

        # Draw Control Points
        for cp in self.control_points:
            color = self.COLOR_CP_CAPTURED if cp["captured"] else self.COLOR_CP_ACTIVE
            pos = (int(cp['pos'].x), int(cp['pos'].y))
            # Glow effect
            for i in range(4, 0, -1):
                glow_color = (*color, 60 // i)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], cp['radius'] + i*2, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], cp['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], cp['radius'], color)

        # Draw Hazards
        for h in self.hazards:
            pos = (int(h['pos'].x), int(h['pos'].y))
            # Glow effect
            for i in range(3, 0, -1):
                glow_color = (*self.COLOR_HAZARD, 80 // i)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.HAZARD_RADIUS + i*2, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.HAZARD_RADIUS, self.COLOR_HAZARD)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.HAZARD_RADIUS, self.COLOR_HAZARD)

        # Draw Anti-Bosons
        for ab in self.anti_bosons:
            pos = (int(ab['pos'].x), int(ab['pos'].y))
            # Trail effect
            for i in range(3):
                trail_pos = ab['pos'] - ab['vel'].normalize() * (i + 1) * 3
                alpha = 150 - i * 50
                pygame.gfxdraw.filled_circle(self.screen, int(trail_pos.x), int(trail_pos.y), self.ANTIBOSON_RADIUS - i, (*self.COLOR_ANTIBOSON, alpha))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ANTIBOSON_RADIUS, self.COLOR_ANTIBOSON)

        # Draw Boson (if not dead)
        if not self.game_over or all(cp["captured"] for cp in self.control_points):
            pos = (int(self.boson_pos.x), int(self.boson_pos.y))
            
            # Gravity field visualization
            gravity_strength = 1.0 if self.is_boson_small else 0.5
            num_rings = int(4 * gravity_strength)
            for i in range(num_rings):
                alpha = int(80 * gravity_strength * (1 - i / num_rings))
                radius = self.boson_radius + 15 + i * 15
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*self.COLOR_BOSON, alpha))

            # Glow effect
            for i in range(5, 0, -1):
                glow_color = (*self.COLOR_BOSON, 100 // i)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.boson_radius + i*2, glow_color)

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.boson_radius, self.COLOR_BOSON)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.boson_radius, self.COLOR_BOSON)

            # Direction indicator
            dir_end = self.boson_pos + self.boson_direction * (self.boson_radius)
            pygame.draw.line(self.screen, self.COLOR_BG, pos, (int(dir_end.x), int(dir_end.y)), 2)
            
    def _render_ui(self):
        # Score and Upgrade Level
        score_text = self.font_ui.render(f"CONTROL POINTS: {self.score}/{self.NUM_CONTROL_POINTS}", True, self.COLOR_TEXT)
        size_mode = "SMALL (HIGH-G)" if self.is_boson_small else "LARGE (LOW-G)"
        upgrade_text = self.font_ui.render(f"BOSON MODE: {size_mode}", True, self.COLOR_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(upgrade_text, (10, 35))

        # Game Over Message
        if self.game_over:
            if all(cp["captured"] for cp in self.control_points):
                msg = "VICTORY"
                color = self.COLOR_CP_ACTIVE
            else:
                msg = "FAILURE"
                color = self.COLOR_ANTIBOSON
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            game_over_surf = self.font_game_over.render(msg, True, color)
            text_rect = game_over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_surf, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the evaluation environment
    os.environ["SDL_VIDEODRIVER"] = "pygame"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Boson Control")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    running = True
    
    while running:
        # --- Human Input Processing ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False

        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()