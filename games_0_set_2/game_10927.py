import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:19:49.936878
# Source Brief: brief_00927.md
# Brief Index: 927
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment simulating a cellular environment where the agent
    manipulates colored vesicles to trigger chain reactions and fulfill
    transport requests. The agent controls a cursor and can freeze time to
    strategically pick up and place vesicles.

    **Visuals:**
    - A dark, organic background representing the inside of a cell.
    - Brightly colored, glowing vesicles that move around.
    - Color-coded transport docks that need to be filled.
    - Particle effects and expanding shockwaves for chain reactions.
    - A clear UI showing score, time, and remaining requests.

    **Gameplay:**
    - The agent moves a cursor around the screen.
    - The 'space' button toggles a time-freeze mechanic.
    - While time is frozen, the 'shift' button can be used to pick up a
      vesicle under the cursor, and press it again to drop it.
    - Dropping a vesicle unfreezes time.
    - Vesicles of the same color that collide trigger a chain reaction,
      destroying nearby vesicles of the same color for points.
    - Dropping a vesicle of the correct color onto a transport dock fulfills
      that request, granting a large reward.
    - The goal is to fulfill all transport requests before time runs out.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]`: Cursor Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - `action[1]`: Time Freeze Toggle (1: 'space' is held, 0: released)
    - `action[2]`: Select/Drop (1: 'shift' is held, 0: released)

    **Reward Structure:**
    - +0.1 for each vesicle destroyed in a chain reaction.
    - +1.0 for fulfilling a transport request.
    - -0.01 per step while time is frozen (encourages quick decisions).
    - +100 for winning the game (all requests fulfilled).
    - -100 for losing the game (time runs out).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manipulate colored vesicles in a cellular environment. Freeze time to strategically place vesicles, "
        "trigger chain reactions, and fulfill transport requests before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press 'space' to toggle time freeze. "
        "While frozen, press 'shift' to pick up or drop a vesicle."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 3600  # 60 seconds * 60 FPS

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_WALL = (30, 60, 100)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    VESICLE_COLORS = {
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "yellow": (255, 255, 0),
        "green": (0, 255, 0),
    }

    # Game Parameters
    CURSOR_SPEED = 6
    VESICLE_RADIUS = 10
    VESICLE_SPEED_MIN = 0.5
    VESICLE_SPEED_MAX = 1.5
    NUM_VESICLES = 25
    NUM_REQUESTS = 3
    REQUEST_RADIUS = 18
    SHOCKWAVE_SPEED = 4
    SHOCKWAVE_MAX_RADIUS = 80
    PARTICLE_LIFESPAN = 30

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_score = pygame.font.SysFont("Consolas", 28, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_frozen = False
        self.cursor_pos = None
        self.vesicles = []
        self.transport_requests = []
        self.held_vesicle_idx = None
        self.shockwaves = []
        self.particles = []
        self.last_space_state = 0
        self.last_shift_state = 0
        self.cell_wall_points = []
        
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_frozen = False
        
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.held_vesicle_idx = None
        
        self.vesicles = []
        self.shockwaves = []
        self.particles = []
        
        self.last_space_state = 0
        self.last_shift_state = 0

        # Use self.np_random for reproducibility
        self.np_random_std = np.random.default_rng(seed)

        # Generate cell wall shape
        self.cell_wall_points = self._generate_cell_wall()
        
        # Generate transport requests
        self.transport_requests = []
        request_colors = self.np_random.choice(list(self.VESICLE_COLORS.keys()), self.NUM_REQUESTS, replace=False)
        for i in range(self.NUM_REQUESTS):
            angle = (i / self.NUM_REQUESTS) * 2 * math.pi
            pos = np.array([
                self.SCREEN_WIDTH / 2 + (self.SCREEN_WIDTH / 2 - 50) * math.cos(angle),
                self.SCREEN_HEIGHT / 2 + (self.SCREEN_HEIGHT / 2 - 50) * math.sin(angle)
            ])
            self.transport_requests.append({
                "pos": pos,
                "color_name": request_colors[i],
                "fulfilled": False
            })

        # Generate vesicles
        for _ in range(self.NUM_VESICLES):
            self._spawn_vesicle()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- Handle Input and State Changes ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_state
        shift_pressed = shift_held and not self.last_shift_state
        
        self._move_cursor(movement)

        if space_pressed:
            self.time_frozen = not self.time_frozen
            # sound: time_freeze_toggle.wav

        if self.time_frozen:
            reward -= 0.01  # Penalty for freezing time
            if shift_pressed:
                self._handle_pickup_drop()
        
        # --- Update Game Logic (if not frozen) ---
        if not self.time_frozen:
            reward += self._update_game_state()

        self.steps += 1
        
        # --- Check for Termination ---
        terminated = self._check_termination()
        if terminated:
            if all(req["fulfilled"] for req in self.transport_requests):
                reward += 100 # Win condition
                # sound: game_win.wav
            else:
                reward -= 100 # Lose condition (time out)
                # sound: game_over.wav
        
        # Update last button states
        self.last_space_state = space_held
        self.last_shift_state = shift_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self):
        """Processes one frame of game logic when time is not frozen."""
        step_reward = 0
        
        # Update and move vesicles
        for v in self.vesicles:
            v["pos"] += v["vel"]
            self._handle_wall_bounce(v)

        # Update held vesicle position to cursor
        if self.held_vesicle_idx is not None:
            self.vesicles[self.held_vesicle_idx]["pos"] = self.cursor_pos.copy()

        # Update shockwaves
        for sw in self.shockwaves:
            sw["radius"] += self.SHOCKWAVE_SPEED

        # Check for collisions and process events
        step_reward += self._handle_collisions()

        # Update particles
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        
        # Clean up dead entities
        self.vesicles = [v for v in self.vesicles if v.get("alive", True)]
        self.shockwaves = [sw for sw in self.shockwaves if sw["radius"] < self.SHOCKWAVE_MAX_RADIUS]
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

        return step_reward

    def _handle_collisions(self):
        """Checks and resolves all collisions, returning any generated rewards."""
        collision_reward = 0
        
        # Vesicle-Vesicle collisions (chain reactions)
        for i in range(len(self.vesicles)):
            for j in range(i + 1, len(self.vesicles)):
                v1 = self.vesicles[i]
                v2 = self.vesicles[j]
                
                if not v1.get("alive", True) or not v2.get("alive", True):
                    continue

                dist_sq = np.sum((v1["pos"] - v2["pos"])**2)
                if dist_sq < (self.VESICLE_RADIUS * 2)**2:
                    if v1["color_name"] == v2["color_name"]:
                        # Trigger chain reaction
                        v1["alive"] = False
                        v2["alive"] = False
                        collision_reward += 0.2  # +0.1 for each of the two initial vesicles
                        self._create_shockwave( (v1["pos"] + v2["pos"]) / 2, v1["color_name"])
                        # sound: chain_reaction_start.wav

        # Shockwave-Vesicle collisions
        for sw in self.shockwaves:
            for v in self.vesicles:
                if not v.get("alive", True): continue
                if v["color_name"] == sw["color_name"]:
                    dist_sq = np.sum((v["pos"] - sw["pos"])**2)
                    if dist_sq < (sw["radius"] + self.VESICLE_RADIUS)**2:
                        v["alive"] = False
                        collision_reward += 0.1
                        self._create_particles(v["pos"], v["color_name"], 10)
                        # sound: vesicle_pop.wav

        # Vesicle-Transport Request collisions
        if self.held_vesicle_idx is None: # Only non-held vesicles can be delivered
            for v in self.vesicles:
                if not v.get("alive", True): continue
                for req in self.transport_requests:
                    if not req["fulfilled"] and v["color_name"] == req["color_name"]:
                        dist_sq = np.sum((v["pos"] - req["pos"])**2)
                        if dist_sq < self.REQUEST_RADIUS**2:
                            req["fulfilled"] = True
                            v["alive"] = False
                            collision_reward += 1.0
                            self._create_particles(req["pos"], req["color_name"], 30, speed_mult=2)
                            # sound: request_fulfilled.wav
        
        return collision_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw cell wall
        pygame.draw.aalines(self.screen, self.COLOR_WALL, True, self.cell_wall_points, 1)

        # Draw transport requests
        for req in self.transport_requests:
            color = self.VESICLE_COLORS[req["color_name"]]
            if req["fulfilled"]:
                # Draw a solid, bright circle when fulfilled
                pygame.gfxdraw.filled_circle(self.screen, int(req["pos"][0]), int(req["pos"][1]), self.REQUEST_RADIUS, color)
                pygame.gfxdraw.aacircle(self.screen, int(req["pos"][0]), int(req["pos"][1]), self.REQUEST_RADIUS, color)
            else:
                # Draw a dashed outline when not fulfilled
                self._draw_dashed_circle(self.screen, color, req["pos"], self.REQUEST_RADIUS, width=2)

        # Draw shockwaves
        for sw in self.shockwaves:
            color = self.VESICLE_COLORS[sw["color_name"]]
            alpha = int(255 * (1 - sw["radius"] / self.SHOCKWAVE_MAX_RADIUS))
            if alpha > 0:
                self._draw_glowing_circle(self.screen, sw["pos"], int(sw["radius"]), color, alpha, glow_size=1.2)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / self.PARTICLE_LIFESPAN))
            if alpha > 0:
                color = (*self.VESICLE_COLORS[p["color_name"]], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), p["radius"], color)
        
        # Draw vesicles
        for i, v in enumerate(self.vesicles):
            if i == self.held_vesicle_idx: continue # Draw held one last
            color = self.VESICLE_COLORS[v["color_name"]]
            self._draw_glowing_circle(self.screen, v["pos"], self.VESICLE_RADIUS, color)

        # Draw cursor and held vesicle
        cursor_color = self.COLOR_CURSOR
        if self.held_vesicle_idx is not None:
            v = self.vesicles[self.held_vesicle_idx]
            color = self.VESICLE_COLORS[v["color_name"]]
            self._draw_glowing_circle(self.screen, self.cursor_pos, self.VESICLE_RADIUS, color)
            cursor_color = color # Cursor takes on color of held item
        
        # Draw cursor lines
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        pygame.draw.line(self.screen, cursor_color, (x - 8, y), (x + 8, y), 1)
        pygame.draw.line(self.screen, cursor_color, (x, y - 8), (x, y + 8), 1)

    def _render_ui(self):
        # Time remaining
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {time_left:.1f}"
        time_surf = self.font_main.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        
        # Requests left
        requests_left = sum(1 for r in self.transport_requests if not r["fulfilled"])
        req_text = f"REQUESTS: {requests_left}/{self.NUM_REQUESTS}"
        req_surf = self.font_main.render(req_text, True, self.COLOR_TEXT)
        self.screen.blit(req_surf, (10, 10))

        # Score
        score_text = f"{self.score:.1f}"
        score_surf = self.font_score.render(score_text, True, self.COLOR_TEXT)
        score_pos = (self.SCREEN_WIDTH / 2 - score_surf.get_width() / 2, self.SCREEN_HEIGHT - score_surf.get_height() - 5)
        self.screen.blit(score_surf, score_pos)

        # Frozen indicator
        if self.time_frozen:
            frozen_text = "FROZEN"
            frozen_surf = self.font_main.render(frozen_text, True, self.VESICLE_COLORS["cyan"])
            frozen_pos = (self.SCREEN_WIDTH / 2 - frozen_surf.get_width() / 2, 10)
            self.screen.blit(frozen_surf, frozen_pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "requests_fulfilled": sum(1 for r in self.transport_requests if r["fulfilled"]),
            "time_frozen": self.time_frozen
        }

    # --- Helper Functions ---

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED  # Up
        if movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED  # Down
        if movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED  # Left
        if movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

    def _handle_pickup_drop(self):
        if self.held_vesicle_idx is None:
            # Attempt to pick up
            for i, v in enumerate(self.vesicles):
                dist_sq = np.sum((self.cursor_pos - v["pos"])**2)
                if dist_sq < self.VESICLE_RADIUS**2:
                    self.held_vesicle_idx = i
                    # sound: pickup.wav
                    break
        else:
            # Drop the vesicle
            self.vesicles[self.held_vesicle_idx]["pos"] = self.cursor_pos.copy()
            self.held_vesicle_idx = None
            self.time_frozen = False # Dropping unfreezes time
            # sound: drop.wav

    def _handle_wall_bounce(self, vesicle):
        # A simple Axis-Aligned Bounding Box bounce for performance
        if vesicle["pos"][0] < self.VESICLE_RADIUS or vesicle["pos"][0] > self.SCREEN_WIDTH - self.VESICLE_RADIUS:
            vesicle["vel"][0] *= -1
            vesicle["pos"][0] = np.clip(vesicle["pos"][0], self.VESICLE_RADIUS, self.SCREEN_WIDTH - self.VESICLE_RADIUS)
        if vesicle["pos"][1] < self.VESICLE_RADIUS or vesicle["pos"][1] > self.SCREEN_HEIGHT - self.VESICLE_RADIUS:
            vesicle["vel"][1] *= -1
            vesicle["pos"][1] = np.clip(vesicle["pos"][1], self.VESICLE_RADIUS, self.SCREEN_HEIGHT - self.VESICLE_RADIUS)

    def _spawn_vesicle(self):
        pos = np.array([
            self.np_random.uniform(self.VESICLE_RADIUS, self.SCREEN_WIDTH - self.VESICLE_RADIUS),
            self.np_random.uniform(self.VESICLE_RADIUS, self.SCREEN_HEIGHT - self.VESICLE_RADIUS)
        ])
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(self.VESICLE_SPEED_MIN, self.VESICLE_SPEED_MAX)
        vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        color_name = self.np_random.choice(list(self.VESICLE_COLORS.keys()))
        
        self.vesicles.append({
            "pos": pos, "vel": vel, "color_name": color_name, "alive": True
        })

    def _create_shockwave(self, pos, color_name):
        self.shockwaves.append({
            "pos": pos,
            "radius": self.VESICLE_RADIUS,
            "color_name": color_name
        })
        self._create_particles(pos, color_name, 20)

    def _create_particles(self, pos, color_name, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.0) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": self.np_random.integers(20, self.PARTICLE_LIFESPAN + 1),
                "color_name": color_name,
                "radius": self.np_random.integers(1, 4)
            })

    def _check_termination(self):
        all_fulfilled = all(req["fulfilled"] for req in self.transport_requests)
        time_up = self.steps >= self.MAX_STEPS
        return all_fulfilled or time_up

    def _generate_cell_wall(self, num_points=20, irregularity=0.3):
        points = []
        center_x, center_y = self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2
        avg_radius_x = self.SCREEN_WIDTH / 2 - 15
        avg_radius_y = self.SCREEN_HEIGHT / 2 - 15
        
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            base_radius_x = avg_radius_x * (1 + irregularity * (self.np_random.random() - 0.5))
            base_radius_y = avg_radius_y * (1 + irregularity * (self.np_random.random() - 0.5))
            x = center_x + math.cos(angle) * base_radius_x
            y = center_y + math.sin(angle) * base_radius_y
            points.append((x, y))
        return points

    def _draw_glowing_circle(self, surface, pos, radius, color, alpha=255, glow_size=1.5):
        """Draws an anti-aliased circle with a soft glow effect."""
        x, y = int(pos[0]), int(pos[1])
        
        # Draw glow
        glow_radius = int(radius * glow_size)
        glow_alpha = int(alpha / 4)
        if glow_alpha > 0:
            glow_color = (*color, glow_alpha)
            pygame.gfxdraw.filled_circle(surface, x, y, glow_radius, glow_color)
            pygame.gfxdraw.aacircle(surface, x, y, glow_radius, glow_color)

        # Draw main circle
        main_color = (*color, alpha) if alpha < 255 else color
        pygame.gfxdraw.filled_circle(surface, x, y, radius, main_color)
        pygame.gfxdraw.aacircle(surface, x, y, radius, main_color)
        
    def _draw_dashed_circle(self, surface, color, center, radius, width=1, dash_length=10):
        """Draws a dashed circle."""
        circumference = 2 * math.pi * radius
        num_dashes = int(circumference / (2 * dash_length))
        if num_dashes == 0: return

        for i in range(num_dashes):
            start_angle = (i * 2 + 0.5) * (math.pi / num_dashes)
            end_angle = (i * 2 + 1.5) * (math.pi / num_dashes)
            
            start_pos = (center[0] + math.cos(start_angle) * radius, center[1] + math.sin(start_angle) * radius)
            end_pos = (center[0] + math.cos(end_angle) * radius, center[1] + math.sin(end_angle) * radius)
            
            pygame.draw.aaline(surface, color, start_pos, end_pos, width)

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # Allow running with a display if not in a headless environment
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        pass
    else:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset(seed=42)
    
    # Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Vesicle Chain Reaction")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping for human keyboard input ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to screen ---
        # The observation is already a rendered frame, just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset(seed=42)

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}")
            obs, info = env.reset(seed=42)

        clock.tick(GameEnv.FPS)
        
    env.close()