import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:19:00.911748
# Source Brief: brief_00923.md
# Brief Index: 923
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An expert-crafted Gymnasium environment simulating a microscopic world.

    The agent guides colored organelles through a dynamic cell environment,
    using movement and time manipulation to deliver a target organelle to
    the nucleus while avoiding destructive obstacles. The environment
    prioritizes high-quality visuals, smooth animations, and satisfying
    game feel, making it suitable for both reinforcement learning and
    human play.

    Action Space: MultiDiscrete([5, 2, 2])
    - Movement: [0: None, 1: Up, 2: Down, 3: Left, 4: Right]
    - Time Manipulation: [0: Released, 1: Held] (Space Bar)
    - Select Organelle: [0: Released, 1: Held] (Shift Key)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - A 640x400 RGB image of the game state.

    Reward Structure:
    - Goal: +100 for delivering the target organelle to the nucleus.
    - Failure: -100 for losing all organelles or running out of time.
    - Events: +5 for destroying an obstacle, +2 for healing an organelle.
    - Survival: +0.1 per step for the target organelle's survival.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Guide colored organelles through a dynamic cell environment. Deliver the target organelle to the nucleus while avoiding obstacles and managing health."
    )
    user_guide = (
        "Controls: Use arrow keys to move the selected organelle. Press Shift to cycle between organelles and hold Space to slow down nearby objects."
    )
    auto_advance = True


    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_MEMBRANE = (40, 30, 60)
    COLOR_NUCLEUS_GLOW = (180, 150, 255)
    COLOR_NUCLEUS_CORE = (230, 220, 255)
    COLOR_OBSTACLE = (50, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TIME_MANIP = (100, 150, 255, 100)
    COLOR_SELECTION = (255, 255, 100)
    
    ORGANELLE_COLORS = {
        "red": (255, 80, 80),
        "green": (80, 255, 80),
        "blue": (80, 80, 255),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("sans-serif", 24)
        self.font_small = pygame.font.SysFont("sans-serif", 16)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.organelles = []
        self.obstacles = []
        self.particles = []
        
        self.selected_organelle_idx = 0
        self.target_organelle_idx = 0
        self.prev_shift_state = 0
        
        self.time_manip_active = False
        self.time_manip_pos = pygame.Vector2(0, 0)
        self.time_manip_radius = 0
        
        self.nucleus_pos = pygame.Vector2(self.WIDTH - 100, self.HEIGHT / 2)
        self.nucleus_radius = 40
        
        # This will be called once to set up the initial state, but reset is needed for gym compliance
        # self.reset() # Avoid calling reset in init
        
        # Critical self-check
        # self.validate_implementation() # Avoid running this in init as it calls step/reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # --- Reset Organelles ---
        self.organelles.clear()
        start_positions = [
            pygame.Vector2(100, self.HEIGHT / 2 - 80),
            pygame.Vector2(100, self.HEIGHT / 2),
            pygame.Vector2(100, self.HEIGHT / 2 + 80),
        ]
        colors = ["red", "green", "blue"]
        # Use self.np_random for shuffling to be deterministic with seed
        shuffled_indices = self.np_random.permutation(len(start_positions))
        shuffled_positions = [start_positions[i] for i in shuffled_indices]

        for i, color_name in enumerate(colors):
            self.organelles.append({
                "pos": shuffled_positions[i].copy(),
                "vel": pygame.Vector2(0, 0),
                "radius": 12,
                "color_name": color_name,
                "color_val": self.ORGANELLE_COLORS[color_name],
                "health": 100,
                "max_health": 100,
            })
        self.target_organelle_idx = self.np_random.integers(0, len(self.organelles))
        self.selected_organelle_idx = 0

        # --- Reset Obstacles ---
        self.obstacles.clear()
        for _ in range(5):
            self._spawn_obstacle()

        # --- Reset Particles & Effects ---
        self.particles.clear()
        self.time_manip_active = False
        self.prev_shift_state = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        events = self._update_game_state()
        reward = self._calculate_reward(events)
        self.score += reward
        
        self.steps += 1
        terminated = self._check_termination()
        truncated = False # Per requirements

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # --- Organelle Selection (Shift) ---
        if shift_held and not self.prev_shift_state and len(self.organelles) > 0:
            self.selected_organelle_idx = (self.selected_organelle_idx + 1) % len(self.organelles)
            # SFX: UI_Switch
        self.prev_shift_state = shift_held

        if not self.organelles:
            return
            
        selected_org = self.organelles[self.selected_organelle_idx]
        
        # --- Movement ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
        
        acceleration = 0.8
        selected_org["vel"] += move_vec * acceleration

        # --- Time Manipulation (Space) ---
        self.time_manip_active = space_held
        if self.time_manip_active:
            self.time_manip_pos = selected_org["pos"].copy()
            base_radius = 80
            # Blue organelle amplifies the effect
            if selected_org["color_name"] == "blue":
                self.time_manip_radius = base_radius * 1.75
            else:
                self.time_manip_radius = base_radius
            # SFX: Time_Slow_Loop (start/stop)

    def _update_game_state(self):
        events = []
        
        # --- Update Organelles ---
        for i, org in enumerate(self.organelles):
            # Apply friction
            org["vel"] *= 0.92
            if org["vel"].length_squared() < 0.1:
                org["vel"] = pygame.Vector2(0, 0)
            
            # Cap speed
            if org["vel"].length() > 5:
                org["vel"].scale_to_length(5)

            org["pos"] += org["vel"]

            # Boundary collision
            if org["pos"].x < org["radius"]: org["pos"].x = org["radius"]; org["vel"].x *= -0.5
            if org["pos"].x > self.WIDTH - org["radius"]: org["pos"].x = self.WIDTH - org["radius"]; org["vel"].x *= -0.5
            if org["pos"].y < org["radius"]: org["pos"].y = org["radius"]; org["vel"].y *= -0.5
            if org["pos"].y > self.HEIGHT - org["radius"]: org["pos"].y = self.HEIGHT - org["radius"]; org["vel"].y *= -0.5
        
        # --- Update Obstacles ---
        difficulty_speed_mod = 1.0 + (self.steps // 200) * 0.05
        for obs in self.obstacles:
            speed = obs["base_speed"] * difficulty_speed_mod
            # Time manipulation slowdown
            if self.time_manip_active and self.time_manip_pos.distance_to(obs["pos"]) < self.time_manip_radius + obs["radius"]:
                speed *= 0.2
            
            obs["pos"] += obs["vel"] * speed
            obs["angle"] = (obs["angle"] + obs["rot_speed"] * speed) % 360

            # Boundary wrapping
            if obs["pos"].x < -obs["radius"]: obs["pos"].x = self.WIDTH + obs["radius"]
            if obs["pos"].x > self.WIDTH + obs["radius"]: obs["pos"].x = -obs["radius"]
            if obs["pos"].y < -obs["radius"]: obs["pos"].y = self.HEIGHT + obs["radius"]
            if obs["pos"].y > self.HEIGHT + obs["radius"]: obs["pos"].y = -obs["radius"]

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] = max(0, p["radius"] - 0.1)

        # --- Collision Detection ---
        # Organelle vs Obstacle
        for org in self.organelles[:]:
            for obs in self.obstacles[:]:
                if org["pos"].distance_to(obs["pos"]) < org["radius"] + obs["radius"]:
                    if org["color_name"] == "red":
                        events.append("obstacle_destroyed")
                        self._create_particles(obs["pos"], (200, 200, 200), 20, 3)
                        self.obstacles.remove(obs)
                        self._spawn_obstacle()
                        # SFX: Explosion
                    else:
                        org["health"] -= 35
                        self._create_particles(org["pos"], (255, 150, 0), 10, 2)
                        # SFX: Damage_Taken
                    # Bounce effect
                    org["vel"] *= -0.8

        # Organelle vs Organelle
        for i in range(len(self.organelles)):
            for j in range(i + 1, len(self.organelles)):
                org1 = self.organelles[i]
                org2 = self.organelles[j]
                dist = org1["pos"].distance_to(org2["pos"])
                if dist < org1["radius"] + org2["radius"]:
                    # Collision response (simple bounce)
                    overlap = (org1["radius"] + org2["radius"]) - dist
                    if dist > 0:
                        normal = (org2["pos"] - org1["pos"]).normalize()
                    else: # If they are on top of each other, push them apart randomly
                        normal = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize()
                    
                    org1["pos"] -= normal * overlap / 2
                    org2["pos"] += normal * overlap / 2
                    
                    # Healing interaction
                    if org1["color_name"] == "green" and org2["health"] < org2["max_health"]:
                        healed_amount = min(25, org2["max_health"] - org2["health"])
                        if healed_amount > 0:
                            org2["health"] += healed_amount
                            events.append("organelle_healed")
                            self._create_particles(org2["pos"], org1["color_val"], 15, 1)
                            # SFX: Heal
                    if org2["color_name"] == "green" and org1["health"] < org1["max_health"]:
                        healed_amount = min(25, org1["max_health"] - org1["health"])
                        if healed_amount > 0:
                            org1["health"] += healed_amount
                            events.append("organelle_healed")
                            self._create_particles(org1["pos"], org2["color_val"], 15, 1)
                            # SFX: Heal

        # --- Check for destroyed organelles ---
        destroyed_indices = [i for i, org in enumerate(self.organelles) if org["health"] <= 0]
        if destroyed_indices:
            for i in sorted(destroyed_indices, reverse=True):
                # SFX: Organelle_Destroyed
                self._create_particles(self.organelles[i]["pos"], self.organelles[i]["color_val"], 30, 4)
                del self.organelles[i]
            
            # Adjust indices after deletion
            if self.target_organelle_idx in destroyed_indices:
                self.target_organelle_idx = -1 # Mark as destroyed
            else:
                self.target_organelle_idx -= sum(1 for di in destroyed_indices if di < self.target_organelle_idx)
            
            if self.organelles:
                self.selected_organelle_idx %= len(self.organelles)
            else:
                events.append("all_organelles_destroyed")

        # --- Organelle vs Nucleus (Win Condition) ---
        if self.target_organelle_idx != -1 and self.organelles and self.target_organelle_idx < len(self.organelles):
            target_org = self.organelles[self.target_organelle_idx]
            if target_org["pos"].distance_to(self.nucleus_pos) < target_org["radius"] + self.nucleus_radius - 10:
                events.append("goal_reached")
                # SFX: Win_Stinger

        return events

    def _calculate_reward(self, events):
        reward = 0.0
        
        # Survival reward for target
        if self.target_organelle_idx != -1 and not self.game_over:
            reward += 0.1

        for event in events:
            if event == "goal_reached":
                reward += 100
                self.game_over = True
            elif event == "obstacle_destroyed":
                reward += 5
            elif event == "organelle_healed":
                reward += 2
            elif event == "all_organelles_destroyed":
                reward -= 100
                self.game_over = True
        
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.score -= 100 # Penalty for running out of time
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "organelles_left": len(self.organelles)}

    def _render_game(self):
        # Draw membrane
        pygame.gfxdraw.aaellipse(self.screen, self.WIDTH // 2, self.HEIGHT // 2, self.WIDTH // 2 - 5, self.HEIGHT // 2 - 5, self.COLOR_MEMBRANE)

        # Draw nucleus
        self._draw_glowing_circle(self.screen, self.COLOR_NUCLEUS_GLOW, self.nucleus_pos, self.nucleus_radius, 15)
        pygame.gfxdraw.filled_circle(self.screen, int(self.nucleus_pos.x), int(self.nucleus_pos.y), self.nucleus_radius, self.COLOR_NUCLEUS_CORE)
        pygame.gfxdraw.aacircle(self.screen, int(self.nucleus_pos.x), int(self.nucleus_pos.y), self.nucleus_radius, self.COLOR_NUCLEUS_GLOW)
        
        # Draw time manipulation field
        if self.time_manip_active:
            # Use a temporary surface for transparency
            temp_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surface, int(self.time_manip_pos.x), int(self.time_manip_pos.y), int(self.time_manip_radius), self.COLOR_TIME_MANIP)
            self.screen.blit(temp_surface, (0, 0))

        # Draw obstacles
        for obs in self.obstacles:
            self._draw_rotated_triangle(self.screen, self.COLOR_OBSTACLE, obs["pos"], obs["radius"], obs["angle"])

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            # Use temp surface for alpha blending
            temp_surface = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surface, int(p["radius"]), int(p["radius"]), int(p["radius"]), color)
            self.screen.blit(temp_surface, (int(p["pos"].x - p["radius"]), int(p["pos"].y - p["radius"])))


        # Draw organelles and their effects
        for i, org in enumerate(self.organelles):
            # Health aura
            health_pct = org["health"] / org["max_health"]
            aura_radius = int(org["radius"] * (1.2 + 0.8 * health_pct))
            aura_color = (*org["color_val"], int(70 * health_pct))
            temp_surface = pygame.Surface((aura_radius*2, aura_radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surface, aura_radius, aura_radius, aura_radius, aura_color)
            self.screen.blit(temp_surface, (int(org["pos"].x-aura_radius), int(org["pos"].y-aura_radius)))
            
            # Core
            pygame.gfxdraw.filled_circle(self.screen, int(org["pos"].x), int(org["pos"].y), org["radius"], org["color_val"])
            pygame.gfxdraw.aacircle(self.screen, int(org["pos"].x), int(org["pos"].y), org["radius"], (255, 255, 255))
            
            # Target indicator
            if i == self.target_organelle_idx:
                self._draw_target_indicator(org["pos"], org["radius"])
            
            # Selection indicator
            if i == self.selected_organelle_idx:
                self._draw_selection_indicator(org["pos"], org["radius"])

    def _render_ui(self):
        # Time bar
        time_pct = 1.0 - (self.steps / self.MAX_STEPS)
        bar_width = int(self.WIDTH * 0.4 * time_pct)
        bar_x = self.WIDTH * 0.3
        pygame.draw.rect(self.screen, (80, 80, 100), (bar_x, 10, self.WIDTH * 0.4, 15))
        pygame.draw.rect(self.screen, (150, 150, 200), (bar_x, 10, bar_width, 15))
        
        # Score text
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Steps text
        steps_text = self.font_small.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 15, 10))

        # Selected Organelle Text
        if self.organelles and self.selected_organelle_idx < len(self.organelles):
            selected_org = self.organelles[self.selected_organelle_idx]
            color_name = selected_org["color_name"].upper()
            selected_text = self.font_small.render(f"Selected: {color_name}", True, selected_org["color_val"])
            self.screen.blit(selected_text, (15, 35))

    def _spawn_obstacle(self):
        edge = self.np_random.integers(4)
        radius = self.np_random.uniform(10, 15)
        if edge == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -radius)
            vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(0.5, 1.5))
        elif edge == 1: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + radius)
            vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1.5, -0.5))
        elif edge == 2: # Left
            pos = pygame.Vector2(-radius, self.np_random.uniform(0, self.HEIGHT))
            vel = pygame.Vector2(self.np_random.uniform(0.5, 1.5), self.np_random.uniform(-1, 1))
        else: # Right
            pos = pygame.Vector2(self.WIDTH + radius, self.np_random.uniform(0, self.HEIGHT))
            vel = pygame.Vector2(self.np_random.uniform(-1.5, -0.5), self.np_random.uniform(-1, 1))
        
        if vel.length_squared() > 0:
            vel.normalize_ip()
        self.obstacles.append({
            "pos": pos,
            "vel": vel,
            "radius": radius,
            "base_speed": self.np_random.uniform(1.0, 2.0),
            "angle": self.np_random.uniform(0, 360),
            "rot_speed": self.np_random.uniform(-3, 3),
        })

    def _create_particles(self, pos, color, count, speed_mult):
        for _ in range(count):
            vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            if vel.length_squared() > 0:
                vel.normalize_ip()
            vel *= self.np_random.uniform(0.5, 1.5) * speed_mult
            life = self.np_random.integers(20, 40)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": self.np_random.uniform(3, 6),
                "color": color,
                "life": life,
                "max_life": life,
            })
            
    def _draw_glowing_circle(self, surface, color, pos, radius, glow_width):
        temp_surf = pygame.Surface((radius * 2 + glow_width * 2, radius * 2 + glow_width * 2), pygame.SRCALPHA)
        center = (radius + glow_width, radius + glow_width)
        for i in range(glow_width, 0, -1):
            alpha = int(100 * (1 - i / glow_width))
            pygame.gfxdraw.filled_circle(temp_surf, center[0], center[1], int(radius + i), (*color, alpha))
        surface.blit(temp_surf, (int(pos.x - center[0]), int(pos.y - center[1])))

    def _draw_rotated_triangle(self, surface, color, pos, radius, angle):
        points = []
        for i in range(3):
            theta = math.radians(angle + 120 * i)
            x = pos.x + radius * math.cos(theta)
            y = pos.y + radius * math.sin(theta)
            points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)
        
    def _draw_selection_indicator(self, pos, radius):
        anim_angle = (self.steps * 6) % 360
        outer_radius = radius + 8 + 2 * math.sin(math.radians(anim_angle * 2))
        points = []
        for i in range(3):
            angle = math.radians(anim_angle + 120 * i)
            points.append((
                int(pos.x + outer_radius * math.cos(angle)),
                int(pos.y + outer_radius * math.sin(angle))
            ))
        pygame.gfxdraw.aatrigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.COLOR_SELECTION)

    def _draw_target_indicator(self, pos, radius):
        anim_angle = (self.steps * 4) % 360
        for i in range(4):
            angle = math.radians(anim_angle + 90 * i)
            start_pos = pos + pygame.Vector2(radius + 4, 0).rotate(math.degrees(angle))
            end_pos = pos + pygame.Vector2(radius + 8, 0).rotate(math.degrees(angle))
            pygame.draw.line(self.screen, self.COLOR_TEXT, start_pos, end_pos, 2)

if __name__ == '__main__':
    # This block allows you to run the environment directly for human play testing
    # Since we are using a dummy video driver, we need to unset it to see the window
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Create a display window
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Organelle Odyssey")
    
    last_shift_press = False

    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        # Human input mapping
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        current_shift_press = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        if current_shift_press and not last_shift_press:
             shift_held = 1 # Register a press only on the rising edge
        last_shift_press = current_shift_press

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Reset the game after a short delay
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)
        
    pygame.quit()