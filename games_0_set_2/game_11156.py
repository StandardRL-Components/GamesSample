import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:32:09.349875
# Source Brief: brief_01156.md
# Brief Index: 1156
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a nanobot explores a procedurally generated
    cellular environment. The goal is to travel as far as possible by collecting
    resources and upgrading abilities.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Explore a procedurally generated cellular world as a nanobot. "
        "Collect resources to upgrade your speed and sensors, and travel as far as you can."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. "
        "Press space to upgrade speed and shift to upgrade sensor range."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000

        # Colors
        self.COLOR_BG = (10, 15, 26)
        self.COLOR_BG_CELLS = (15, 22, 38)
        self.COLOR_WALL = (34, 68, 136)
        self.COLOR_WALL_HIGHLIGHT = (50, 100, 200)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 200, 0)
        self.COLOR_RESOURCE = (0, 255, 136)
        self.COLOR_RESOURCE_GLOW = (0, 200, 100)
        self.COLOR_SENSOR = (0, 100, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 30)

        # Player
        self.PLAYER_RADIUS = 8
        self.INITIAL_SPEED = 2.0
        self.INITIAL_SENSOR_RANGE = 100
        self.INITIAL_RESOURCES = 0
        
        # World
        self.CELL_SIZE = 40
        self.INITIAL_WALL_DENSITY = 0.25
        self.WALL_DENSITY_INCREASE_RATE = 0.05
        self.DISTANCE_PER_DENSITY_INCREASE = 500
        self.RESOURCE_NODE_SPAWN_CHANCE = 0.05

        # Upgrades
        self.UPGRADE_COST_SPEED = 10
        self.UPGRADE_COST_SENSOR = 15
        self.SPEED_UPGRADE_AMOUNT = 0.5
        self.SENSOR_UPGRADE_AMOUNT = 20

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
        self.font_main = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 36)

        # --- Internal State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.nanobot_pos = pygame.Vector2(0, 0)
        self.camera_offset = pygame.Vector2(0, 0)
        self.resources = 0
        self.upgrade_levels = {}
        self.current_speed = 0
        self.current_sensor_range = 0
        self.distance_traveled = 0
        self.last_distance_from_origin = 0
        self.world_grid = {}
        self.resource_nodes = {}
        self.particles = []
        self.background_cells = []
        self.last_space_held = False
        self.last_shift_held = False
        self.steps_since_game_over = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.steps_since_game_over = 0
        
        self.nanobot_pos = pygame.Vector2(0, 0)
        self.camera_offset = pygame.Vector2(0, 0)
        
        self.resources = self.INITIAL_RESOURCES
        self.upgrade_levels = {"speed": 1, "sensor_range": 1}
        self.current_speed = self.INITIAL_SPEED
        self.current_sensor_range = self.INITIAL_SENSOR_RANGE
        
        self.distance_traveled = 0
        self.last_distance_from_origin = 0
        
        self.world_grid.clear()
        self.resource_nodes.clear()
        self.particles.clear()
        
        self.last_space_held = False
        self.last_shift_held = False

        # Generate a clear starting area
        for x in range(-5, 6):
            for y in range(-5, 6):
                self.world_grid[(x, y)] = 0 # 0 for empty space

        self._generate_background_cells()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            self.steps_since_game_over += 1
            terminated = True
            truncated = self.steps >= self.MAX_STEPS
            return self._get_observation(), 0, terminated, truncated, self._get_info()

        self.steps += 1
        step_reward = 0

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Upgrades trigger on press, not hold
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        upgrade_feedback = self._handle_upgrades(space_pressed, shift_pressed)
        step_reward += upgrade_feedback
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        # --- Movement ---
        move_vector = pygame.Vector2(0, 0)
        if movement == 1: move_vector.y = -1
        elif movement == 2: move_vector.y = 1
        elif movement == 3: move_vector.x = -1
        elif movement == 4: move_vector.x = 1

        if move_vector.length() > 0:
            move_vector.normalize_ip()
            target_pos = self.nanobot_pos + move_vector * self.current_speed
            
            # --- Collision Detection ---
            if self._check_collision(target_pos):
                self.game_over = True
                step_reward -= 100 # Terminal penalty
                # sound: player_die.wav
            else:
                self.nanobot_pos = target_pos
        
        # --- Update Game State ---
        if not self.game_over:
            # Passive resource gain
            self.resources += 0.01

            # Distance reward
            current_dist_from_origin = self.nanobot_pos.length()
            dist_delta = current_dist_from_origin - self.last_distance_from_origin
            if dist_delta > 0:
                step_reward += dist_delta * 0.1
                self.distance_traveled += dist_delta # More accurate distance traveled
            self.last_distance_from_origin = current_dist_from_origin

            # Resource collection
            step_reward += self._collect_resources()
        
        # Update camera to smoothly follow player
        self.camera_offset = self.nanobot_pos.copy()

        # Update particles
        self._update_particles()
        
        # --- Termination ---
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if not terminated and self.distance_traveled >= 5000:
             step_reward += 100 # Goal reward
             terminated = True
        
        self.score += step_reward

        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_upgrades(self, space_pressed, shift_pressed):
        reward = 0
        if space_pressed and self.resources >= self.UPGRADE_COST_SPEED:
            self.resources -= self.UPGRADE_COST_SPEED
            self.upgrade_levels["speed"] += 1
            self.current_speed += self.SPEED_UPGRADE_AMOUNT
            reward += 0.5 # Small reward for upgrading
            self._create_particles(self.nanobot_pos, 15, self.COLOR_PLAYER, 1.5)
            # sound: upgrade_speed.wav
        
        if shift_pressed and self.resources >= self.UPGRADE_COST_SENSOR:
            self.resources -= self.UPGRADE_COST_SENSOR
            self.upgrade_levels["sensor_range"] += 1
            self.current_sensor_range += self.SENSOR_UPGRADE_AMOUNT
            reward += 0.5 # Small reward for upgrading
            self._create_particles(self.nanobot_pos, 15, self.COLOR_SENSOR, 1.5)
            # sound: upgrade_sensor.wav
        return reward

    def _get_grid_coords(self, world_pos):
        return int(world_pos.x // self.CELL_SIZE), int(world_pos.y // self.CELL_SIZE)

    def _get_cell_type(self, gx, gy):
        if (gx, gy) not in self.world_grid:
            # Procedurally generate new cells
            dist_milestone = self.distance_traveled // self.DISTANCE_PER_DENSITY_INCREASE
            current_density = self.INITIAL_WALL_DENSITY + dist_milestone * self.WALL_DENSITY_INCREASE_RATE
            
            # Ensure connectivity by preventing 2x2 wall blocks
            neighbors = [
                self.world_grid.get((gx-1, gy), 0),
                self.world_grid.get((gx, gy-1), 0),
                self.world_grid.get((gx-1, gy-1), 0)
            ]
            if sum(neighbors) == 3: # If this would form a 2x2 block, force it to be empty
                 self.world_grid[(gx, gy)] = 0
            elif self.np_random.random() < current_density:
                self.world_grid[(gx, gy)] = 1 # Wall
            else:
                self.world_grid[(gx, gy)] = 0 # Empty
                if self.np_random.random() < self.RESOURCE_NODE_SPAWN_CHANCE:
                    # Place resource node in the center of the cell
                    node_pos = pygame.Vector2((gx + 0.5) * self.CELL_SIZE, (gy + 0.5) * self.CELL_SIZE)
                    self.resource_nodes[(gx, gy)] = node_pos
        return self.world_grid.get((gx, gy), 0)

    def _check_collision(self, pos):
        radius = self.PLAYER_RADIUS
        # Check a 3x3 grid around the player for potential collisions
        gx_base, gy_base = self._get_grid_coords(pos)
        for gx_offset in range(-1, 2):
            for gy_offset in range(-1, 2):
                gx, gy = gx_base + gx_offset, gy_base + gy_offset
                if self._get_cell_type(gx, gy) == 1:
                    wall_rect = pygame.Rect(gx * self.CELL_SIZE, gy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    # Simple but effective circle-rect collision
                    closest_x = max(wall_rect.left, min(pos.x, wall_rect.right))
                    closest_y = max(wall_rect.top, min(pos.y, wall_rect.bottom))
                    distance_squared = (pos.x - closest_x)**2 + (pos.y - closest_y)**2
                    if distance_squared < radius**2:
                        return True
        return False

    def _collect_resources(self):
        collected_reward = 0
        collected_nodes = []
        for gx_gy, pos in self.resource_nodes.items():
            if self.nanobot_pos.distance_to(pos) < self.PLAYER_RADIUS + 5: # 5 is resource radius
                collected_nodes.append(gx_gy)
                self.resources += 1
                collected_reward += 1
                self._create_particles(pos, 20, self.COLOR_RESOURCE, 2.0)
                # sound: collect_resource.wav
        
        for key in collected_nodes:
            del self.resource_nodes[key]
            
        return collected_reward

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": self.distance_traveled,
            "resources": int(self.resources),
            "upgrade_speed": self.upgrade_levels["speed"],
            "upgrade_sensor": self.upgrade_levels["sensor_range"],
        }

    def close(self):
        pygame.quit()

    # --- Rendering Methods ---
    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        
        # Center of screen where player is drawn
        screen_center = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)

        self._render_background(screen_center)
        self._render_world(screen_center)
        self._render_particles(screen_center)
        self._render_player(screen_center)
        self._render_ui()

    def _generate_background_cells(self):
        self.background_cells = []
        for _ in range(50):
            self.background_cells.append({
                "pos": pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                "radius": self.np_random.uniform(20, 80),
                "parallax": self.np_random.uniform(0.1, 0.3)
            })

    def _render_background(self, screen_center):
        for cell in self.background_cells:
            # Parallax effect
            offset_x = (screen_center.x - self.camera_offset.x * cell["parallax"]) % self.WIDTH
            offset_y = (screen_center.y - self.camera_offset.y * cell["parallax"]) % self.HEIGHT
            
            pos_x = (cell["pos"].x + offset_x) % (self.WIDTH + cell["radius"]*2) - cell["radius"]
            pos_y = (cell["pos"].y + offset_y) % (self.HEIGHT + cell["radius"]*2) - cell["radius"]

            pygame.gfxdraw.aacircle(self.screen, int(pos_x), int(pos_y), int(cell["radius"]), self.COLOR_BG_CELLS)

    def _render_world(self, screen_center):
        # Calculate visible grid range
        start_gx = int((self.camera_offset.x - self.WIDTH / 2) / self.CELL_SIZE) - 1
        end_gx = int((self.camera_offset.x + self.WIDTH / 2) / self.CELL_SIZE) + 1
        start_gy = int((self.camera_offset.y - self.HEIGHT / 2) / self.CELL_SIZE) - 1
        end_gy = int((self.camera_offset.y + self.HEIGHT / 2) / self.CELL_SIZE) + 1

        # Render resource nodes first (they are behind walls)
        for gx in range(start_gx, end_gx):
            for gy in range(start_gy, end_gy):
                if (gx, gy) in self.resource_nodes:
                    node_pos = self.resource_nodes[(gx, gy)]
                    draw_pos = node_pos - self.camera_offset + screen_center
                    self._draw_glowing_circle(draw_pos, 5, self.COLOR_RESOURCE, self.COLOR_RESOURCE_GLOW)

        # Render walls
        for gx in range(start_gx, end_gx):
            for gy in range(start_gy, end_gy):
                if self._get_cell_type(gx, gy) == 1:
                    rect = pygame.Rect(
                        (gx * self.CELL_SIZE) - self.camera_offset.x + screen_center.x,
                        (gy * self.CELL_SIZE) - self.camera_offset.y + screen_center.y,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                    # Draw lines on edges bordering empty space for a cleaner look
                    if self._get_cell_type(gx, gy - 1) == 0: # Top
                        pygame.draw.line(self.screen, self.COLOR_WALL_HIGHLIGHT, rect.topleft, rect.topright, 2)
                    if self._get_cell_type(gx, gy + 1) == 0: # Bottom
                        pygame.draw.line(self.screen, self.COLOR_WALL, rect.bottomleft, rect.bottomright, 2)
                    if self._get_cell_type(gx - 1, gy) == 0: # Left
                        pygame.draw.line(self.screen, self.COLOR_WALL, rect.topleft, rect.bottomleft, 2)
                    if self._get_cell_type(gx + 1, gy) == 0: # Right
                        pygame.draw.line(self.screen, self.COLOR_WALL, rect.topright, rect.bottomright, 2)

    def _render_player(self, screen_center):
        # Sensor Range
        if not self.game_over:
            s_surf = pygame.Surface((self.current_sensor_range * 2, self.current_sensor_range * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s_surf, int(self.current_sensor_range), int(self.current_sensor_range), int(self.current_sensor_range), (*self.COLOR_SENSOR, 30))
            pygame.gfxdraw.aacircle(s_surf, int(self.current_sensor_range), int(self.current_sensor_range), int(self.current_sensor_range), (*self.COLOR_SENSOR, 60))
            self.screen.blit(s_surf, (screen_center.x - self.current_sensor_range, screen_center.y - self.current_sensor_range))

        # Player Nanobot
        if self.game_over:
            # Death effect
            t = min(1.0, self.steps_since_game_over / 20.0)
            radius = self.PLAYER_RADIUS * (1.0 - t)
            color = tuple(int(c * (1.0 - t)) for c in self.COLOR_PLAYER)
            self._draw_glowing_circle(screen_center, radius, color, color)
        else:
            self._draw_glowing_circle(screen_center, self.PLAYER_RADIUS, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)


    def _draw_glowing_circle(self, pos, radius, color, glow_color):
        if radius <= 0: return
        
        surf = pygame.Surface((radius * 3, radius * 3), pygame.SRCALPHA)
        center_pos = (surf.get_width() // 2, surf.get_height() // 2)

        # Draw multiple concentric circles for a bloom/glow effect
        for i in range(int(radius), 0, -2):
            alpha = 1 - (i / radius)
            current_color = (
                glow_color[0], glow_color[1], glow_color[2], int(alpha * 50)
            )
            pygame.gfxdraw.filled_circle(surf, center_pos[0], center_pos[1], i + int(radius/2), current_color)
        
        pygame.gfxdraw.filled_circle(surf, center_pos[0], center_pos[1], int(radius), color)
        pygame.gfxdraw.aacircle(surf, center_pos[0], center_pos[1], int(radius), color)

        self.screen.blit(surf, (pos.x - center_pos[0], pos.y - center_pos[1]))


    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, pos, color=self.COLOR_TEXT, shadow_color=self.COLOR_TEXT_SHADOW):
            shadow_surf = font.render(text, True, shadow_color)
            text_surf = font.render(text, True, color)
            self.screen.blit(shadow_surf, (pos[0] + 1, pos[1] + 1))
            self.screen.blit(text_surf, pos)

        # Resources
        draw_text(f"RESOURCES: {int(self.resources)}", self.font_main, (10, 10))
        
        # Upgrades
        speed_text = f"SPEED: LV {self.upgrade_levels['speed']}"
        sensor_text = f"SENSOR: LV {self.upgrade_levels['sensor_range']}"
        draw_text(speed_text, self.font_main, (self.WIDTH - 150, 10))
        draw_text(sensor_text, self.font_main, (self.WIDTH - 150, 30))

        # Distance
        dist_text = f"DISTANCE: {int(self.distance_traveled)}"
        text_width = self.font_title.size(dist_text)[0]
        draw_text(dist_text, self.font_title, (self.WIDTH/2 - text_width/2, self.HEIGHT - 40))

        if self.game_over:
            end_text = "SYSTEM FAILURE"
            text_width, text_height = self.font_title.size(end_text)
            draw_text(end_text, self.font_title, (self.WIDTH/2 - text_width/2, self.HEIGHT/2 - text_height/2), color=(255, 50, 50))


    # --- Particle System ---
    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.0, 3.0) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 30)
            self.particles.append([pos.copy(), vel, lifespan, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1] # Update position
            p[1] *= 0.95 # Apply drag
            p[2] -= 1 # Decrease lifespan
        self.particles = [p for p in self.particles if p[2] > 0]

    def _render_particles(self, screen_center):
        if not self.particles: return
        for pos, vel, lifespan, color in self.particles:
            draw_pos = pos - self.camera_offset + screen_center
            alpha = max(0, min(255, int(255 * (lifespan / 20.0))))
            size = max(1, int(3 * (lifespan / 20.0)))
            rect = pygame.Rect(draw_pos.x - size // 2, draw_pos.y - size // 2, size, size)
            
            # Create a temporary surface for the particle to handle alpha correctly
            part_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            part_surf.fill((*color[:3], alpha))
            self.screen.blit(part_surf, rect.topleft)

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Nanobot Explorer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
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
                total_reward = 0
                print("--- ENV RESET ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            print("--- ENV RESET ---")

        clock.tick(env.FPS)
        
    env.close()