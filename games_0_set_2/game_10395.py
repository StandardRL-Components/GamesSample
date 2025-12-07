import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:24:37.227476
# Source Brief: brief_00395.md
# Brief Index: 395
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Teleport through a growing network of nodes. Explore and activate new nodes to expand the network before time runs out."
    )
    user_guide = (
        "Use ↑ and ↓ arrow keys to select a destination node. Press space to teleport."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_TIME = 180.0
    WIN_NODES = 10
    MAX_STEPS = 1000
    NODE_SPAWN_RADIUS_MIN = 80
    NODE_SPAWN_RADIUS_MAX = 120

    # --- Colors ---
    COLOR_BG = (10, 15, 30)
    COLOR_GRID = (20, 30, 50)
    COLOR_PARTICLE = (50, 60, 90)
    COLOR_CURRENT_NODE = (0, 255, 136)
    COLOR_OPTION_NODE = (68, 136, 255)
    COLOR_VISITED_NODE = (60, 70, 100)
    COLOR_CONNECTION = (40, 50, 80)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TIMER_DANGER = (255, 80, 80)
    COLOR_SELECTOR = (255, 255, 255)
    COLOR_DILATION_TINT = (100, 0, 150)

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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.dt = 1.0 / self.FPS

        self.nodes = {}
        self.connections = {}
        self.visited_nodes = set()
        self.current_node_id = 0
        self.next_node_id = 1
        self.teleport_options = []
        self.selected_option_index = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.remaining_time = self.MAX_TIME
        self.time_dilation_factor = 1.0
        self.prev_space_held = False
        self.particles = []
        self.camera_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.camera_target_pos = np.copy(self.camera_pos)
        self.camera_zoom = 1.0
        self.camera_target_zoom = 1.0
        self.teleport_flash_alpha = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.remaining_time = self.MAX_TIME
        self.time_dilation_factor = 1.0
        self.prev_space_held = False
        self.selected_option_index = 0
        self.teleport_flash_alpha = 0
        
        self._generate_initial_network()
        self._generate_particles(100)
        
        center_pos = self.nodes[0]
        self.camera_pos = np.array(center_pos, dtype=float)
        self.camera_target_pos = np.copy(self.camera_pos)
        self.camera_zoom = 1.0
        self.camera_target_zoom = 1.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        truncated = False

        # --- Handle Input ---
        num_options = len(self.teleport_options)
        if num_options > 0:
            if movement == 1:  # Up
                self.selected_option_index = (self.selected_option_index - 1 + num_options) % num_options
            elif movement == 2:  # Down
                self.selected_option_index = (self.selected_option_index + 1) % num_options

        is_press = space_held and not self.prev_space_held
        if is_press and num_options > 0:
            reward += self._teleport()
            # SFX: Teleport_Zap.wav

        self.prev_space_held = space_held

        # --- Update Game State ---
        self.steps += 1
        self.remaining_time -= self.dt * self.time_dilation_factor
        self._update_animations()

        # --- Check Termination ---
        if len(self.visited_nodes) >= self.WIN_NODES:
            reward += 100
            terminated = True
            self.game_over = True
            # SFX: Win_Jingle.wav
        elif self.remaining_time <= 0:
            self.remaining_time = 0
            reward -= 100
            terminated = True
            self.game_over = True
            # SFX: Lose_Sound.wav
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _teleport(self):
        reward = 0.1  # Base reward for any teleport
        dest_id = self.teleport_options[self.selected_option_index]

        if dest_id not in self.visited_nodes:
            reward += 1.0  # Bonus for exploring a new node
            self.visited_nodes.add(dest_id)
        
        self.current_node_id = dest_id
        self.time_dilation_factor *= 1.2
        self.teleport_flash_alpha = 255

        self._ensure_connections(self.current_node_id)
        self._generate_teleport_options()
        self.selected_option_index = 0
        
        self.camera_target_pos = np.array(self.nodes[self.current_node_id], dtype=float)
        self._update_camera_zoom()

        return reward

    def _update_animations(self):
        # Camera smoothing
        self.camera_pos += (self.camera_target_pos - self.camera_pos) * 0.1
        self.camera_zoom += (self.camera_target_zoom - self.camera_zoom) * 0.05

        # Teleport flash fade
        if self.teleport_flash_alpha > 0:
            self.teleport_flash_alpha = max(0, self.teleport_flash_alpha - 25)

        # Particle movement
        for p in self.particles:
            p[0] += p[2] * self.time_dilation_factor
            p[1] += p[3] * self.time_dilation_factor
            if p[0] < 0 or p[0] > self.SCREEN_WIDTH or p[1] < 0 or p[1] > self.SCREEN_HEIGHT:
                p[0] = random.uniform(0, self.SCREEN_WIDTH)
                p[1] = random.uniform(0, self.SCREEN_HEIGHT)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_network()
        self._render_nodes()
        self._render_teleport_options()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "nodes_visited": len(self.visited_nodes)}

    # --- Network Generation ---
    def _generate_initial_network(self):
        self.nodes = {0: (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)}
        self.connections = {0: set()}
        self.visited_nodes = {0}
        self.current_node_id = 0
        self.next_node_id = 1
        self._ensure_connections(0)
        self._generate_teleport_options()

    def _ensure_connections(self, node_id):
        while len(self.connections.get(node_id, set())) < 3:
            self._create_new_node_connected_to(node_id)

    def _create_new_node_connected_to(self, parent_id):
        parent_pos = self.nodes[parent_id]
        new_id = self.next_node_id
        
        attempts = 0
        while attempts < 50:
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(self.NODE_SPAWN_RADIUS_MIN, self.NODE_SPAWN_RADIUS_MAX)
            new_pos = (parent_pos[0] + math.cos(angle) * radius, 
                       parent_pos[1] + math.sin(angle) * radius)
            
            # Check for collisions with other nodes
            is_valid = True
            for pos in self.nodes.values():
                dist_sq = (new_pos[0] - pos[0])**2 + (new_pos[1] - pos[1])**2
                if dist_sq < (self.NODE_SPAWN_RADIUS_MIN / 2)**2:
                    is_valid = False
                    break
            if is_valid:
                break
            attempts += 1

        self.nodes[new_id] = new_pos
        self.connections.setdefault(new_id, set()).add(parent_id)
        self.connections.setdefault(parent_id, set()).add(new_id)
        self.next_node_id += 1

    def _generate_teleport_options(self):
        neighbors = list(self.connections[self.current_node_id])
        random.shuffle(neighbors)
        self.teleport_options = neighbors[:3]

    def _generate_particles(self, count):
        self.particles = []
        for _ in range(count):
            x = random.uniform(0, self.SCREEN_WIDTH)
            y = random.uniform(0, self.SCREEN_HEIGHT)
            speed = random.uniform(0.1, 0.5)
            angle = random.uniform(0, 2 * math.pi)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particles.append([x, y, vx, vy])

    # --- Rendering ---
    def _world_to_screen(self, pos):
        x, y = pos
        center_x, center_y = self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2
        screen_x = (x - self.camera_pos[0]) * self.camera_zoom + center_x
        screen_y = (y - self.camera_pos[1]) * self.camera_zoom + center_y
        return int(screen_x), int(screen_y)
    
    def _update_camera_zoom(self):
        if not self.visited_nodes:
            self.camera_target_zoom = 1.0
            return
        
        min_x = min(self.nodes[nid][0] for nid in self.visited_nodes)
        max_x = max(self.nodes[nid][0] for nid in self.visited_nodes)
        min_y = min(self.nodes[nid][1] for nid in self.visited_nodes)
        max_y = max(self.nodes[nid][1] for nid in self.visited_nodes)
        
        world_width = max(max_x - min_x, 1) + 150 # Padding
        world_height = max(max_y - min_y, 1) + 150 # Padding
        
        zoom_x = self.SCREEN_WIDTH / world_width
        zoom_y = self.SCREEN_HEIGHT / world_height
        
        self.camera_target_zoom = min(zoom_x, zoom_y, 1.2) # Cap max zoom

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        for p in self.particles:
            pygame.gfxdraw.pixel(self.screen, int(p[0]), int(p[1]), self.COLOR_PARTICLE)

    def _render_network(self):
        drawn_connections = set()
        for node_id, neighbors in self.connections.items():
            if node_id not in self.visited_nodes: continue
            start_pos = self._world_to_screen(self.nodes[node_id])
            for neighbor_id in neighbors:
                if neighbor_id not in self.visited_nodes: continue
                # Avoid drawing lines twice
                if tuple(sorted((node_id, neighbor_id))) in drawn_connections: continue
                
                end_pos = self._world_to_screen(self.nodes[neighbor_id])
                pygame.draw.aaline(self.screen, self.COLOR_CONNECTION, start_pos, end_pos)
                drawn_connections.add(tuple(sorted((node_id, neighbor_id))))

    def _render_nodes(self):
        for node_id, pos in self.nodes.items():
            if node_id not in self.visited_nodes: continue
            screen_pos = self._world_to_screen(pos)
            
            if node_id == self.current_node_id:
                self._draw_glowing_circle(screen_pos, 12 * self.camera_zoom, self.COLOR_CURRENT_NODE)
            elif node_id in self.teleport_options:
                # Handled by _render_teleport_options
                pass
            else:
                radius = int(6 * self.camera_zoom)
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, self.COLOR_VISITED_NODE)
                pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius, self.COLOR_VISITED_NODE)

    def _render_teleport_options(self):
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
        for i, option_id in enumerate(self.teleport_options):
            pos = self.nodes[option_id]
            screen_pos = self._world_to_screen(pos)
            
            radius = int((8 + pulse * 3) * self.camera_zoom)
            self._draw_glowing_circle(screen_pos, radius, self.COLOR_OPTION_NODE)
            
            # Draw selector
            if i == self.selected_option_index:
                selector_radius = int(radius * 1.8)
                alpha = 100 + pulse * 100
                self._draw_dashed_circle(screen_pos, selector_radius, self.COLOR_SELECTOR, alpha)

            # Draw option number
            text_surf = self.font_small.render(str(i + 1), True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=screen_pos)
            self.screen.blit(text_surf, text_rect)

    def _render_effects(self):
        # Time Dilation Tint
        dilation_alpha = min(255, (self.time_dilation_factor - 1.0) * 20)
        if dilation_alpha > 0:
            tint_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            tint_surface.fill((*self.COLOR_DILATION_TINT, int(dilation_alpha)))
            self.screen.blit(tint_surface, (0, 0))

        # Teleport Flash
        if self.teleport_flash_alpha > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, self.teleport_flash_alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # Node Counter
        node_text = f"NODE: {len(self.visited_nodes):02d} / {self.WIN_NODES}"
        node_surf = self.font_main.render(node_text, True, self.COLOR_TEXT)
        self.screen.blit(node_surf, (20, 10))

        # Timer
        timer_color = self.COLOR_TEXT if self.remaining_time > 30 else self.COLOR_TIMER_DANGER
        timer_text = f"TIME: {self.remaining_time:.1f}"
        timer_surf = self.font_main.render(timer_text, True, timer_color)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(timer_surf, timer_rect)

    # --- Drawing Utilities ---
    def _draw_glowing_circle(self, pos, radius, color):
        radius = int(max(1, radius))
        for i in range(4):
            alpha = 80 - i * 20
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + i * 2, (*color, alpha))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

    def _draw_dashed_circle(self, pos, radius, color, alpha):
        num_segments = 20
        for i in range(num_segments):
            if i % 2 == 0: continue
            angle1 = (i / num_segments) * 2 * math.pi
            angle2 = ((i + 1) / num_segments) * 2 * math.pi
            start_pos = (pos[0] + math.cos(angle1) * radius, pos[1] + math.sin(angle1) * radius)
            end_pos = (pos[0] + math.cos(angle2) * radius, pos[1] + math.sin(angle2) * radius)
            pygame.draw.aaline(self.screen, (*color, alpha), start_pos, end_pos)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block will not be run by the autograder, but is useful for local testing.
    # Un-comment the following line to run in a window instead of headless
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    pygame.display.set_caption("Teleport Network")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # [movement, space, shift]
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # Key DOWN events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
            
            # Key UP events
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN]:
                    action[0] = 0
                if event.key == pygame.K_SPACE:
                    action[1] = 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Reset single-press action
        if action[0] in [1, 2]:
            action[0] = 0
            
        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()
    pygame.quit()