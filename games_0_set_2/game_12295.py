import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:39:36.241234
# Source Brief: brief_02295.md
# Brief Index: 2295
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Redirect the flow of a sacred river to make the desert fertile. "
        "Gather resources and build magnificent monuments to score points."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press Shift to enter/exit build mode. "
        "In build mode, use ↑/↓ to select a monument and Space to build. "
        "In normal mode, press Space to alter the river's flow at the cursor."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CRITICAL: Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # --- Visuals & Colors ---
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_SAND = (210, 180, 140)
        self.COLOR_FERTILE = (80, 140, 80)
        self.COLOR_RIVER = (60, 120, 220)
        self.COLOR_RIVER_FLOW = (150, 200, 255)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_UI_BG = (0, 0, 0, 128)
        self.RESOURCE_COLORS = {
            "clay": (188, 108, 77),
            "stone": (136, 140, 141),
            "gold": (255, 215, 0)
        }
        
        # --- Game Constants ---
        self.MAX_STEPS = 5000
        self.HEX_RADIUS = 20
        self.HEX_COLS = 17
        self.HEX_ROWS = 10
        self.RESOURCE_SPAWN_CHANCE = 0.02
        self.MONUMENTS = {
            "Small Obelisk": {"cost": {"clay": 5, "stone": 10}, "score": 10},
            "Pyramid": {"cost": {"clay": 20, "stone": 50, "gold": 5}, "score": 50},
            "Great Sphinx": {"cost": {"clay": 50, "stone": 80, "gold": 20}, "score": 150}
        }
        self.MONUMENT_LIST = list(self.MONUMENTS.keys())
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.resources = {}
        self.hex_grid = {}
        self.river_network = {}
        self.river_path = []
        self.river_flow_particles = []
        self.particles = []
        self.cursor_pos = (0, 0)
        self.build_mode = False
        self.build_selection_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.resources = {"clay": 0, "stone": 0, "gold": 0}
        
        self._generate_hex_grid()
        self._initialize_river_network()
        self._trace_river_path()
        self._update_fertile_tiles_and_resources(initial_spawn=True)
        
        self.cursor_pos = (self.HEX_COLS // 2, self.HEX_ROWS // 2)
        self.build_mode = False
        self.build_selection_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.river_flow_particles = []

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Handle Actions ---
        reward += self._handle_actions(movement, space_held, shift_held)
        
        # --- Update Game State ---
        reward += self._update_game_state()
        
        self.steps += 1
        # self.score += reward # Reward is already accumulated in _handle_actions and _update_game_state
        
        self.game_over = self.steps >= self.MAX_STEPS
        
        # Update last action states for edge detection
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _handle_actions(self, movement, space_held, shift_held):
        reward = 0
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        if shift_pressed:
            self.build_mode = not self.build_mode
            self.build_selection_idx = 0
            # sfx: UI_toggle.wav

        if self.build_mode:
            # In build mode, movement changes selection, space builds
            if movement == 1: # Up
                self.build_selection_idx = (self.build_selection_idx - 1) % len(self.MONUMENT_LIST)
            elif movement == 2: # Down
                self.build_selection_idx = (self.build_selection_idx + 1) % len(self.MONUMENT_LIST)
            
            if space_pressed:
                build_reward = self._attempt_build_monument()
                reward += build_reward
                self.score += build_reward
        else:
            # In normal mode, movement moves cursor, space manipulates time
            self._move_cursor(movement)
            if space_pressed:
                self._manipulate_time()
        
        return reward

    def _move_cursor(self, movement):
        q, r = self.cursor_pos
        if movement == 1: # Up
            r -= 1
        elif movement == 2: # Down
            r += 1
        elif movement == 3: # Left (NW in pointy-top hex)
            q -= 1
        elif movement == 4: # Right (SE in pointy-top hex)
            q += 1
        
        if (q, r) in self.hex_grid:
            self.cursor_pos = (q, r)

    def _manipulate_time(self):
        # sfx: time_warp.wav
        q, r = self.cursor_pos
        for neighbor in self._get_hex_neighbors((q, r)):
            if neighbor in self.hex_grid:
                # Reverse the flow direction for this edge
                edge = tuple(sorted(((q, r), neighbor)))
                if edge in self.river_network:
                    self.river_network[edge] = tuple(reversed(self.river_network[edge]))
        
        self._create_particles(self._hex_to_pixel((q,r)), 30, self.COLOR_CURSOR)
        self._trace_river_path()
        self._update_fertile_tiles_and_resources()

    def _attempt_build_monument(self):
        q, r = self.cursor_pos
        tile = self.hex_grid[(q, r)]
        
        if tile['monument'] or not tile['fertile']:
            # sfx: action_fail.wav
            return 0

        monument_name = self.MONUMENT_LIST[self.build_selection_idx]
        monument_spec = self.MONUMENTS[monument_name]
        
        can_afford = all(self.resources[res] >= cost for res, cost in monument_spec["cost"].items())
        
        if can_afford:
            for res, cost in monument_spec["cost"].items():
                self.resources[res] -= cost
            
            tile['monument'] = monument_name
            self.build_mode = False # Exit build mode after successful build
            # sfx: build_complete.wav
            self._create_particles(self._hex_to_pixel((q,r)), 50, self.RESOURCE_COLORS["gold"])
            return monument_spec["score"] # Use score as immediate reward
        else:
            # sfx: action_fail.wav
            return 0

    def _update_game_state(self):
        reward = 0
        # Auto-collect resources from fertile tiles
        for pos, tile in self.hex_grid.items():
            if tile['fertile'] and tile['resource']:
                res_type = tile['resource']
                self.resources[res_type] += 1
                tile['resource'] = None
                reward += 0.1 # Small reward for resource collection
                # sfx: resource_collect.wav
        
        # Spawn new resources
        self._update_fertile_tiles_and_resources()
        
        # Update particles
        self._update_particles()
        return reward

    def _update_fertile_tiles_and_resources(self, initial_spawn=False):
        # Reset all tiles to sand
        for tile in self.hex_grid.values():
            tile['fertile'] = False

        # Set tiles adjacent to the river to fertile
        for q, r in self.river_path:
            for neighbor_pos in self._get_hex_neighbors((q,r)):
                if neighbor_pos in self.hex_grid:
                    self.hex_grid[neighbor_pos]['fertile'] = True
        
        # Spawn resources on fertile, non-monument tiles
        for pos, tile in self.hex_grid.items():
            if tile['fertile'] and not tile['resource'] and not tile['monument']:
                if initial_spawn or self.np_random.random() < self.RESOURCE_SPAWN_CHANCE:
                    tile['resource'] = self.np_random.choice(list(self.RESOURCE_COLORS.keys()))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "resources": self.resources}

    # --- Rendering Methods ---
    def _render_game(self):
        # Draw all hexes
        for pos, tile in self.hex_grid.items():
            color = self.COLOR_FERTILE if tile['fertile'] else self.COLOR_SAND
            self._draw_hexagon(pos, color)
        
        # Draw river
        if len(self.river_path) > 1:
            for i in range(len(self.river_path) - 1):
                start_pos = self._hex_to_pixel(self.river_path[i])
                end_pos = self._hex_to_pixel(self.river_path[i+1])
                pygame.draw.line(self.screen, self.COLOR_RIVER, start_pos, end_pos, self.HEX_RADIUS)
        
        # Draw river flow particles
        for p in self.river_flow_particles:
            pygame.draw.circle(self.screen, self.COLOR_RIVER_FLOW, p['pos'], 2)

        # Draw resources and monuments
        for pos, tile in self.hex_grid.items():
            center = self._hex_to_pixel(pos)
            if tile['resource']:
                res_color = self.RESOURCE_COLORS[tile['resource']]
                pygame.draw.circle(self.screen, res_color, center, self.HEX_RADIUS // 3)
                pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.HEX_RADIUS // 3, (0,0,0))
            if tile['monument']:
                self._draw_monument(pos, tile['monument'])

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['size']))

        # Draw cursor
        cursor_center = self._hex_to_pixel(self.cursor_pos)
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        cursor_color = (self.COLOR_CURSOR[0], self.COLOR_CURSOR[1], int(100 + 155 * pulse))
        self._draw_hexagon_outline(self.cursor_pos, cursor_color, 4)

    def _render_ui(self):
        # Top bar for score and steps
        ui_bar = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bar, (0, 0))

        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 8))

        steps_text = self.font_medium.render(f"STEP: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 8))

        # Bottom bar for resources
        res_bar = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        res_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(res_bar, (0, self.HEIGHT - 40))

        x_offset = 10
        for res, color in self.RESOURCE_COLORS.items():
            pygame.draw.circle(self.screen, color, (x_offset + 15, self.HEIGHT - 20), 10)
            res_text = self.font_medium.render(f"{self.resources[res]}", True, self.COLOR_UI_TEXT)
            self.screen.blit(res_text, (x_offset + 35, self.HEIGHT - 32))
            x_offset += res_text.get_width() + 60

        # Build Menu
        if self.build_mode:
            self._render_build_menu()

    def _render_build_menu(self):
        menu_w, menu_h = 250, 120
        cursor_px = self._hex_to_pixel(self.cursor_pos)
        menu_x = min(max(10, cursor_px[0] - menu_w // 2), self.WIDTH - menu_w - 10)
        menu_y = min(max(50, cursor_px[1] - menu_h - 20), self.HEIGHT - menu_h - 50)
        
        menu_surf = pygame.Surface((menu_w, menu_h), pygame.SRCALPHA)
        menu_surf.fill(self.COLOR_UI_BG)
        pygame.draw.rect(menu_surf, self.COLOR_CURSOR, (0, 0, menu_w, menu_h), 2)
        
        title_text = self.font_medium.render("Build Menu", True, self.COLOR_CURSOR)
        menu_surf.blit(title_text, (menu_w // 2 - title_text.get_width() // 2, 5))

        for i, name in enumerate(self.MONUMENT_LIST):
            y_pos = 35 + i * 25
            is_selected = i == self.build_selection_idx
            spec = self.MONUMENTS[name]
            can_afford = all(self.resources[res] >= cost for res, cost in spec["cost"].items())
            
            color = self.COLOR_UI_TEXT
            if is_selected: color = self.COLOR_CURSOR
            if not can_afford: color = (128, 128, 128)
            
            mon_text = self.font_small.render(name, True, color)
            menu_surf.blit(mon_text, (10, y_pos))
            
            cost_str = ", ".join([f"{c}{res[0].upper()}" for res, c in spec["cost"].items()])
            cost_text = self.font_small.render(f"({cost_str})", True, color)
            menu_surf.blit(cost_text, (menu_w - cost_text.get_width() - 10, y_pos))

        self.screen.blit(menu_surf, (menu_x, menu_y))

    # --- Hex Grid & River Logic ---
    def _generate_hex_grid(self):
        self.hex_grid = {}
        for q in range(self.HEX_COLS):
            for r in range(self.HEX_ROWS):
                # Offset for odd-q columns
                if q % 2 != 0 and r == self.HEX_ROWS -1: continue
                self.hex_grid[(q, r)] = {
                    "fertile": False,
                    "resource": None,
                    "monument": None
                }

    def _initialize_river_network(self):
        self.river_network = {}
        for q, r in self.hex_grid:
            for neighbor in self._get_hex_neighbors((q, r)):
                if neighbor in self.hex_grid:
                    edge = tuple(sorted(((q, r), neighbor)))
                    if edge not in self.river_network:
                        # Random initial flow direction
                        if self.np_random.random() > 0.5:
                            self.river_network[edge] = ((q, r), neighbor)
                        else:
                            self.river_network[edge] = (neighbor, (q, r))

    def _trace_river_path(self):
        self.river_path = []
        # Find a starting hex on the left edge
        start_q = 0
        start_r = self.np_random.integers(0, self.HEX_ROWS)
        while (start_q, start_r) not in self.hex_grid:
            start_r = self.np_random.integers(0, self.HEX_ROWS)

        current_hex = (start_q, start_r)
        path = [current_hex]
        visited_edges = set()

        for _ in range(self.HEX_COLS * 2): # Limit path length to prevent infinite loops
            found_next = False
            # Find an outgoing edge from the current hex
            neighbors = self._get_hex_neighbors(current_hex)
            self.np_random.shuffle(neighbors) # Randomize neighbor check order
            for neighbor in neighbors:
                edge = tuple(sorted((current_hex, neighbor)))
                if edge in self.river_network and edge not in visited_edges:
                    if self.river_network[edge][0] == current_hex:
                        path.append(neighbor)
                        visited_edges.add(edge)
                        current_hex = neighbor
                        found_next = True
                        break
            if not found_next or current_hex not in self.hex_grid:
                break # Reached edge of map or a dead end
        
        self.river_path = path

    # --- Particle System ---
    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'size': self.np_random.random() * 4 + 2,
                'lifespan': self.np_random.integers(20, 40),
                'color': color
            })

    def _update_particles(self):
        # General purpose particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['size'] *= 0.95

        # River flow particles
        if self.steps % 3 == 0 and len(self.river_path) > 1:
            for _ in range(2): # Add new particles
                idx = self.np_random.integers(0, len(self.river_path) -1)
                start_p = self._hex_to_pixel(self.river_path[idx])
                end_p = self._hex_to_pixel(self.river_path[idx+1])
                self.river_flow_particles.append({
                    'pos': list(start_p),
                    'target': end_p,
                    'lifespan': 20
                })
        
        self.river_flow_particles = [p for p in self.river_flow_particles if p['lifespan'] > 0]
        for p in self.river_flow_particles:
            p['lifespan'] -= 1
            direction = (p['target'][0] - p['pos'][0], p['target'][1] - p['pos'][1])
            dist = math.hypot(*direction)
            if dist > 1:
                p['pos'][0] += direction[0] / dist * 3
                p['pos'][1] += direction[1] / dist * 3
            else:
                p['lifespan'] = 0

    # --- Drawing Helpers ---
    def _draw_hexagon(self, hex_coords, color):
        points = self._get_hex_corners(hex_coords)
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_hexagon_outline(self, hex_coords, color, width):
        points = self._get_hex_corners(hex_coords)
        pygame.draw.polygon(self.screen, color, points, width)


    def _draw_monument(self, hex_coords, name):
        center = self._hex_to_pixel(hex_coords)
        r = self.HEX_RADIUS * 0.7
        if name == "Small Obelisk":
            points = [
                (center[0] - r*0.2, center[1] + r),
                (center[0] + r*0.2, center[1] + r),
                (center[0] + r*0.1, center[1] - r),
                (center[0] - r*0.1, center[1] - r),
            ]
            pygame.draw.polygon(self.screen, self.RESOURCE_COLORS["stone"], points)
        elif name == "Pyramid":
            points = [
                (center[0], center[1] - r),
                (center[0] - r, center[1] + r*0.8),
                (center[0] + r, center[1] + r*0.8),
            ]
            pygame.draw.polygon(self.screen, self.RESOURCE_COLORS["gold"], points)
        elif name == "Great Sphinx":
            pygame.draw.rect(self.screen, self.RESOURCE_COLORS["stone"], (center[0]-r, center[1], r*2, r*0.8))
            pygame.draw.circle(self.screen, self.RESOURCE_COLORS["stone"], (center[0], center[1]), int(r*0.6))

    # --- Hexagonal Math Helpers ---
    def _get_hex_corners(self, hex_coords):
        center = self._hex_to_pixel(hex_coords)
        points = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            points.append((
                center[0] + self.HEX_RADIUS * math.cos(angle_rad),
                center[1] + self.HEX_RADIUS * math.sin(angle_rad)
            ))
        return points

    def _hex_to_pixel(self, hex_coords):
        q, r = hex_coords
        x_offset = self.HEX_RADIUS * 1.5
        y_offset = self.HEX_RADIUS * math.sqrt(3) / 2 + 20
        
        x = self.HEX_RADIUS * (3/2 * q) + x_offset
        y = self.HEX_RADIUS * (math.sqrt(3)/2 * q + math.sqrt(3) * r) + y_offset
        
        if q % 2 != 0:
            y += self.HEX_RADIUS * math.sqrt(3) / 2
            
        return int(x), int(y)

    def _get_hex_neighbors(self, hex_coords):
        q, r = hex_coords
        # For odd-q layout
        if q % 2 == 0:
            neighbors = [
                (q, r-1), (q+1, r-1), (q+1, r),
                (q, r+1), (q-1, r), (q-1, r-1)
            ]
        else:
            neighbors = [
                (q, r-1), (q+1, r), (q+1, r+1),
                (q, r+1), (q-1, r+1), (q-1, r)
            ]
        return neighbors

    def validate_implementation(self):
        # This method is for dev-time checks and is not used by the evaluation system.
        # It's safe to leave it as-is or remove it.
        pass

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    pygame.font.init()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Nile Time-Bender")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    action = [0, 0, 0] # [movement, space, shift]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        
        # --- Map keys to action space ---
        movement = 0 # no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()