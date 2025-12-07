import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:03:47.185962
# Source Brief: brief_00241.md
# Brief Index: 241
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Symbiotic Clones', a puzzle-platformer.
    The agent controls a cursor to place different types of 'organisms'
    which interact with the environment and each other to create a path
    from a start point to an exit portal.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A puzzle game where you place different types of 'organisms' to build a path from a start point to an exit portal."
    )
    user_guide = (
        "Controls: Use ↑↓←→ to move the cursor. Press space to place an organism. Press shift to cycle between organism types."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 20
    CELL_SIZE = 20

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_WALL = (40, 50, 70)
    COLOR_GRID = (25, 30, 45)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_GLOW = (255, 255, 0, 50)
    COLOR_START = (100, 255, 100)
    COLOR_EXIT = (255, 100, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_DESTRUCTIBLE = (100, 80, 80)

    # Organism Types & Colors
    ORG_NONE = 0
    ORG_GREEN = 1 # Platform
    ORG_RED = 2   # Destroyer
    ORG_BLUE = 3  # Bridge
    ORG_YELLOW = 4# Activator
    
    ORGANISM_DATA = {
        ORG_GREEN: {"name": "GROW", "color": (0, 255, 128)},
        ORG_RED: {"name": "BURST", "color": (255, 80, 80)},
        ORG_BLUE: {"name": "SPAN", "color": (80, 150, 255)},
        ORG_YELLOW: {"name": "CHARGE", "color": (255, 220, 0)},
    }

    # Tile Types
    TILE_EMPTY = 0
    TILE_WALL = 1
    TILE_DESTRUCTIBLE = 2
    TILE_START = 3
    TILE_EXIT = 4
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Exact spaces as per specification
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Game state (initialized in reset)
        self.level_index = 0
        self.level_templates = self._define_levels()
        self.grid = None
        self.start_pos = None
        self.exit_pos = None
        self.organisms = []
        self.particles = []
        self.clone_counts = {}
        self.unlocked_organisms = []
        self.cursor_pos = [0, 0]
        self.selected_organism_idx = 0
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.last_space_held = False
        self.last_shift_held = False

        # self.reset() is called by the environment wrapper, no need to call it here.

    def _define_levels(self):
        return [
            { # Level 1: Intro to Green
                "layout": [
                    "################################",
                    "#S                             #",
                    "######                         #",
                    "#                              #",
                    "#           #####              #",
                    "#                              #",
                    "#                        E     #",
                    "#                     #####    #",
                    "#                              #",
                    "################################",
                ],
                "clones": {self.ORG_GREEN: 3},
            },
            { # Level 2: Intro to Red
                "layout": [
                    "################################",
                    "#S  DDD                        #",
                    "######D                        #",
                    "#     D                  E     #",
                    "#     D               #####    #",
                    "#     D                        #",
                    "#     D                        #",
                    "################################",
                ],
                "clones": {self.ORG_GREEN: 2, self.ORG_RED: 1},
            },
            { # Level 3: Intro to Blue
                "layout": [
                    "################################",
                    "#S                             #",
                    "#####        ##########        #",
                    "#                              #",
                    "#                              #",
                    "#                              #",
                    "#        ##########        E   #",
                    "#                          ### #",
                    "################################",
                ],
                "clones": {self.ORG_GREEN: 1, self.ORG_BLUE: 1},
            },
             { # Level 4: Blue + Yellow Interaction
                "layout": [
                    "################################",
                    "#S                             #",
                    "#####         #######          #",
                    "#                              #",
                    "#                              #",
                    "#                              #",
                    "#         #######          E   #",
                    "#                          ### #",
                    "################################",
                ],
                "clones": {self.ORG_BLUE: 1, self.ORG_YELLOW: 1},
            },
        ]

    def _load_level(self, level_idx):
        template = self.level_templates[level_idx % len(self.level_templates)]
        
        # Adjust layout to fit screen height
        layout = template["layout"][:] # Make a copy
        while len(layout) < self.GRID_HEIGHT:
            layout.insert(1, "#" * self.GRID_WIDTH) # Insert empty rows after first
        layout = layout[:self.GRID_HEIGHT]
        layout[-1] = "#" * self.GRID_WIDTH # Ensure bottom border

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        for y, row in enumerate(layout):
            for x, char in enumerate(row):
                if char == '#': self.grid[x, y] = self.TILE_WALL
                elif char == 'D': self.grid[x, y] = self.TILE_DESTRUCTIBLE
                elif char == 'S':
                    self.grid[x, y] = self.TILE_START
                    self.start_pos = (x, y)
                    self.cursor_pos = [x, y]
                elif char == 'E':
                    self.grid[x, y] = self.TILE_EXIT
                    self.exit_pos = (x, y)
        
        self.clone_counts = template["clones"].copy()
        self.unlocked_organisms = sorted([org for org in self.clone_counts.keys()])
        self.selected_organism_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._load_level(self.level_index)

        self.organisms = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Input ---
        # 1. Cycle Organism (on rising edge of shift)
        if shift_held and not self.last_shift_held and len(self.unlocked_organisms) > 1:
            self.selected_organism_idx = (self.selected_organism_idx + 1) % len(self.unlocked_organisms)
            # sfx: UI_CYCLE

        # 2. Move Cursor
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        new_cursor_x = max(0, min(self.GRID_WIDTH - 1, self.cursor_pos[0] + dx))
        new_cursor_y = max(0, min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + dy))
        self.cursor_pos = [new_cursor_x, new_cursor_y]

        # --- Pathfinding for reward calculation ---
        dist_before = self._calculate_path_distance()

        # 3. Clone Organism (on rising edge of space)
        event_reward = 0
        cloned = False
        if space_held and not self.last_space_held:
            cloned = True
            event_reward = self._handle_clone_action()
            reward += event_reward
            # sfx: CLONE_SUCCESS or CLONE_FAIL

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Update Game State ---
        self._update_animations()
        
        dist_after = self._calculate_path_distance()

        # --- Calculate Reward ---
        if cloned:
            if dist_after is not None and (dist_before is None or dist_after < dist_before):
                reward += 1.0 # Progress reward
            else:
                reward -= 0.1 # Wasted action penalty
        
        self.score += reward
        self.steps += 1
        
        # --- Check Termination ---
        terminated = self._check_termination(dist_after)
        if terminated:
            self.game_over = True
            if dist_after == 0: # Reached exit
                self.score += 100
                reward += 100
                self.level_index += 1 # Progress to next level on next reset
                # sfx: LEVEL_COMPLETE
            else: # Failed
                self.score -= 10
                reward -= 10
                # sfx: LEVEL_FAIL
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_clone_action(self):
        cx, cy = self.cursor_pos
        if not self.unlocked_organisms:
            return 0
        selected_type = self.unlocked_organisms[self.selected_organism_idx]

        # Check if placement is valid
        is_on_organism = any(o['pos'] == (cx, cy) for o in self.organisms)
        is_on_structure = self.grid[cx, cy] not in [self.TILE_EMPTY, self.TILE_DESTRUCTIBLE]
        has_clones = self.clone_counts.get(selected_type, 0) > 0

        if not has_clones or is_on_organism or is_on_structure:
            return 0

        self.clone_counts[selected_type] -= 1
        
        organism = {
            "type": selected_type,
            "pos": (cx, cy),
            "state": "active",
            "anim_progress": 0.0,
        }
        self.organisms.append(organism)
        
        # Trigger immediate effect and check for interactions
        return self._run_organism_effect(organism)

    def _run_organism_effect(self, organism):
        ox, oy = organism['pos']
        event_reward = 0
        
        if organism['type'] == self.ORG_RED:
            # sfx: BURST_EXPLODE
            self._create_particles(ox, oy, self.ORGANISM_DATA[self.ORG_RED]['color'], 20)
            # Destroy adjacent destructible blocks and green platforms
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = ox + dx, oy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    # Destroy destructible walls
                    if self.grid[nx, ny] == self.TILE_DESTRUCTIBLE:
                        self.grid[nx, ny] = self.TILE_EMPTY
                    # Check for Green+Red interaction (detrimental)
                    for other_org in self.organisms:
                        if other_org['type'] == self.ORG_GREEN and other_org['pos'] == (nx, ny):
                            other_org['state'] = 'destroyed'
                            event_reward = -2.0 # Detrimental interaction
                            self._create_particles(nx, ny, (200, 200, 200), 10)
                            # sfx: INTERACTION_NEGATIVE

        elif organism['type'] == self.ORG_YELLOW:
            # sfx: CHARGE_ACTIVATE
            self._create_particles(ox, oy, self.ORGANISM_DATA[self.ORG_YELLOW]['color'], 20, life=60)
            # Check for Blue+Yellow interaction (beneficial)
            for other_org in self.organisms:
                if other_org['type'] == self.ORG_BLUE and other_org['state'] == 'temporary':
                    # Check if adjacent
                    o2x, o2y = other_org['pos']
                    if abs(ox - o2x) + abs(oy - o2y) <= 1:
                        other_org['state'] = 'permanent'
                        event_reward = 5.0 # Beneficial interaction
                        # sfx: INTERACTION_POSITIVE
        
        # Cleanup red organism immediately after effect
        if organism['type'] == self.ORG_RED:
            organism['state'] = 'destroyed'
            
        return event_reward

    def _update_animations(self):
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] *= 0.98

        # Update organisms
        for org in self.organisms:
            if org['anim_progress'] < 1.0:
                org['anim_progress'] += 0.1 # Simple linear interpolation
            org['anim_progress'] = min(1.0, org['anim_progress'])
        
        # Remove destroyed organisms
        self.organisms = [o for o in self.organisms if o['state'] != 'destroyed']

    def _calculate_path_distance(self):
        q = deque([(self.start_pos, 0)])
        visited = {self.start_pos}
        
        walkable = self._get_walkable_grid()

        while q:
            (x, y), dist = q.popleft()

            if (x, y) == self.exit_pos:
                return dist

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and walkable.get((nx, ny), False):
                    visited.add((nx, ny))
                    q.append(((nx, ny), dist + 1))
        
        return None # No path found

    def _get_walkable_grid(self):
        walkable = {}
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] in [self.TILE_EMPTY, self.TILE_START, self.TILE_EXIT]:
                    walkable[(x, y)] = True
        
        for org in self.organisms:
            ox, oy = org['pos']
            if org['type'] == self.ORG_GREEN:
                platform_height = int(3 * org['anim_progress'])
                for i in range(1, platform_height + 1):
                    walkable[(ox, oy - i)] = True
            elif org['type'] == self.ORG_BLUE:
                # Find nearest walls
                left_wall = ox - 1
                while left_wall >= 0 and self.grid[left_wall, oy] == self.TILE_EMPTY:
                    left_wall -= 1
                right_wall = ox + 1
                while right_wall < self.GRID_WIDTH and self.grid[right_wall, oy] == self.TILE_EMPTY:
                    right_wall += 1
                
                if self.grid[left_wall, oy] == self.TILE_WALL and self.grid[right_wall, oy] == self.TILE_WALL:
                    for i in range(left_wall + 1, right_wall):
                         walkable[(i, oy)] = True

        return walkable

    def _check_termination(self, path_dist):
        # 1. Success condition
        if path_dist == 0:
            return True
        
        # 2. Step limit
        if self.steps >= 1000:
            return True
            
        # 3. Failure condition: no path and no useful clones left
        if path_dist is None:
            # Check if any clones are left that could potentially create a path
            if any(count > 0 for count in self.clone_counts.values()):
                return False # Still has moves to try
            else:
                return True # Out of clones and no path
        
        return False

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
            "level": self.level_index,
            "clones_remaining": self.clone_counts,
            "path_to_exit_exists": self._calculate_path_distance() is not None
        }

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

        # Draw grid cells
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                tile_type = self.grid[x, y]
                if tile_type == self.TILE_WALL:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                elif tile_type == self.TILE_DESTRUCTIBLE:
                    pygame.draw.rect(self.screen, self.COLOR_DESTRUCTIBLE, rect)
                elif tile_type == self.TILE_START:
                    pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, int(self.CELL_SIZE * 0.4), self.COLOR_START)
                elif tile_type == self.TILE_EXIT:
                    self._render_portal(rect.centerx, rect.centery)

        # Draw organisms and their effects
        self._render_organisms_and_effects()
        self._render_particles()
        self._render_cursor()

    def _render_portal(self, x, y):
        t = self.steps * 0.1
        for i in range(5, 0, -1):
            radius = int(self.CELL_SIZE * 0.2 + i * 2 + math.sin(t + i) * 2)
            alpha = 100 + i * 20
            color = (*self.COLOR_EXIT, alpha)
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _render_organisms_and_effects(self):
        for org in self.organisms:
            ox, oy = org['pos']
            cx, cy = int((ox + 0.5) * self.CELL_SIZE), int((oy + 0.5) * self.CELL_SIZE)
            color = self.ORGANISM_DATA[org['type']]['color']
            
            # Base organism shape
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, 6, color)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, 6, color)

            # Effects
            if org['type'] == self.ORG_GREEN:
                height = int(3 * self.CELL_SIZE * org['anim_progress'])
                rect = pygame.Rect(ox * self.CELL_SIZE + 4, (oy+1) * self.CELL_SIZE - height, self.CELL_SIZE-8, height)
                pygame.draw.rect(self.screen, color, rect)
            elif org['type'] == self.ORG_BLUE:
                left_wall = ox - 1
                while left_wall >= 0 and self.grid[left_wall, oy] == self.TILE_EMPTY: left_wall -= 1
                right_wall = ox + 1
                while right_wall < self.GRID_WIDTH and self.grid[right_wall, oy] == self.TILE_EMPTY: right_wall += 1

                if self.grid[left_wall, oy] == self.TILE_WALL and self.grid[right_wall, oy] == self.TILE_WALL:
                    start_x = (left_wall + 1) * self.CELL_SIZE
                    end_x = right_wall * self.CELL_SIZE
                    bridge_width = int((end_x - start_x) * org['anim_progress'])
                    bridge_y = int((oy + 0.5) * self.CELL_SIZE)
                    
                    alpha = 100 if org['state'] == 'temporary' else 255
                    glow_alpha = 150 if org['state'] == 'permanent' else 0

                    pygame.draw.line(self.screen, (*color, alpha), (start_x, bridge_y), (start_x + bridge_width, bridge_y), 4)
                    if glow_alpha > 0:
                        pygame.draw.line(self.screen, (*color, glow_alpha // 4), (start_x, bridge_y), (start_x + bridge_width, bridge_y), 10)


    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                color = (*p['color'], int(p['life'] * 4))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
    
    def _create_particles(self, grid_x, grid_y, color, count, life=30):
        cx = (grid_x + 0.5) * self.CELL_SIZE
        cy = (grid_y + 0.5) * self.CELL_SIZE
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                "pos": [cx, cy],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "radius": random.uniform(2, 5),
                "life": random.randint(life // 2, life),
                "color": color
            })

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        center_x = int((cx + 0.5) * self.CELL_SIZE)
        center_y = int((cy + 0.5) * self.CELL_SIZE)
        size = self.CELL_SIZE // 2
        
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, size, self.COLOR_CURSOR_GLOW)
        
        # Crosshair
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (center_x - size, center_y), (center_x + size, center_y), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (center_x, center_y - size), (center_x, center_y + size), 1)
    
    def _render_ui(self):
        # Score and Steps
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font_small.render(f"STEPS: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 30))
        level_text = self.font_small.render(f"LEVEL: {self.level_index + 1}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 50))
        
        # Clone counts and selection
        if not self.unlocked_organisms: return
        
        selected_type = self.unlocked_organisms[self.selected_organism_idx]
        
        ui_x = self.SCREEN_WIDTH - 150
        ui_y = self.SCREEN_HEIGHT - 20 - (len(self.unlocked_organisms) * 25)

        for i, org_type in enumerate(self.unlocked_organisms):
            data = self.ORGANISM_DATA[org_type]
            color = data['color']
            name = data['name']
            count = self.clone_counts[org_type]
            
            y_pos = ui_y + i * 25
            
            # Selection indicator
            if org_type == selected_type:
                rect = pygame.Rect(ui_x - 5, y_pos - 3, 140, 24)
                pygame.draw.rect(self.screen, (*self.COLOR_CURSOR, 50), rect, border_radius=4)
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 1, border_radius=4)

            # Icon
            pygame.gfxdraw.filled_circle(self.screen, ui_x + 10, y_pos + 9, 8, color)
            pygame.gfxdraw.aacircle(self.screen, ui_x + 10, y_pos + 9, 8, color)
            
            # Text
            text = self.font_small.render(f"{name}: {count}", True, self.COLOR_TEXT)
            self.screen.blit(text, (ui_x + 30, y_pos))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Manual play example
    # Set the video driver to a real one for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Symbiotic Clones")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Track rising edge for manual play
    last_shift_pressed = False
    last_space_pressed = False

    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # The environment's step function handles rising edge logic,
        # so we just pass the current key state.
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        obs, reward, term, trunc, info = env.step(action)
        
        if term:
            print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

        # Display the frame from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for smooth play

    env.close()