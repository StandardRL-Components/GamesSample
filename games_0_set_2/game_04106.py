
# Generated: 2025-08-28T01:26:27.650632
# Source Brief: brief_04106.md
# Brief Index: 4106

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. SHIFT to cycle crystal type. SPACE to place a crystal."
    )

    game_description = (
        "Redirect a laser beam through a crystal-filled cavern to hit the target. You have a limited number of crystals to place. Plan your moves carefully!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 12
    TILE_WIDTH, TILE_HEIGHT = 48, 24
    MAX_MOVES = 10
    MAX_LASER_BOUNCES = 100

    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_GRID_LINE = (30, 40, 60)
    COLOR_WALL = (80, 90, 110)
    COLOR_TARGET = (0, 255, 150)
    COLOR_TARGET_GLOW = (100, 255, 200)
    COLOR_LASER_ORIGIN = (255, 200, 0)
    COLOR_LASER = (255, 50, 50)
    COLOR_LASER_GLOW = (255, 150, 150)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_CURSOR_INVALID = (255, 0, 0)
    COLOR_UI_BG = (0, 0, 0, 150)
    COLOR_UI_TEXT = (220, 220, 240)

    # --- Crystal Definitions ---
    # Types: 0:Mirror(/), 1:Mirror(\), 2:Splitter(+), 3:Passthrough(O), 4:Blocker(X)
    CRYSTAL_COLORS = [
        (100, 150, 255),  # Blue
        (200, 100, 255),  # Purple
        (255, 200, 100),  # Orange
        (100, 255, 220),  # Cyan
        (200, 200, 200),  # Grey
    ]
    CRYSTAL_GLOWS = [
        (150, 200, 255),
        (220, 150, 255),
        (255, 220, 150),
        (150, 255, 240),
        (230, 230, 230),
    ]

    # Directions: 0:E(1,0), 1:N(0,-1), 2:W(-1,0), 3:S(0,1)
    DIR_VECTORS = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        self.iso_origin_x = self.SCREEN_WIDTH // 2
        self.iso_origin_y = 60
        
        self.reset()
        
        # This call is for development and can be removed in production
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.hit_target = False
        
        self.moves_remaining = self.MAX_MOVES
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_crystal_type = 0
        self.crystals = []
        self.laser_path = []
        self.laser_particles = []

        self.last_space_held = False
        self.last_shift_held = False
        
        self._generate_level()
        
        self._calculate_laser_path()
        self.last_distance_to_target = self._get_laser_dist_to_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        recalculate_laser = False

        # --- Handle Actions ---
        self._move_cursor(movement)

        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        if shift_pressed:
            # Sfx: UI_Cycle_Sound
            self.selected_crystal_type = (self.selected_crystal_type + 1) % len(self.CRYSTAL_COLORS)

        if space_pressed:
            if self._is_valid_placement(self.cursor_pos):
                # Sfx: Crystal_Place_Sound
                self.crystals.append({"pos": list(self.cursor_pos), "type": self.selected_crystal_type})
                self.moves_remaining -= 1
                reward -= 1
                recalculate_laser = True
            else:
                # Sfx: Error_Sound
                reward -= 0.1 # Small penalty for invalid placement attempt

        if recalculate_laser:
            self._calculate_laser_path()
            new_dist = self._get_laser_dist_to_target()
            # Reward for getting closer to the target
            dist_delta = self.last_distance_to_target - new_dist
            reward += dist_delta * 0.1
            self.last_distance_to_target = new_dist
        
        self.steps += 1
        self.score += reward

        # --- Update Game State ---
        self._update_particles()

        # --- Termination Check ---
        terminated = False
        if self.hit_target:
            # Sfx: Victory_Jingle
            self.score += 100
            reward += 100
            terminated = True
            self.game_over = True
        elif self.moves_remaining <= 0:
            # Sfx: Failure_Sound
            self.score -= 10
            reward -= 10
            terminated = True
            self.game_over = True
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_remaining": self.moves_remaining}

    # --- Game Logic ---
    def _generate_level(self):
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.laser_origin_pos = [0, self.GRID_HEIGHT // 2]
        self.laser_origin_dir_idx = 0 # East
        self.target_pos = [self.GRID_WIDTH - 1, self.GRID_HEIGHT // 2]

        # Place some walls, ensuring they are not on the origin or target
        for _ in range(self.GRID_WIDTH * self.GRID_HEIGHT // 8):
            x, y = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
            if [x,y] != self.laser_origin_pos and [x,y] != self.target_pos and self.grid[x,y] == 0:
                self.grid[x,y] = 1 # 1 for wall

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

    def _is_valid_placement(self, pos):
        if self.grid[pos[0], pos[1]] == 1: return False # On a wall
        if pos == self.laser_origin_pos or pos == self.target_pos: return False # On origin/target
        if any(c['pos'] == pos for c in self.crystals): return False # On another crystal
        return True

    def _calculate_laser_path(self):
        self.laser_path.clear()
        self.hit_target = False
        
        beams = [ (self.laser_origin_pos, self.laser_origin_dir_idx) ]
        bounces = 0

        visited_states = set()

        while beams and bounces < self.MAX_LASER_BOUNCES:
            start_pos, dir_idx = beams.pop(0)
            
            state_tuple = (tuple(start_pos), dir_idx)
            if state_tuple in visited_states: continue
            visited_states.add(state_tuple)
            
            direction = self.DIR_VECTORS[dir_idx]
            current_pos = list(start_pos)
            
            for i in range(max(self.GRID_WIDTH, self.GRID_HEIGHT)):
                bounces += 1
                next_pos = [current_pos[0] + direction[0], current_pos[1] + direction[1]]

                # Check boundaries
                if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                    self.laser_path.append((current_pos, next_pos))
                    break

                # Check for target
                if next_pos == self.target_pos:
                    self.laser_path.append((current_pos, next_pos))
                    self.hit_target = True
                    # Sfx: Laser_Hit_Target
                    return

                # Check for wall
                if self.grid[next_pos[0], next_pos[1]] == 1:
                    self.laser_path.append((current_pos, next_pos))
                    # Reflection logic
                    if direction[0] != 0: new_dir_idx = (dir_idx + 2) % 4 # Horizontal hit
                    else: new_dir_idx = (dir_idx + 2) % 4 # Vertical hit (same logic for cardinal)
                    beams.append((next_pos, new_dir_idx))
                    break

                # Check for crystal
                crystal_hit = next((c for c in self.crystals if c['pos'] == next_pos), None)
                if crystal_hit:
                    self.laser_path.append((current_pos, next_pos))
                    # Sfx: Laser_Hit_Crystal
                    new_beams = self._refract_laser(dir_idx, crystal_hit['type'], next_pos)
                    beams.extend(new_beams)
                    break
                
                current_pos = next_pos

    def _refract_laser(self, dir_idx, crystal_type, pos):
        # 0:E, 1:N, 2:W, 3:S
        # Type 0: Mirror /
        if crystal_type == 0:
            if dir_idx == 0: return [(pos, 1)] # E -> N
            if dir_idx == 1: return [(pos, 0)] # N -> E
            if dir_idx == 2: return [(pos, 3)] # W -> S
            if dir_idx == 3: return [(pos, 2)] # S -> W
        # Type 1: Mirror \
        elif crystal_type == 1:
            if dir_idx == 0: return [(pos, 3)] # E -> S
            if dir_idx == 1: return [(pos, 2)] # N -> W
            if dir_idx == 2: return [(pos, 1)] # W -> N
            if dir_idx == 3: return [(pos, 0)] # S -> E
        # Type 2: Splitter +
        elif crystal_type == 2:
            if dir_idx in [0, 2]: return [(pos, 1), (pos, 3)] # Horizontal -> Vertical
            if dir_idx in [1, 3]: return [(pos, 0), (pos, 2)] # Vertical -> Horizontal
        # Type 3: Passthrough
        elif crystal_type == 3:
            return [(pos, dir_idx)]
        # Type 4: Blocker
        elif crystal_type == 4:
            return [(pos, (dir_idx + 2) % 4)] # Reflect back
        return []

    def _get_laser_dist_to_target(self):
        if not self.laser_path:
            p = self.laser_origin_pos
            return abs(p[0] - self.target_pos[0]) + abs(p[1] - self.target_pos[1])
        
        last_point = self.laser_path[-1][1]
        return abs(last_point[0] - self.target_pos[0]) + abs(last_point[1] - self.target_pos[1])

    def _update_particles(self):
        if self.np_random.random() < 0.8: # Spawn rate
            for start, end in self.laser_path:
                if self.np_random.random() < 0.2:
                    start_screen = self._grid_to_screen_center(start[0], start[1])
                    self.laser_particles.append([list(start_screen), list(end), 0.0, 1.0]) # pos, target, progress, lifespan

        new_particles = []
        for p in self.laser_particles:
            p[2] += 0.1 # progress speed
            p[3] -= 0.02 # lifespan decay
            if p[3] > 0 and p[2] <= 1.0:
                new_particles.append(p)
        self.laser_particles = new_particles

    # --- Rendering ---
    def _grid_to_screen_center(self, x, y):
        screen_x = self.iso_origin_x + (x - y) * self.TILE_WIDTH / 2
        screen_y = self.iso_origin_y + (x + y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y + self.TILE_HEIGHT / 2)

    def _draw_iso_tile(self, surface, x, y, color):
        points = [
            self._grid_to_screen_center(x, y),
            self._grid_to_screen_center(x + 1, y),
            self._grid_to_screen_center(x + 1, y + 1),
            self._grid_to_screen_center(x, y + 1)
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                self._draw_iso_tile(self.screen, x, y, self.COLOR_GRID_LINE)
        
        # Draw walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == 1:
                    center = self._grid_to_screen_center(x,y)
                    points = [
                        (center[0], center[1] - self.TILE_HEIGHT),
                        (center[0] + self.TILE_WIDTH/2, center[1] - self.TILE_HEIGHT/2),
                        (center[0], center[1]),
                        (center[0] - self.TILE_WIDTH/2, center[1] - self.TILE_HEIGHT/2),
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_WALL)
                    pygame.gfxdraw.aapolygon(self.screen, points, (0,0,0,50))

        # Draw target and origin
        tx, ty = self._grid_to_screen_center(*self.target_pos)
        pygame.gfxdraw.filled_circle(self.screen, tx, ty, 8, self.COLOR_TARGET)
        pygame.gfxdraw.aacircle(self.screen, tx, ty, 8, self.COLOR_TARGET_GLOW)
        ox, oy = self._grid_to_screen_center(*self.laser_origin_pos)
        pygame.gfxdraw.filled_circle(self.screen, ox, oy, 6, self.COLOR_LASER_ORIGIN)
        pygame.gfxdraw.aacircle(self.screen, ox, oy, 6, self.COLOR_LASER_ORIGIN)

        # Draw placed crystals
        for crystal in self.crystals:
            cx, cy = self._grid_to_screen_center(*crystal['pos'])
            color = self.CRYSTAL_COLORS[crystal['type']]
            glow = self.CRYSTAL_GLOWS[crystal['type']]
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, 7, glow)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, 5, color)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, 7, glow)

        # Draw laser path
        for start_grid, end_grid in self.laser_path:
            start_px = self._grid_to_screen_center(*start_grid)
            end_px = self._grid_to_screen_center(*end_grid)
            pygame.draw.aaline(self.screen, self.COLOR_LASER_GLOW, start_px, end_px, blend=1)
            pygame.draw.line(self.screen, self.COLOR_LASER, start_px, end_px, 2)
            
        # Draw laser particles
        for p in self.laser_particles:
            start_px = p[0]
            end_grid_px = self._grid_to_screen_center(*p[1])
            x = int(start_px[0] + (end_grid_px[0] - start_px[0]) * p[2])
            y = int(start_px[1] + (end_grid_px[1] - start_px[1]) * p[2])
            size = int(p[3] * 3)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, x, y, size, self.COLOR_LASER_GLOW)

        # Draw cursor
        cursor_color = self.COLOR_CURSOR if self._is_valid_placement(self.cursor_pos) else self.COLOR_CURSOR_INVALID
        self._draw_iso_tile(self.screen, self.cursor_pos[0], self.cursor_pos[1], cursor_color)

    def _render_ui(self):
        ui_bar = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        ui_bar.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_bar, (0, self.SCREEN_HEIGHT - 50))
        
        # Moves remaining
        moves_text = self.font_large.render(f"CRYSTALS: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (15, self.SCREEN_HEIGHT - 38))
        
        # Selected crystal
        sel_text = self.font_small.render("Selected:", True, self.COLOR_UI_TEXT)
        self.screen.blit(sel_text, (self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 38))
        
        cx, cy = self.SCREEN_WIDTH - 60, self.SCREEN_HEIGHT - 25
        color = self.CRYSTAL_COLORS[self.selected_crystal_type]
        glow = self.CRYSTAL_GLOWS[self.selected_crystal_type]
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, 12, glow)
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, 10, color)
        pygame.gfxdraw.aacircle(self.screen, cx, cy, 12, glow)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            msg = "TARGET HIT!" if self.hit_target else "OUT OF CRYSTALS"
            color = self.COLOR_TARGET if self.hit_target else (255, 80, 80)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def validate_implementation(self):
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

if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Laser Cavern")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                action.fill(0)

        if not terminated:
            keys = pygame.key.get_pressed()
            
            mov = 0
            if keys[pygame.K_UP]: mov = 1
            elif keys[pygame.K_DOWN]: mov = 2
            elif keys[pygame.K_LEFT]: mov = 3
            elif keys[pygame.K_RIGHT]: mov = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = np.array([mov, space, shift])
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    pygame.quit()