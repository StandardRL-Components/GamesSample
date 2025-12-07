
# Generated: 2025-08-27T21:53:42.395051
# Source Brief: brief_02936.md
# Brief Index: 2936

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place a crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a crystal cavern, placing crystals to redirect light beams and illuminate all targets."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 24
    GRID_HEIGHT = 16
    TILE_WIDTH = 48
    TILE_HEIGHT = TILE_WIDTH // 2
    TILE_DEPTH = 20  # For wall height

    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    NUM_TARGETS = 5
    INITIAL_CRYSTALS = 10
    MAX_STEPS = 1000

    # Element IDs for the grid
    EMPTY = 0
    WALL = 1
    
    # Colors
    COLOR_BG = (20, 20, 40)
    COLOR_WALL = (70, 70, 85)
    COLOR_WALL_SIDE = (50, 50, 65)
    COLOR_WALL_TOP = (90, 90, 105)
    COLOR_TARGET = (255, 215, 0)
    COLOR_TARGET_LIT = (255, 255, 180)
    COLOR_CRYSTAL = (0, 220, 255)
    COLOR_CRYSTAL_FACET = (180, 240, 255)
    COLOR_LIGHT_BEAM = (255, 255, 255)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_CRYSTAL = (0, 180, 220)

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT // 4)

        # Initialize state variables
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.light_source_pos = (0, 0)
        self.light_source_dir = (1, 0)
        self.targets = []
        self.crystals = []
        self.cursor_pos = (0, 0)
        self.num_crystals_remaining = 0
        self.illuminated_targets = set()
        self.light_path = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.num_crystals_remaining = self.INITIAL_CRYSTALS
        self.crystals = []
        self.illuminated_targets = set()
        
        self._procedural_generation()
        
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        
        self._calculate_light_path()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = -0.01  # Small time penalty

        # 1. Handle cursor movement
        px, py = self.cursor_pos
        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right
        self.cursor_pos = (
            np.clip(px, 0, self.GRID_WIDTH - 1),
            np.clip(py, 0, self.GRID_HEIGHT - 1)
        )

        # 2. Handle crystal placement
        placed_crystal = False
        if space_held:
            can_place = (self.num_crystals_remaining > 0 and
                         self.grid[self.cursor_pos[1], self.cursor_pos[0]] == self.EMPTY and
                         self.cursor_pos not in self.crystals and
                         self.cursor_pos not in self.targets and
                         self.cursor_pos != self.light_source_pos)
            if can_place:
                # sfx: Crystal place sound
                self.crystals.append(self.cursor_pos)
                self.num_crystals_remaining -= 1
                placed_crystal = True
                reward -= 0.1

        # 3. Recalculate game state if a crystal was placed
        if placed_crystal:
            prev_illuminated_count = len(self.illuminated_targets)
            self._calculate_light_path()
            newly_illuminated = len(self.illuminated_targets) - prev_illuminated_count
            
            if newly_illuminated > 0:
                # sfx: Target illuminated sound
                reward += newly_illuminated * 1.0

            if len(self.illuminated_targets) == self.NUM_TARGETS and prev_illuminated_count < self.NUM_TARGETS:
                # sfx: Final target chime
                reward += 10.0

        # 4. Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.win:
                # sfx: Win fanfare
                reward += 100.0
            else:
                # sfx: Loss sound
                reward -= 100.0
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _procedural_generation(self):
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        # Add border walls
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL

        # Add random internal walls
        num_walls = self.np_random.integers(5, 15)
        for _ in range(num_walls):
            start_x = self.np_random.integers(2, self.GRID_WIDTH - 3)
            start_y = self.np_random.integers(2, self.GRID_HEIGHT - 3)
            length = self.np_random.integers(2, 6)
            direction = self.np_random.choice([0, 1]) # 0 for horizontal, 1 for vertical
            if direction == 0:
                end_x = min(self.GRID_WIDTH - 2, start_x + length)
                self.grid[start_y, start_x:end_x] = self.WALL
            else:
                end_y = min(self.GRID_HEIGHT - 2, start_y + length)
                self.grid[start_y:end_y, start_x] = self.WALL

        # Get all valid empty spots
        empty_spots = self._get_empty_spots()
        if len(empty_spots) < self.NUM_TARGETS + 1:
            return self.reset() # Failsafe, regenerate if not enough space

        # Place light source
        idx = self.np_random.integers(len(empty_spots))
        self.light_source_pos = empty_spots.pop(idx)
        self.light_source_dir = random.choice([(1,0), (-1,0), (0,1), (0,-1)])

        # Place targets, ensuring one is solvable with one crystal
        self.targets = []
        
        # Calculate where a single crystal can redirect light to
        initial_path = self._trace_beam(self.light_source_pos, self.light_source_dir, [])
        reachable_with_one_crystal = set()
        for point in initial_path:
            for new_dir in [(0, self.light_source_dir[0]), (self.light_source_dir[1], 0)]: # simplified redirection
                if new_dir == (0,0): continue
                redirected_path = self._trace_beam(point, new_dir, [point])
                for p in redirected_path:
                    if p in empty_spots: reachable_with_one_crystal.add(p)

        # If no spot is reachable, regenerate
        if not reachable_with_one_crystal:
            return self._procedural_generation()

        # Place one target in a guaranteed solvable spot
        solvable_target_pos = random.choice(list(reachable_with_one_crystal))
        self.targets.append(solvable_target_pos)
        empty_spots.remove(solvable_target_pos)
        
        # Place remaining targets
        self.np_random.shuffle(empty_spots)
        for _ in range(self.NUM_TARGETS - 1):
            if not empty_spots: break
            self.targets.append(empty_spots.pop(0))

    def _get_empty_spots(self):
        return [(x, y) for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH) if self.grid[y, x] == self.EMPTY]

    def _check_termination(self):
        if self.game_over:
            return True
        
        if len(self.illuminated_targets) == self.NUM_TARGETS:
            self.game_over = True
            self.win = True
            return True
        
        if self.num_crystals_remaining == 0 and len(self.illuminated_targets) < self.NUM_TARGETS:
             self.game_over = True
             return True

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        
        return False

    def _calculate_light_path(self):
        self.illuminated_targets.clear()
        self.light_path = self._trace_beam(self.light_source_pos, self.light_source_dir, self.crystals)
        
        for pos in self.light_path:
            if pos in self.targets:
                self.illuminated_targets.add(pos)

    def _trace_beam(self, start_pos, start_dir, crystals_list):
        path = [start_pos]
        pos = start_pos
        direction = start_dir
        
        for _ in range(self.GRID_WIDTH * self.GRID_HEIGHT): # Safety break
            next_pos = (pos[0] + direction[0], pos[1] + direction[1])
            
            if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                break
            
            if self.grid[next_pos[1], next_pos[0]] == self.WALL:
                break
            
            pos = next_pos
            path.append(pos)
            
            if pos in crystals_list:
                # sfx: Light redirect sound
                # A horizontal beam (dx, 0) reflects to a vertical beam (0, dx)
                # A vertical beam (0, dy) reflects to a horizontal beam (dy, 0)
                dx, dy = direction
                if dx != 0: direction = (0, dx)
                elif dy != 0: direction = (dy, 0)
        
        return path

    def _grid_to_iso(self, x, y):
        iso_x = self.origin_x + (x - y) * (self.TILE_WIDTH / 2)
        iso_y = self.origin_y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(iso_x), int(iso_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render from back to front
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_pos = self._grid_to_iso(x, y)
                
                if self.grid[y, x] == self.WALL:
                    self._draw_iso_cube(screen_pos, self.TILE_WIDTH, self.TILE_HEIGHT, self.TILE_DEPTH)
                else:
                    if (x, y) in self.targets:
                        is_lit = (x, y) in self.illuminated_targets
                        self._draw_target(screen_pos, is_lit)
                    
                    if (x, y) == self.light_source_pos:
                        self._draw_light_source(screen_pos)

                    if (x, y) in self.crystals:
                        self._draw_crystal(screen_pos)
        
        self._draw_cursor()
        self._draw_light_beam()

    def _draw_iso_cube(self, pos, w, h, d):
        x, y = pos
        w_half, h_half = w // 2, h // 2
        
        p_top = (x, y - d)
        p_left = (x - w_half, y - h_half - d)
        p_right = (x + w_half, y - h_half - d)
        p_bottom = (x, y - h + h_half - d)
        
        pygame.gfxdraw.filled_polygon(self.screen, [p_top, p_right, p_bottom, p_left], self.COLOR_WALL_TOP)
        
        p_bottom_left = (x - w_half, y - h_half)
        pygame.gfxdraw.filled_polygon(self.screen, [p_left, p_bottom, p_bottom_left, (p_left[0], p_bottom_left[1])], self.COLOR_WALL_SIDE)
        
        p_bottom_right = (x + w_half, y - h_half)
        pygame.gfxdraw.filled_polygon(self.screen, [p_right, p_bottom, p_bottom_right, (p_right[0], p_bottom_right[1])], self.COLOR_WALL)

    def _draw_target(self, pos, is_lit):
        x, y = pos
        color = self.COLOR_TARGET_LIT if is_lit else self.COLOR_TARGET
        radius = self.TILE_HEIGHT // 2
        
        if is_lit:
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            glow_radius = int(radius * (1.5 + pulse * 0.5))
            
            temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.COLOR_TARGET_LIT, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (x - glow_radius, y - glow_radius))

        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)

    def _draw_light_source(self, pos):
        x, y = pos
        w_half, h_half = self.TILE_WIDTH // 2, self.TILE_HEIGHT // 2
        points = [
            (x, y - h_half), (x + w_half, y), (x, y + h_half), (x - w_half, y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_TARGET_LIT)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TARGET_LIT)

    def _draw_crystal(self, pos):
        x, y = pos
        w_half, h_half = (self.TILE_WIDTH-10) // 2, (self.TILE_HEIGHT-5) // 2
        points = [
            (x, y - h_half), (x + w_half, y), (x, y + h_half), (x - w_half, y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL_FACET)
        
        pygame.draw.aaline(self.screen, self.COLOR_CRYSTAL_FACET, (x-w_half//2, y-h_half//2), (x+w_half//2, y+h_half//2))

    def _draw_cursor(self):
        if self.game_over: return
        x, y = self._grid_to_iso(*self.cursor_pos)
        w_half, h_half = self.TILE_WIDTH // 2, self.TILE_HEIGHT // 2
        points = [
            (x, y - h_half), (x + w_half, y), (x, y + h_half), (x - w_half, y)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, points, 2)

    def _draw_light_beam(self):
        if len(self.light_path) < 2: return
        
        pulse = (math.sin(self.steps * 0.3) + 1) / 2
        alpha = 150 + int(pulse * 105)
        
        iso_path = [self._grid_to_iso(p[0], p[1]) for p in self.light_path]
        
        pygame.draw.lines(self.screen, (*self.COLOR_LIGHT_BEAM, alpha // 4), False, iso_path, width=12)
        pygame.draw.lines(self.screen, (*self.COLOR_LIGHT_BEAM, alpha // 2), False, iso_path, width=6)
        pygame.draw.lines(self.screen, self.COLOR_LIGHT_BEAM, False, iso_path, width=2)

    def _render_ui(self):
        crystal_text = self.font_ui.render(f"Crystals:", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (10, 10))
        
        for i in range(self.num_crystals_remaining):
            x = 110 + i * 20
            y = 12
            points = [(x,y+8), (x+8,y+16), (x+16,y+8), (x+8,y)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_UI_CRYSTAL)

        score_text = self.font_ui.render(f"Score: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)
        
        target_text = self.font_ui.render(f"Targets: {len(self.illuminated_targets)}/{self.NUM_TARGETS}", True, self.COLOR_UI_TEXT)
        target_rect = target_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 35))
        self.screen.blit(target_text, target_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_TARGET_LIT if self.win else (200, 50, 50)
            
            text = self.font_game_over.render(msg, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_left": self.num_crystals_remaining,
            "illuminated_targets": len(self.illuminated_targets)
        }

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    movement_key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4
    }
    
    running = True
    while running:
        action = [0, 0, 0] # move, space, shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in movement_key_map:
                    action[0] = movement_key_map[event.key]
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False

        if not done and any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
        
    env.close()