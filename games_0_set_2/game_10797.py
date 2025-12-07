import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:59:42.246483
# Source Brief: brief_00797.md
# Brief Index: 797
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
    GameEnv: Sonic Forest
    Match musical instrument tiles to terraform a sonic forest and ascend.
    The agent controls a cursor to select and match adjacent tiles.
    Successful matches grow the tree, allowing for higher ascent.
    The goal is to reach the top of the forest.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Match musical instrument tiles to grow a sonic forest. Ascend the tree by clearing matching pairs and reach the top to win."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to select a tile, then move and press space on an adjacent matching tile. Press shift to deselect."
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 40
        self.GRID_COLS = self.WIDTH // self.TILE_SIZE
        self.GRID_ROWS = self.HEIGHT // self.TILE_SIZE
        self.MAX_STEPS = 2000
        self.WIN_HEIGHT = 1000
        self.INITIAL_BRANCH_DENSITY = 0.20
        self.BRANCH_DENSITY_INCREASE_INTERVAL = 50
        self.BRANCH_DENSITY_INCREASE_AMOUNT = 0.05
        self.INSTRUMENT_UNLOCK_INTERVAL = 100

        # --- Colors ---
        self.COLOR_BG_TOP = (20, 10, 40)
        self.COLOR_BG_BOTTOM = (60, 40, 90)
        self.COLOR_BRANCH = (80, 50, 30)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 255)
        self.COLOR_TEXT_SHADOW = (20, 20, 30)
        
        self.INSTRUMENT_COLORS = [
            (50, 200, 50),   # 0: Percussion (Green)
            (50, 150, 255),  # 1: Wind (Blue)
            (255, 80, 80),   # 2: String (Red)
            (250, 180, 50),  # 3: Brass (Orange)
            (200, 100, 255)  # 4: Keys (Purple)
        ]

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_tile = pygame.font.SysFont("Arial", 20, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.selected_tile_pos = None
        self.height = 0
        self.last_milestone_height = 0
        self.camera_y = 0.0
        self.target_camera_y = 0.0
        self.tiles = {}
        self.branches = {}
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.current_instrument_types = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS - 3]
        self.selected_tile_pos = None
        
        self.height = 0
        self.last_milestone_height = 0
        
        self.tiles = {}
        self.branches = {}
        self.particles = []
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self._initialize_world()
        self.camera_y = self.target_camera_y = (self.GRID_ROWS - 1) * self.TILE_SIZE - self.HEIGHT + self.TILE_SIZE
        self._update_player_height()
        self._ensure_matches()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        if self.game_over:
            return self._get_observation(), reward, True, False, self._get_info()

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward += self._handle_input(movement, space_held, shift_held)
        
        self._update_particles()
        self._update_camera()

        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.height >= self.WIN_HEIGHT:
                reward += 100 # Goal-oriented reward
            elif self._count_possible_matches() == 0:
                reward += -100 # Failure penalty
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        # Wrap cursor around screen edges
        self.cursor_pos[0] %= self.GRID_COLS
        self.cursor_pos[1] %= self.GRID_ROWS
        
        cursor_world_pos = self._screen_to_world(self.cursor_pos)

        # --- Deselect (Shift) ---
        if shift_held and not self.last_shift_held:
            if self.selected_tile_pos:
                # Sound: Deselect
                self.selected_tile_pos = None
        
        # --- Select/Match (Space) ---
        if space_held and not self.last_space_held:
            if self.selected_tile_pos is None:
                if cursor_world_pos in self.tiles:
                    # Sound: Select
                    self.selected_tile_pos = cursor_world_pos
            else:
                # Attempt match
                reward += self._attempt_match(self.selected_tile_pos, cursor_world_pos)
                self.selected_tile_pos = None

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        return reward

    def _attempt_match(self, pos1, pos2):
        if pos1 == pos2: return 0 # Cannot match with self
        
        is_adjacent = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1
        tiles_exist = pos1 in self.tiles and pos2 in self.tiles
        
        if is_adjacent and tiles_exist and self.tiles[pos1]['type'] == self.tiles[pos2]['type']:
            # --- Successful Match ---
            # Sound: Match success
            match_type = self.tiles[pos1]['type']
            
            del self.tiles[pos1]
            del self.tiles[pos2]
            
            self._terraform(pos1, pos2)
            self._create_particles(pos1, pos2, match_type)
            
            height_before = self.height
            self._update_player_height()
            
            reward = 1.0 # Base reward for a match
            
            # Milestone reward
            if self.height > height_before:
                new_milestone = math.floor(self.height / self.INSTRUMENT_UNLOCK_INTERVAL)
                old_milestone = math.floor(self.last_milestone_height / self.INSTRUMENT_UNLOCK_INTERVAL)
                if new_milestone > old_milestone:
                    reward += 5.0
            self.last_milestone_height = self.height

            self._ensure_matches()
            return reward
        else:
            # Sound: Match fail
            return 0

    def _terraform(self, pos1, pos2):
        # Create a new branch at the midpoint, if it connects to an existing branch
        mx, my = (pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2
        mid_pos = (int(round(mx)), int(round(my)))
        
        # A branch can only grow if it's supported by another branch below it
        is_supported = False
        for dx, dy in [(0, 1), (-1, 1), (1, 1)]: # Check below and diagonally below
            if (mid_pos[0] + dx, mid_pos[1] + dy) in self.branches:
                is_supported = True
                break
        
        if is_supported:
            self.branches[mid_pos] = {'growth': 1.0}

    def _update_player_height(self):
        # Use BFS to find the highest reachable branch
        q = deque()
        visited = set()
        
        # Start search from the base ground level
        for x in range(self.GRID_COLS):
            pos = (x, self.GRID_ROWS - 1)
            if pos in self.branches:
                q.append(pos)
                visited.add(pos)

        min_y = self.GRID_ROWS - 1
        while q:
            x, y = q.popleft()
            min_y = min(min_y, y)
            
            # Check neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                nx, ny = x + dx, y + dy
                neighbor_pos = (nx, ny)
                if neighbor_pos in self.branches and neighbor_pos not in visited:
                    visited.add(neighbor_pos)
                    q.append(neighbor_pos)
        
        # Height is inverse of y-coordinate
        self.height = (self.GRID_ROWS - 1 - min_y)
        self.target_camera_y = min_y * self.TILE_SIZE - self.HEIGHT / 2

    def _initialize_world(self):
        # Create the starting ground/trunk
        for x in range(self.GRID_COLS):
            self.branches[(x, self.GRID_ROWS - 1)] = {'growth': 1.0}
        
        # Populate initial area with tiles and random branches
        self._generate_world_chunk(y_start=-self.GRID_ROWS * 2, y_end=self.GRID_ROWS)

    def _generate_world_chunk(self, y_start, y_end):
        # Determine available instrument types based on height
        num_types = min(len(self.INSTRUMENT_COLORS), 2 + math.floor(self.height / self.INSTRUMENT_UNLOCK_INTERVAL))
        self.current_instrument_types = list(range(num_types))
        
        # Determine branch density
        density_milestones = math.floor(self.height / self.BRANCH_DENSITY_INCREASE_INTERVAL)
        branch_density = self.INITIAL_BRANCH_DENSITY + density_milestones * self.BRANCH_DENSITY_INCREASE_AMOUNT
        
        for y in range(y_start, y_end):
            for x in range(self.GRID_COLS):
                pos = (x, y)
                if pos in self.tiles or pos in self.branches: continue

                # Place random obstructing branches (less dense)
                if self.np_random.random() < branch_density and y < self.GRID_ROWS - 2:
                    self.branches[pos] = {'growth': 1.0}
                # Place tiles on empty spaces
                elif self.np_random.random() < 0.7:
                    self.tiles[pos] = {'type': self.np_random.choice(self.current_instrument_types)}

    def _ensure_matches(self):
        if self._count_possible_matches() > 0:
            return

        # Force a match if none exist
        empty_adjacent_pairs = []
        for y in range(int(self.camera_y / self.TILE_SIZE), int((self.camera_y + self.HEIGHT) / self.TILE_SIZE) + 1):
            for x in range(self.GRID_COLS):
                pos1 = (x, y)
                if pos1 not in self.tiles and pos1 not in self.branches:
                    # Check right neighbor
                    pos2 = (x + 1, y)
                    if (x + 1 < self.GRID_COLS and pos2 not in self.tiles and pos2 not in self.branches):
                        empty_adjacent_pairs.append((pos1, pos2))
                    # Check down neighbor
                    pos3 = (x, y + 1)
                    if pos3 not in self.tiles and pos3 not in self.branches:
                         empty_adjacent_pairs.append((pos1, pos3))
        
        if empty_adjacent_pairs:
            pos1, pos2 = random.choice(empty_adjacent_pairs) # Use random instead of np_random if np_random is not a generator
            match_type = self.np_random.choice(self.current_instrument_types)
            self.tiles[pos1] = {'type': match_type}
            self.tiles[pos2] = {'type': match_type}

    def _count_possible_matches(self):
        count = 0
        checked_tiles = set()
        for pos, tile in self.tiles.items():
            if pos in checked_tiles: continue
            
            # Check right neighbor
            neighbor_r = (pos[0] + 1, pos[1])
            if neighbor_r in self.tiles and self.tiles[neighbor_r]['type'] == tile['type']:
                count += 1
            
            # Check down neighbor
            neighbor_d = (pos[0], pos[1] + 1)
            if neighbor_d in self.tiles and self.tiles[neighbor_d]['type'] == tile['type']:
                count += 1
            
            checked_tiles.add(pos)
        return count

    def _check_termination(self):
        if self.height >= self.WIN_HEIGHT:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        if self._count_possible_matches() == 0 and not self._can_add_tiles():
            return True
        return False

    def _can_add_tiles(self):
        # Check if there is any empty space left on the screen to add a guaranteed match
        for y in range(int(self.camera_y / self.TILE_SIZE), int((self.camera_y + self.HEIGHT) / self.TILE_SIZE) + 1):
            for x in range(self.GRID_COLS):
                if (x, y) not in self.tiles and (x, y) not in self.branches:
                    return True
        return False

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "height": self.height}

    def _world_to_screen(self, pos):
        x, y = pos
        screen_x = x * self.TILE_SIZE
        screen_y = (y * self.TILE_SIZE) - self.camera_y
        return int(screen_x), int(screen_y)

    def _screen_to_world(self, pos):
        x, y = pos
        world_x = x
        world_y = int((y * self.TILE_SIZE + self.camera_y) / self.TILE_SIZE)
        return world_x, world_y

    def _render_game(self):
        # --- Background Gradient ---
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = [int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio) for i in range(3)]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # --- Render Branches ---
        for pos, branch in self.branches.items():
            sx, sy = self._world_to_screen(pos)
            if -self.TILE_SIZE < sx < self.WIDTH and -self.TILE_SIZE < sy < self.HEIGHT:
                rect = pygame.Rect(sx, sy, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_BRANCH, rect)

        # --- Render Tiles ---
        instrument_symbols = ['▲', '●', '■', '★', '♪']
        for pos, tile in self.tiles.items():
            sx, sy = self._world_to_screen(pos)
            if -self.TILE_SIZE < sx < self.WIDTH and -self.TILE_SIZE < sy < self.HEIGHT:
                rect = pygame.Rect(sx + 2, sy + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4)
                color = self.INSTRUMENT_COLORS[tile['type']]
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
                
                symbol = instrument_symbols[tile['type']]
                text_surf = self.font_tile.render(symbol, True, (255,255,255))
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)

        # --- Render Selection Highlight ---
        if self.selected_tile_pos:
            sx, sy = self._world_to_screen(self.selected_tile_pos)
            rect = pygame.Rect(sx, sy, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=7)

        # --- Render Cursor ---
        cursor_screen_pos = (self.cursor_pos[0] * self.TILE_SIZE, self.cursor_pos[1] * self.TILE_SIZE)
        rect = pygame.Rect(cursor_screen_pos[0], cursor_screen_pos[1], self.TILE_SIZE, self.TILE_SIZE)
        
        # Pulsing effect for cursor
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        alpha = int(50 + pulse * 50)
        glow_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_CURSOR, alpha), glow_surface.get_rect(), border_radius=7)
        self.screen.blit(glow_surface, rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=7)

        # --- Render Particles ---
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['size']), p['color'])

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, pos, font, color, shadow_color):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Display Height
        height_text = f"Height: {self.height}m"
        draw_text(height_text, (10, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Display Score
        score_text = f"Score: {int(self.score)}"
        draw_text(score_text, (10, 40), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

    def _update_camera(self):
        # Smooth camera movement
        self.camera_y += (self.target_camera_y - self.camera_y) * 0.05

    def _create_particles(self, pos1, pos2, tile_type):
        sx1, sy1 = self._world_to_screen(pos1)
        sx2, sy2 = self._world_to_screen(pos2)
        mx, my = (sx1 + sx2) / 2 + self.TILE_SIZE / 2, (sy1 + sy2) / 2 + self.TILE_SIZE / 2
        color = self.INSTRUMENT_COLORS[tile_type]
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': mx, 'y': my,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'size': self.np_random.uniform(3, 7),
                'life': 40,
                'color': (*color, 200)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['size'] *= 0.95
            if p['life'] <= 0 or p['size'] < 1:
                self.particles.remove(p)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver if running locally
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sonic Forest")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # --- Manual Control Mapping ---
    # ARROWS: move, SPACE: select/match, LSHIFT: deselect
    
    while not terminated:
        movement, space, shift = 0, 0, 0 # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, term, trunc, info = env.step(action)
        terminated = term

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for smooth play

    pygame.quit()
    print(f"Game Over! Final Score: {env.score}, Final Height: {env.height}")