import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
from collections import deque
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Domino Cascade Puzzle Environment for Gymnasium.

    The agent places different types of dominoes on a grid to trigger
    chain reactions. These reactions transform untraversable terrain into
    traversable paths. The goal is to create a continuous path from a

    start point to an end point.
    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]` (Movement): 0=none, 1=up, 2=down, 3=left, 4=right
    - `actions[1]` (Place Domino): 0=released, 1=held (triggers on press)
    - `actions[2]` (Cycle Type): 0=released, 1=held (triggers on press)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game state.

    **Reward Structure:**
    - +1 for each terrain block converted from untraversable to traversable.
    - +100 for successfully connecting the start and end points.
    - -10 for running out of dominoes without a solution.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Place different types of dominoes to create a chain reaction and build a path from the start to the end point."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press Shift to cycle domino types and Space to place a domino, triggering a cascade."
    )
    auto_advance = True

    # --- Constants ---
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CELL_SIZE = SCREEN_WIDTH // GRID_WIDTH
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (26, 28, 44)
    COLOR_GRID_LINES = (40, 42, 60)
    COLOR_UNTRAVERSABLE = (61, 64, 80)
    COLOR_TRAVERSABLE = (79, 131, 204)
    COLOR_START = (82, 214, 138)
    COLOR_END = (217, 87, 99)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (230, 230, 230)
    DOMINO_TYPES = [
        {'color': (255, 240, 150), 'fall_delay': 5, 'name': 'Light'},  # Fast
        {'color': (255, 180, 100), 'fall_delay': 10, 'name': 'Medium'}, # Medium
        {'color': (255, 120, 120), 'fall_delay': 15, 'name': 'Heavy'},  # Slow
    ]

    # --- Tile Types ---
    TILE_UNTRAVERSABLE = 0
    TILE_TRAVERSABLE = 1
    TILE_START = 2
    TILE_END = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.domino_inventory = None
        self.selected_domino_idx = None
        self.start_pos = None
        self.end_pos = None
        self.placed_dominoes = None
        self.falling_dominoes = None
        self.particles = None
        self.game_mode = None # 'PLACING' or 'CASCADING'
        self.steps = None
        self.score = None
        self.level = None
        self.has_terminated = None
        self.space_was_held = False
        self.shift_was_held = False
        self.reward_this_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.level = options.get("level", 1) if options else 1
        self.has_terminated = False
        self.game_mode = 'PLACING'
        self.reward_this_step = 0

        self._generate_level()

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.domino_inventory = {i: 5 + (self.level // 2) for i in range(len(self.DOMINO_TYPES))}
        self.selected_domino_idx = 0
        self.placed_dominoes = []
        self.falling_dominoes = deque()
        self.particles = []

        self.space_was_held = False
        self.shift_was_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        is_done = self.has_terminated or self.steps >= self.MAX_STEPS
        if is_done:
            return self._get_observation(), 0, self.has_terminated, self.steps >= self.MAX_STEPS, self._get_info()

        self.steps += 1
        self.reward_this_step = 0

        if self.game_mode == 'PLACING':
            self._handle_placing_action(action)
        elif self.game_mode == 'CASCADING':
            self._update_cascade()

        terminated = self.has_terminated
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_placing_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Handle Cycle Domino Type (on key press) ---
        if shift_held and not self.shift_was_held:
            self.selected_domino_idx = (self.selected_domino_idx + 1) % len(self.DOMINO_TYPES)
        self.shift_was_held = shift_held

        # --- Handle Place Domino (on key press) ---
        if space_held and not self.space_was_held:
            self._place_domino()
        self.space_was_held = space_held

    def _place_domino(self):
        x, y = self.cursor_pos
        is_occupied = any(d['pos'] == [x, y] for d in self.placed_dominoes)
        is_valid_tile = self.grid[y, x] == self.TILE_TRAVERSABLE
        has_dominoes = self.domino_inventory[self.selected_domino_idx] > 0

        if is_valid_tile and not is_occupied and has_dominoes:
            domino_type = self.DOMINO_TYPES[self.selected_domino_idx]
            new_domino = {
                'pos': [x, y],
                'type_idx': self.selected_domino_idx,
                'state': 'standing',
                'angle': 0,
                'fall_timer': -1
            }
            self.placed_dominoes.append(new_domino)
            self.domino_inventory[self.selected_domino_idx] -= 1

            self._trigger_domino(new_domino, delay=1)
            self.game_mode = 'CASCADING'

    def _trigger_domino(self, domino, delay):
        if domino['state'] == 'standing':
            domino['state'] = 'falling'
            domino['fall_timer'] = delay
            self.falling_dominoes.append(domino)

    def _update_cascade(self):
        if not self.falling_dominoes:
            self.game_mode = 'PLACING'
            # Check for game end conditions now that the cascade is over.
            victory = self._check_path_exists(self.start_pos, self.end_pos)
            if victory:
                self.reward_this_step += 100
                self.score += 100
                self.has_terminated = True
            elif sum(self.domino_inventory.values()) == 0 and not self._any_dominoes_on_grid():
                self.reward_this_step -= 10
                self.score -= 10
                self.has_terminated = True
            return

        for _ in range(len(self.falling_dominoes)):
            domino = self.falling_dominoes.popleft()
            domino['fall_timer'] -= 1

            if domino['fall_timer'] > 0:
                self.falling_dominoes.append(domino)
                continue

            domino['state'] = 'fallen'
            x, y = domino['pos']

            if self.grid[y, x] == self.TILE_UNTRAVERSABLE:
                self.grid[y, x] = self.TILE_TRAVERSABLE
                self.reward_this_step += 1
                self.score += 1

            self._create_particles(x, y, self.DOMINO_TYPES[domino['type_idx']]['color'])

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    neighbor_domino = next((d for d in self.placed_dominoes if d['pos'] == [nx, ny]), None)
                    if neighbor_domino and neighbor_domino['state'] == 'standing':
                        delay = self.DOMINO_TYPES[neighbor_domino['type_idx']]['fall_delay']
                        self._trigger_domino(neighbor_domino, delay)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._update_and_render_particles()
        self._render_dominoes()
        self._render_cursor()

    def _render_grid(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                tile_type = self.grid[y, x]
                color = self.COLOR_UNTRAVERSABLE
                if tile_type == self.TILE_TRAVERSABLE:
                    color = self.COLOR_TRAVERSABLE
                elif tile_type == self.TILE_START:
                    color = self.COLOR_START
                elif tile_type == self.TILE_END:
                    color = self.COLOR_END
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect, 1)

    def _render_dominoes(self):
        for domino in self.placed_dominoes:
            x, y = domino['pos']
            px, py = (x + 0.5) * self.CELL_SIZE, (y + 0.5) * self.CELL_SIZE
            domino_w, domino_h = self.CELL_SIZE * 0.2, self.CELL_SIZE * 0.7
            color = self.DOMINO_TYPES[domino['type_idx']]['color']

            if domino['state'] == 'falling':
                domino_type = self.DOMINO_TYPES[domino['type_idx']]
                progress = 1.0 - (domino['fall_timer'] / domino_type['fall_delay'])
                domino['angle'] = -90 * progress

            surf = pygame.Surface((domino_w, domino_h), pygame.SRCALPHA)
            surf.fill(color)
            rotated_surf = pygame.transform.rotate(surf, domino['angle'])
            rect = rotated_surf.get_rect(center=(int(px), int(py)))
            self.screen.blit(rotated_surf, rect)

    def _render_cursor(self):
        if self.game_mode == 'PLACING':
            x, y = self.cursor_pos
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)

            domino_type = self.DOMINO_TYPES[self.selected_domino_idx]
            domino_w, domino_h = self.CELL_SIZE * 0.2, self.CELL_SIZE * 0.7
            preview_surf = pygame.Surface((domino_w, domino_h), pygame.SRCALPHA)
            preview_surf.fill(domino_type['color'] + (128,))
            rect = preview_surf.get_rect(center=rect.center)
            self.screen.blit(preview_surf, rect)

    def _render_ui(self):
        ui_x, ui_y = 10, 10
        for i, domino_type in enumerate(self.DOMINO_TYPES):
            count = self.domino_inventory[i]
            text = f"{domino_type['name']}: {count}"
            text_surf = self.font_small.render(text, True, self.COLOR_TEXT)
            self.screen.blit(text_surf, (ui_x, ui_y + i * 20))
            if i == self.selected_domino_idx and self.game_mode == 'PLACING':
                rect = pygame.Rect(ui_x - 4, ui_y + i * 20 - 2, text_surf.get_width() + 8, text_surf.get_height() + 4)
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 1)

        score_text = f"Score: {self.score}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))
        
        level_text = f"Level: {self.level}"
        level_surf = self.font_small.render(level_text, True, self.COLOR_TEXT)
        self.screen.blit(level_surf, (self.SCREEN_WIDTH - level_surf.get_width() - 10, 30))

        if self.has_terminated or self.steps >= self.MAX_STEPS:
            victory = self._check_path_exists(self.start_pos, self.end_pos)
            msg = "VICTORY!" if victory and self.has_terminated else "OUT OF DOMINOES"
            color = self.COLOR_START if victory and self.has_terminated else self.COLOR_END
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            pygame.draw.rect(self.screen, self.COLOR_BG, msg_rect.inflate(20, 20))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "cursor_pos": self.cursor_pos,
            "domino_inventory": self.domino_inventory,
            "game_mode": self.game_mode,
        }

    def _generate_level(self):
        self.grid = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), self.TILE_UNTRAVERSABLE, dtype=np.int8)
        self.start_pos = (1, self.np_random.integers(1, self.GRID_HEIGHT - 1))
        self.end_pos = (self.GRID_WIDTH - 2, self.np_random.integers(1, self.GRID_HEIGHT - 1))
        self.grid[self.start_pos[1], self.start_pos[0]] = self.TILE_START
        self.grid[self.end_pos[1], self.end_pos[0]] = self.TILE_END

        path_points = self._generate_random_walk(self.start_pos, self.end_pos)
        for x, y in path_points:
            if self.grid[y, x] == self.TILE_UNTRAVERSABLE:
                self.grid[y, x] = self.TILE_TRAVERSABLE
        
        for _ in range(self.level * 2):
            cx, cy = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
            radius = self.np_random.integers(2, 5)
            for y in range(cy - radius, cy + radius):
                for x in range(cx - radius, cx + radius):
                    if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                        if self.grid[y,x] == self.TILE_UNTRAVERSABLE:
                            if self.np_random.random() > 0.4:
                                self.grid[y,x] = self.TILE_TRAVERSABLE

    def _generate_random_walk(self, start, end):
        path = []
        pos = list(start)
        while tuple(pos) != end:
            path.append(tuple(pos))
            dx = np.sign(end[0] - pos[0])
            dy = np.sign(end[1] - pos[1])
            if dx != 0 and (self.np_random.random() < 0.7 or dy == 0):
                pos[0] += dx
            elif dy != 0:
                pos[1] += dy
        path.append(end)
        return path

    def _check_path_exists(self, start, end):
        q = deque([start])
        visited = {start}
        while q:
            x, y = q.popleft()
            if (x, y) == end:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    tile = self.grid[ny, nx]
                    if tile in [self.TILE_TRAVERSABLE, self.TILE_END, self.TILE_START]:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return False

    def _any_dominoes_on_grid(self):
        return any(d['state'] != 'fallen' for d in self.placed_dominoes)

    def _create_particles(self, grid_x, grid_y, color):
        px = (grid_x + 0.5) * self.CELL_SIZE
        py = (grid_y + 0.5) * self.CELL_SIZE
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _update_and_render_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                alpha = max(0, 255 * (p['lifespan'] / 30))
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]),
                    int(p['radius']), p['color'] + (int(alpha),)
                )
                active_particles.append(p)
        self.particles = active_particles

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv()
    
    # --- Manual Play ---
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Domino Cascade")
    clock = pygame.time.Clock()
    
    obs, info = env.reset(options={"level": 1})
    done = False
    
    while not done:
        # --- Action Mapping for Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward}, Score: {info['score']}")

        if terminated or truncated:
            print("Game Over!")
            print(f"Final Score: {info['score']}")
            pygame.time.wait(2000)
            
            # Determine next level
            next_level = info["level"]
            if terminated and info['score'] > 0: # Victory
                next_level += 1
            
            obs, info = env.reset(options={"level": next_level})

        # --- Rendering for Manual Play ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling (for quitting) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30)

    env.close()