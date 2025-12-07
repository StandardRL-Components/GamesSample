import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:21:12.340934
# Source Brief: brief_00954.md
# Brief Index: 954
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent guides a lucid dreamer through a
    procedurally generated labyrinth. The primary mechanic is placing portals
    to teleport and navigate to the exit. The agent controls a pointer to
    select locations for interaction.

    **Visuals:**
    - Dreamlike, surreal aesthetic with glowing elements and subtle animations.
    - Player: A bright, glowing white orb.
    - Pointer: A semi-transparent crosshair.
    - Walls: Dark, desaturated blue.
    - Exit: A glowing golden arch.
    - Portals: Swirling, color-coded vortexes (Green/Purple).
    - Dream Symbols: Pulsing geometric shapes that unlock portal upgrades.

    **Gameplay:**
    1. Move the pointer using directional actions.
    2. Use the 'Shift' action to cycle through unlocked portal types.
    3. Use the 'Space' action at the pointer's location to:
       - Activate a Dream Symbol, permanently unlocking portal upgrades.
       - Place one end of a portal pair. Placing the second end activates it.
    4. The player avatar only moves by teleporting through an active portal pair
       when one portal is placed on top of the other, or when the player is
       already on one end of a newly completed pair.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Guide a lucid dreamer through a surreal labyrinth by placing portals to teleport and navigate to the exit."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the pointer. Press space to place portals or activate symbols, and shift to cycle portal types."
    )
    auto_advance = True

    # --- Colors and Style ---
    COLOR_BG = (15, 10, 40)
    COLOR_WALL = (40, 50, 90)
    COLOR_PATH = (25, 20, 60)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_POINTER = (255, 255, 255, 128)
    COLOR_EXIT = (255, 223, 0)
    PORTAL_COLORS = {
        "standard": (0, 255, 150),
        "long_range": (180, 50, 255)
    }
    SYMBOL_COLOR = (0, 200, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_icon = pygame.font.SysFont("monospace", 20, bold=True)

        # --- Game Configuration ---
        self.max_steps = 2000

        # --- Persistent State (Progression) ---
        self.maze_w = 20
        self.maze_h = 15
        self.unlocked_portal_types = ["standard"]

        # --- Initialize State (will be overwritten by reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.maze = np.array([[]])
        self.player_pos = [0, 0]
        self.pointer_pos = [0, 0]
        self.exit_pos = (0, 0)
        self.symbols = []
        self.visited_tiles = set()
        self.portals = {}
        self.active_portal_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()
        # self.validate_implementation() # Removed for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Generate the labyrinth and place entities
        self.maze, self.player_pos, self.exit_pos, self.symbols = self._generate_maze(self.maze_w, self.maze_h)
        self.pointer_pos = list(self.player_pos)

        self.visited_tiles = {tuple(self.player_pos)}

        # Reset portal states
        self.portals = {
            "standard": {"pos_a": None, "pos_b": None},
            "long_range": {"pos_a": None, "pos_b": None}
        }
        self.active_portal_type_idx = 0

        # Reset input state for edge detection
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Actions ---
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game State & Rewards ---
        self.reward_this_step -= 0.01  # Small penalty per step to encourage efficiency

        terminated = self._check_termination()

        self.score += self.reward_this_step

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # 1. Shift: Cycle portal types (on press)
        is_shift_press = shift_held and not self.last_shift_held
        if is_shift_press:
            # SFX: UI_SWITCH
            self.active_portal_type_idx = (self.active_portal_type_idx + 1) % len(self.unlocked_portal_types)

        # 2. Movement: Move pointer
        px, py = self.pointer_pos
        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right
        self.pointer_pos = [max(0, min(self.maze_w - 1, px)), max(0, min(self.maze_h - 1, py))]

        # 3. Space: Interact (on press)
        is_space_press = space_held and not self.last_space_held
        if is_space_press:
            self._handle_interaction()

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _handle_interaction(self):
        target_pos = tuple(self.pointer_pos)

        # Priority 1: Activate Dream Symbol
        for i, (sx, sy, activated) in enumerate(self.symbols):
            if not activated and (sx, sy) == target_pos:
                self.symbols[i] = (sx, sy, True)
                self.reward_this_step += 5.0
                # SFX: SYMBOL_ACTIVATE
                if "long_range" not in self.unlocked_portal_types and sum(s[2] for s in self.symbols) >= 3:
                    self.unlocked_portal_types.append("long_range")
                return

        # Priority 2: Place Portal (if not on a wall)
        if self.maze[target_pos[1], target_pos[0]] == 0:
            portal_type = self.unlocked_portal_types[self.active_portal_type_idx]
            portal_set = self.portals[portal_type]

            if portal_set["pos_a"] is None:
                portal_set["pos_a"] = target_pos
                # SFX: PORTAL_PLACE_A
            elif portal_set["pos_b"] is None:
                if target_pos == portal_set["pos_a"]: return
                if portal_type == "standard":
                    dist = abs(target_pos[0] - portal_set["pos_a"][0]) + abs(target_pos[1] - portal_set["pos_a"][1])
                    if dist > 1:
                        # SFX: ACTION_FAIL
                        return
                portal_set["pos_b"] = target_pos
                # SFX: PORTAL_PLACE_B
                self._check_and_teleport()
            else: # Reset this portal type
                portal_set["pos_a"] = target_pos
                portal_set["pos_b"] = None
                # SFX: PORTAL_PLACE_A

    def _check_and_teleport(self):
        player_tuple = tuple(self.player_pos)
        for p_data in self.portals.values():
            pos_a, pos_b = p_data["pos_a"], p_data["pos_b"]
            if pos_a is not None and pos_b is not None:
                if player_tuple == pos_a:
                    self._teleport_player(pos_b)
                    return
                if player_tuple == pos_b:
                    self._teleport_player(pos_a)
                    return

    def _teleport_player(self, destination):
        # SFX: TELEPORT
        self.player_pos = list(destination)
        self.reward_this_step += 1.0

        if tuple(self.player_pos) not in self.visited_tiles:
            self.reward_this_step += 0.1
            self.visited_tiles.add(tuple(self.player_pos))
        else:
            self.reward_this_step -= 0.01 # Penalty for re-visiting

    def _check_termination(self):
        if tuple(self.player_pos) == self.exit_pos:
            self.game_over = True
            self.reward_this_step += 100.0
            # SFX: LEVEL_COMPLETE
            self.maze_w = min(50, int(self.maze_w * 1.05))
            self.maze_h = min(40, int(self.maze_h * 1.05))
            return True

        if self.steps >= self.max_steps:
            self.game_over = True
            self.reward_this_step -= 10.0
            # SFX: GAME_OVER
            return True

        return False

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "unlocked_portals": len(self.unlocked_portal_types)}

    def _get_maze_params(self):
        # Calculate tile size and offsets to center the maze
        usable_w, usable_h = self.screen_width - 20, self.screen_height - 60
        tile_w = usable_w / self.maze_w
        tile_h = usable_h / self.maze_h
        tile_size = min(tile_w, tile_h)
        offset_x = (self.screen_width - self.maze_w * tile_size) / 2
        offset_y = ((self.screen_height - 50) - self.maze_h * tile_size) / 2 + 50
        return tile_size, offset_x, offset_y

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_game()
        self._render_ui()

    def _render_background_effects(self):
        for i in range(20):
            t = self.steps + i * 20
            x = (self.screen_width / 2) + math.sin(t * 0.01 + i) * (self.screen_width / 2.2)
            y = (self.screen_height / 2) + math.cos(t * 0.015 + i) * (self.screen_height / 2.2)
            size = 2 + math.sin(t * 0.02) * 1.5
            alpha = 30 + math.sin(t * 0.03) * 20
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(size), (*self.COLOR_WALL, int(alpha)))

    def _render_game(self):
        tile_size, ox, oy = self._get_maze_params()

        # Draw maze paths and walls
        for y in range(self.maze_h):
            for x in range(self.maze_w):
                rect = (int(ox + x * tile_size), int(oy + y * tile_size), math.ceil(tile_size), math.ceil(tile_size))
                color = self.COLOR_WALL if self.maze[y, x] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)

        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(ox + ex * tile_size, oy + ey * tile_size, tile_size, tile_size)
        for i in range(5, 0, -1):
            glow_size = i * 4
            alpha = 150 - i * 30
            glow_rect = exit_rect.inflate(glow_size, glow_size)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_EXIT, alpha), s.get_rect(), border_radius=int(i*2))
            self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=3)


        # Draw symbols
        for x, y, activated in self.symbols:
            if not activated:
                cx = ox + (x + 0.5) * tile_size
                cy = oy + (y + 0.5) * tile_size
                pulse = (math.sin(self.steps * 0.1) + 1) / 2
                radius = tile_size * 0.2 + pulse * tile_size * 0.1
                pygame.gfxdraw.filled_circle(self.screen, int(cx), int(cy), int(radius), self.SYMBOL_COLOR)
                pygame.gfxdraw.aacircle(self.screen, int(cx), int(cy), int(radius), self.SYMBOL_COLOR)

        # Draw portals
        for p_type, p_data in self.portals.items():
            color = self.PORTAL_COLORS[p_type]
            for pos in [p_data["pos_a"], p_data["pos_b"]]:
                if pos:
                    cx = ox + (pos[0] + 0.5) * tile_size
                    cy = oy + (pos[1] + 0.5) * tile_size
                    for i in range(4):
                        t = self.steps + i * 20
                        radius = tile_size * 0.3 * (1 - i*0.2) * (0.8 + 0.2 * math.sin(t*0.15))
                        angle = t * 0.05 + i
                        px = cx + math.cos(angle) * tile_size * 0.1
                        py = cy + math.sin(angle) * tile_size * 0.1
                        alpha = 200 - i * 40
                        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(radius), (*color, alpha))

        # Draw player
        px, py = self.player_pos
        player_cx = int(ox + (px + 0.5) * tile_size)
        player_cy = int(oy + (py + 0.5) * tile_size)
        player_rad = int(tile_size * 0.3)
        for i in range(8, 0, -1):
            alpha = 150 - i * 18
            pygame.gfxdraw.filled_circle(self.screen, player_cx, player_cy, player_rad + i, (*self.COLOR_PLAYER, alpha))
        pygame.gfxdraw.filled_circle(self.screen, player_cx, player_cy, player_rad, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_cx, player_cy, player_rad, self.COLOR_PLAYER)


        # Draw pointer
        ptx, pty = self.pointer_pos
        pointer_cx = ox + (ptx + 0.5) * tile_size
        pointer_cy = oy + (pty + 0.5) * tile_size
        size = tile_size * 0.4
        pygame.draw.line(self.screen, self.COLOR_POINTER, (pointer_cx - size, pointer_cy), (pointer_cx + size, pointer_cy), 2)
        pygame.draw.line(self.screen, self.COLOR_POINTER, (pointer_cx, pointer_cy - size), (pointer_cx, pointer_cy + size), 2)

    def _render_ui(self):
        # Top bar background
        pygame.draw.rect(self.screen, (0,0,0,150), (0, 0, self.screen_width, 40))

        # Score and Steps
        score_text = self.font_ui.render(f"SCORE: {self.score:.2f}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.max_steps}", True, (255, 255, 255))
        self.screen.blit(steps_text, (self.screen_width - steps_text.get_width() - 10, 10))

        # Portal Info
        active_type = self.unlocked_portal_types[self.active_portal_type_idx]
        color = self.PORTAL_COLORS[active_type]
        portal_text = self.font_ui.render(f"PORTAL: {active_type.upper()}", True, color)
        self.screen.blit(portal_text, (180, 10))

        # Symbol Info
        collected_symbols = sum(s[2] for s in self.symbols)
        symbol_text = self.font_ui.render(f"SYMBOLS: {collected_symbols}/3", True, self.SYMBOL_COLOR)
        self.screen.blit(symbol_text, (380, 10))


    def _generate_maze(self, w, h):
        maze = np.ones((h, w), dtype=np.int8)
        # Use np_random for maze generation
        start_x, start_y = self.np_random.integers(0, w), self.np_random.integers(0, h)
        stack = [(start_x, start_y)]
        maze[start_y, start_x] = 0

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx * 2, cy + dy * 2
                if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))

            if neighbors:
                # Use np_random to choose neighbor
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]
                maze[ny, nx] = 0
                maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

        path_cells = np.argwhere(maze == 0)
        start_pos_idx = self.np_random.integers(0, len(path_cells))
        start_pos = path_cells[start_pos_idx]

        # Find the furthest point for the exit using BFS
        q = [(list(start_pos), 0)]
        visited = {tuple(start_pos)}
        farthest_pos, max_dist = list(start_pos), 0
        while q:
            (cy, cx), dist = q.pop(0)
            if dist > max_dist:
                max_dist = dist
                farthest_pos = (cx, cy)
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == 0 and (ny, nx) not in visited:
                    visited.add((ny,nx))
                    q.append(((ny, nx), dist + 1))
        
        exit_pos = farthest_pos
        
        # Place symbols
        symbols = []
        possible_symbol_locs = [tuple(c) for c in path_cells if tuple(c) != tuple(start_pos) and tuple(c) != exit_pos]
        # Use np_random for shuffling
        self.np_random.shuffle(possible_symbol_locs)
        for i in range(3):
            if possible_symbol_locs:
                sy, sx = possible_symbol_locs.pop() # Note: np.argwhere gives (row, col) which is (y, x)
                symbols.append((sx, sy, False))

        return maze, [start_pos[1], start_pos[0]], exit_pos, symbols

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # Ensure we are not using the dummy driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Instructions ---
    print("--- Lucid Labyrinth ---")
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("  Q: Quit")

    # Pygame setup for human play
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Lucid Labyrinth")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action Mapping for Human Play ---
        movement = 0 # None
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    env.close()