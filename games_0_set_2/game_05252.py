import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Press Space on a blue square to solve a puzzle. Avoid making noise."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a haunted maze, solving sound-based puzzles. Each move risks making noise, attracting unseen entities. Solve all puzzles and reach the green exit to escape."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 10
        self.CELL_SIZE = 40
        self.NUM_PUZZLES = 5
        self.NUM_NOISE_SOURCES = 3
        self.MAX_STEPS = 1000
        self.MAX_NOISE = 3

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_WALL = (60, 70, 90)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PUZZLE = (0, 150, 255)
        self.COLOR_PUZZLE_SOLVED = (0, 75, 128)
        self.COLOR_EXIT = (0, 255, 150)
        self.COLOR_DANGER = (200, 30, 30)
        self.COLOR_UI_TEXT = (220, 220, 220)

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_ui_large = pygame.font.Font(None, 40)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.noise_level = 0
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.puzzles = []
        self.noise_sources = []
        self.walls = set()
        self.effects = []
        self.noise_flash_timer = 0
        self.game_over = False

        self.reset()
        # self.validate_implementation() # Optional validation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.noise_level = 0
        self.game_over = False
        self.effects = []
        self.noise_flash_timer = 0

        # --- Procedural Generation ---
        self._generate_maze()
        
        # Get all valid floor cells
        all_cells = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        
        # Player start
        # FIX: self.np_random.choice on a list of tuples returns a numpy array,
        # which causes comparison errors later. Select an index instead to preserve the tuple type.
        player_pos_idx = self.np_random.integers(len(all_cells))
        self.player_pos = all_cells[player_pos_idx]
        
        # Place exit far from player
        possible_exits = sorted(all_cells, key=lambda p: -self._dist(p, self.player_pos))
        self.exit_pos = possible_exits[0]
        
        # Place puzzles and noise sources
        available_cells = [c for c in all_cells if c != self.player_pos and c != self.exit_pos]
        self.np_random.shuffle(available_cells)
        
        self.puzzles = []
        for i in range(self.NUM_PUZZLES):
            pos = available_cells.pop()
            self.puzzles.append({"pos": pos, "solved": False})
            
        self.noise_sources = []
        for i in range(self.NUM_NOISE_SOURCES):
            pos = available_cells.pop()
            # Initial radius ensures random agent survives ~50 steps
            self.noise_sources.append({"pos": pos, "base_radius": 1.5, "current_radius": 1.5})

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Movement ---
        prev_pos = self.player_pos
        moved = False
        if movement > 0:
            reward -= 0.1 # Movement penalty
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            next_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            if self._is_valid_move(self.player_pos, next_pos):
                self.player_pos = next_pos
                moved = True

        # --- Handle Interaction (Space) ---
        if space_held:
            for puzzle in self.puzzles:
                if puzzle["pos"] == self.player_pos and not puzzle["solved"]:
                    puzzle["solved"] = True
                    self.score += 1
                    reward += 10
                    # sfx: puzzle_solve.wav
                    self._add_effect('solve', self.player_pos)
                    
                    # Increase difficulty
                    for source in self.noise_sources:
                        source["current_radius"] += 1.0
                    break

        # --- Update Noise Level ---
        if moved:
            was_in_danger = self._is_in_danger_zone(prev_pos)
            is_in_danger = self._is_in_danger_zone(self.player_pos)
            if is_in_danger and not was_in_danger:
                self.noise_level += 1
                self.noise_flash_timer = 10
                # sfx: danger_sting.wav
        
        # --- Update Game State ---
        self.steps += 1
        self._update_effects()

        # --- Check Termination Conditions ---
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 100
            terminated = True
            self.game_over = True
            # sfx: win_fanfare.wav
            self._add_effect('win', self.player_pos)
        elif self.noise_level >= self.MAX_NOISE:
            reward -= 100
            terminated = True
            self.game_over = True
            # sfx: game_over_sound.wav
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
            
        terminated = terminated or truncated

        assert self.noise_level <= self.MAX_NOISE, "Noise level exceeded max"
        assert self.score <= self.NUM_PUZZLES, "Score exceeded puzzle count"

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "noise": self.noise_level}

    # --- Helper & Rendering Methods ---

    def _dist(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _is_in_danger_zone(self, pos):
        for source in self.noise_sources:
            if self._dist(pos, source["pos"]) <= source["current_radius"]:
                return True
        return False

    def _generate_maze(self):
        self.walls = set()
        # Add all walls initially
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if x < self.GRID_W - 1: self.walls.add(tuple(sorted(((x, y), (x + 1, y)))))
                if y < self.GRID_H - 1: self.walls.add(tuple(sorted(((x, y), (x, y + 1)))))

        # Randomized DFS
        start_node = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))
        visited = {start_node}
        stack = [start_node]

        while stack:
            current = stack[-1]
            x, y = current
            neighbors = []
            if x > 0 and (x - 1, y) not in visited: neighbors.append((x - 1, y))
            if x < self.GRID_W - 1 and (x + 1, y) not in visited: neighbors.append((x + 1, y))
            if y > 0 and (x, y - 1) not in visited: neighbors.append((x, y - 1))
            if y < self.GRID_H - 1 and (x, y + 1) not in visited: neighbors.append((x, y + 1))

            if neighbors:
                # FIX: self.np_random.choice on a list of tuples returns a numpy array,
                # which causes comparison errors in tuple(sorted(...)).
                # Instead, we pick an index to get the original tuple.
                next_node_idx = self.np_random.integers(len(neighbors))
                next_node = neighbors[next_node_idx]
                
                wall = tuple(sorted((current, next_node)))
                if wall in self.walls:
                    self.walls.remove(wall)
                visited.add(next_node)
                stack.append(next_node)
            else:
                stack.pop()
    
    def _is_valid_move(self, from_pos, to_pos):
        tx, ty = to_pos
        if not (0 <= tx < self.GRID_W and 0 <= ty < self.GRID_H):
            return False
        wall = tuple(sorted((from_pos, to_pos)))
        if wall in self.walls:
            return False
        return True

    def _add_effect(self, type, pos, lifespan=20):
        self.effects.append({"type": type, "pos": pos, "life": lifespan, "max_life": lifespan})

    def _update_effects(self):
        self.effects = [e for e in self.effects if e['life'] > 0]
        for e in self.effects:
            e['life'] -= 1
        if self.noise_flash_timer > 0:
            self.noise_flash_timer -= 1

    def _world_to_screen(self, x, y):
        return int(x * self.CELL_SIZE + self.CELL_SIZE / 2), int(y * self.CELL_SIZE + self.CELL_SIZE / 2)

    def _render_game(self):
        # --- Background Noise Pulse ---
        if self.noise_level > 0:
            danger_alpha = int(30 + (self.noise_level / self.MAX_NOISE) * 70)
            if self.noise_flash_timer > 0:
                flash_progress = self.noise_flash_timer / 10
                danger_alpha = int(120 * flash_progress)
            
            danger_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            danger_surface.fill((*self.COLOR_DANGER, danger_alpha))
            self.screen.blit(danger_surface, (0, 0))

        # --- Grid lines (subtle) ---
        for x in range(self.GRID_W + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.CELL_SIZE, 0), (x * self.CELL_SIZE, self.HEIGHT))
        for y in range(self.GRID_H + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.CELL_SIZE), (self.WIDTH, y * self.CELL_SIZE))

        # --- Walls ---
        for wall in self.walls:
            p1, p2 = wall
            # Wall coordinates are grid-based, convert to screen centers
            start_pos = (p1[0] * self.CELL_SIZE + self.CELL_SIZE / 2, p1[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
            end_pos = (p2[0] * self.CELL_SIZE + self.CELL_SIZE / 2, p2[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
            pygame.draw.line(self.screen, self.COLOR_WALL, start_pos, end_pos, 3)


        # --- Exit ---
        ex, ey = self._world_to_screen(self.exit_pos[0], self.exit_pos[1])
        exit_rect = pygame.Rect(ex - self.CELL_SIZE/2, ey - self.CELL_SIZE/2, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect.inflate(-8, -8))

        # --- Puzzles ---
        for puzzle in self.puzzles:
            px, py = self._world_to_screen(puzzle["pos"][0], puzzle["pos"][1])
            color = self.COLOR_PUZZLE_SOLVED if puzzle["solved"] else self.COLOR_PUZZLE
            size = self.CELL_SIZE * 0.4
            pygame.draw.rect(self.screen, color, (px - size/2, py - size/2, size, size), border_radius=3)
            
        # --- Player ---
        plx, ply = self._world_to_screen(self.player_pos[0], self.player_pos[1])
        player_size = self.CELL_SIZE * 0.6
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (plx - player_size/2, ply - player_size/2, player_size, player_size), border_radius=4)
        pygame.gfxdraw.aacircle(self.screen, int(plx), int(ply), int(player_size * 0.7), (*self.COLOR_PLAYER, 100))
        
        # --- Effects ---
        for e in self.effects:
            progress = 1 - (e['life'] / e['max_life'])
            pos_x, pos_y = self._world_to_screen(e['pos'][0], e['pos'][1])
            if e['type'] == 'solve':
                radius = int(progress * self.CELL_SIZE)
                alpha = int(255 * (1 - progress))
                pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, radius, (*self.COLOR_PUZZLE, alpha))
            elif e['type'] == 'win':
                radius = int(progress * self.CELL_SIZE * 2)
                alpha = int(255 * (1 - progress))
                pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, radius, (*self.COLOR_EXIT, alpha))

    def _render_ui(self):
        # --- Score ---
        score_text = self.font_ui.render(f"Puzzles: {self.score}/{self.NUM_PUZZLES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 10))

        # --- Noise Meter ---
        noise_label = self.font_ui.render("Noise", True, self.COLOR_UI_TEXT)
        self.screen.blit(noise_label, (15, 10))
        bar_width = 100
        bar_height = 15
        bar_x, bar_y = 15, 35
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
        fill_width = (self.noise_level / self.MAX_NOISE) * bar_width
        pygame.draw.rect(self.screen, self.COLOR_DANGER, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # --- Game Over Text ---
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.player_pos == self.exit_pos and self.steps < self.MAX_STEPS:
                msg = "ESCAPED"
                color = self.COLOR_EXIT
            else:
                msg = "CAPTURED"
                color = self.COLOR_DANGER
            
            end_text = self.font_ui_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Haunted Grid")
    clock = pygame.time.Clock()
    running = True
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space, shift])

        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to Screen ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # We can just re-render from the env's internal screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Wait 3 seconds before reset
            obs, info = env.reset()

        clock.tick(10) # Control human play speed

    env.close()