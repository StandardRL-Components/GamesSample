import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:33:14.789349
# Source Brief: brief_01129.md
# Brief Index: 1129
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    Transform into different geometric shapes to navigate a procedurally generated
    labyrinth and collect hidden artifacts before time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Transform into different geometric shapes to navigate a procedurally generated "
        "labyrinth and collect hidden artifacts before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to cycle through shapes "
        "to pass through matching gates."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1500  # ~50 seconds

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (60, 70, 90)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    
    COLOR_GATE_CIRCLE = (255, 80, 80)
    COLOR_GATE_SQUARE = (80, 255, 80)
    COLOR_GATE_TRIANGLE = (80, 80, 255)
    GATE_COLORS = [COLOR_GATE_CIRCLE, COLOR_GATE_SQUARE, COLOR_GATE_TRIANGLE]

    COLOR_ARTIFACT = (255, 220, 0)
    COLOR_ARTIFACT_GLOW = (255, 240, 100)
    
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_TIMER_GOOD = (100, 255, 100)
    COLOR_TIMER_BAD = (255, 100, 100)

    # Game settings
    PLAYER_SIZE = 12
    PLAYER_SPEED = 0.25  # Interpolation factor
    INITIAL_TIME_SECONDS = 60
    INITIAL_ARTIFACTS = 3
    
    # Shape IDs
    SHAPE_CIRCLE = 0
    SHAPE_SQUARE = 1
    SHAPE_TRIANGLE = 2
    SHAPE_NAMES = ["CIRCLE", "SQUARE", "TRIANGLE"]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_shape = pygame.font.SysFont("monospace", 12, bold=True)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_target_pos = np.zeros(2, dtype=np.float32)
        self.player_shape = self.SHAPE_CIRCLE
        self.player_transform_progress = 0.0
        
        self.artifacts = []
        self.artifacts_collected_count = 0
        self.num_artifacts_total = 0
        
        self.walls = []
        self.gates = []
        self.grid_cols = 0
        self.grid_rows = 0
        self.cell_width = 0
        self.cell_height = 0

        self.particles = []
        self.prev_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.INITIAL_TIME_SECONDS * self.FPS
        
        self.num_artifacts_total = self.INITIAL_ARTIFACTS
        self.artifacts_collected_count = 0

        self._generate_level(complexity=0)

        start_cell_x = self.np_random.integers(0, self.grid_cols)
        start_cell_y = self.np_random.integers(0, self.grid_rows)
        self.player_pos = self._get_cell_center(start_cell_x, start_cell_y)
        self.player_target_pos = self.player_pos.copy()
        
        self.player_shape = self.SHAPE_CIRCLE
        self.player_transform_progress = 1.0

        self.particles.clear()
        self.prev_space_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01  # Time penalty
        
        # --- Calculate distance to nearest artifact for reward ---
        prev_dist_to_artifact = self._get_min_dist_to_artifact()

        # --- Handle Input ---
        self._handle_input(movement, space_held)

        # --- Update Game State ---
        self._update_player_position()
        self._update_particles()
        self.time_remaining -= 1
        
        # --- Check for Artifact Collection ---
        collected_artifact_reward = self._check_artifact_collection()
        if collected_artifact_reward > 0:
            reward += collected_artifact_reward
            # Regenerate level with increased complexity
            self._generate_level(complexity=self.artifacts_collected_count)
            # Find a safe spot for the player in the new maze
            self.player_target_pos = self._find_safe_spawn()
            self.player_pos = self.player_target_pos.copy()


        # --- Calculate movement reward ---
        new_dist_to_artifact = self._get_min_dist_to_artifact()
        if new_dist_to_artifact < prev_dist_to_artifact:
            reward += 0.1

        # --- Check Termination ---
        terminated = False
        if self.artifacts_collected_count >= self.num_artifacts_total:
            reward += 100  # Victory bonus
            terminated = True
            self._create_particles(self.player_pos, 50, self.COLOR_ARTIFACT)
        elif self.time_remaining <= 0:
            reward -= 10  # Timeout penalty
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Shape shifting on space PRESS (not hold)
        if space_held and not self.prev_space_held:
            self.player_shape = (self.player_shape + 1) % 3  # Cycle through 3 shapes
            self.player_transform_progress = 0.0
            self._create_particles(self.player_pos, 20, self.GATE_COLORS[self.player_shape])
            # sfx: shape_change.wav
        self.prev_space_held = space_held

        # Movement
        current_cell_x = int(self.player_target_pos[0] / self.cell_width)
        current_cell_y = int(self.player_target_pos[1] / self.cell_height)
        
        target_cell_x, target_cell_y = current_cell_x, current_cell_y

        if movement == 1: target_cell_y -= 1  # Up
        elif movement == 2: target_cell_y += 1  # Down
        elif movement == 3: target_cell_x -= 1  # Left
        elif movement == 4: target_cell_x += 1  # Right
        
        if (target_cell_x, target_cell_y) != (current_cell_x, current_cell_y):
            if self._is_move_valid(current_cell_x, current_cell_y, target_cell_x, target_cell_y):
                self.player_target_pos = self._get_cell_center(target_cell_x, target_cell_y)
                # sfx: move.wav

    def _update_player_position(self):
        # Smooth interpolation
        self.player_pos += (self.player_target_pos - self.player_pos) * self.PLAYER_SPEED
        # Animate shape transformation
        if self.player_transform_progress < 1.0:
            self.player_transform_progress = min(1.0, self.player_transform_progress + 0.1)

    def _check_artifact_collection(self):
        for i in range(len(self.artifacts) - 1, -1, -1):
            artifact_pos = self.artifacts[i]
            dist = np.linalg.norm(self.player_pos - artifact_pos)
            if dist < self.PLAYER_SIZE + 5: # 5 is artifact size
                self.artifacts.pop(i)
                self.artifacts_collected_count += 1
                self.score += 1.0
                self._create_particles(artifact_pos, 30, self.COLOR_ARTIFACT)
                # sfx: collect_artifact.wav
                return 1.0
        return 0.0

    def _get_min_dist_to_artifact(self):
        if not self.artifacts:
            return 0
        player_pos = self.player_pos
        distances = [np.linalg.norm(player_pos - art_pos) for art_pos in self.artifacts]
        return min(distances)

    def _is_move_valid(self, cx, cy, tx, ty):
        # Boundary check
        if not (0 <= tx < self.grid_cols and 0 <= ty < self.grid_rows):
            return False

        # Check for walls/gates
        for gate in self.gates:
            g_cx, g_cy, g_tx, g_ty, g_shape = gate
            # Check for passage from current to target
            if (cx, cy, tx, ty) == (g_cx, g_cy, g_tx, g_ty):
                return self.player_shape == g_shape
            # Check for passage from target to current (reverse)
            if (cx, cy, tx, ty) == (g_tx, g_ty, g_cx, g_cy):
                return self.player_shape == g_shape
        
        # If no gate is found, it's a solid wall
        return False

    def _get_cell_center(self, x, y):
        return np.array([
            x * self.cell_width + self.cell_width / 2,
            y * self.cell_height + self.cell_height / 2
        ], dtype=np.float32)
    
    def _find_safe_spawn(self):
        """Finds a cell that is not occupied by an artifact."""
        occupied_cells = {
            (int(art[0] / self.cell_width), int(art[1] / self.cell_height))
            for art in self.artifacts
        }
        
        possible_cells = []
        for x in range(self.grid_cols):
            for y in range(self.grid_rows):
                if (x, y) not in occupied_cells:
                    possible_cells.append((x,y))

        if not possible_cells:
            # Fallback to a random cell if all else fails
            return self._get_cell_center(self.np_random.integers(0, self.grid_cols), self.np_random.integers(0, self.grid_rows))
        
        spawn_cell = possible_cells[self.np_random.integers(0, len(possible_cells))]
        return self._get_cell_center(spawn_cell[0], spawn_cell[1])


    def _generate_level(self, complexity=0):
        """Generates the maze, gates, and artifacts."""
        self.walls.clear()
        self.gates.clear()
        self.artifacts.clear()

        # Difficulty scaling
        base_cols = 16
        base_rows = 10
        self.grid_cols = max(8, base_cols - complexity)
        self.grid_rows = max(5, base_rows - int(complexity * 0.6))
        
        self.cell_width = self.SCREEN_WIDTH / self.grid_cols
        self.cell_height = self.SCREEN_HEIGHT / self.grid_rows

        grid = [[{'visited': False, 'walls': [True, True, True, True]} for _ in range(self.grid_rows)] for _ in range(self.grid_cols)]
        stack = []
        
        start_x, start_y = self.np_random.integers(0, self.grid_cols), self.np_random.integers(0, self.grid_rows)
        grid[start_x][start_y]['visited'] = True
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack.pop()
            neighbors = []
            # Top
            if cy > 0 and not grid[cx][cy-1]['visited']: neighbors.append(((cx, cy-1), 0, 2))
            # Right
            if cx < self.grid_cols - 1 and not grid[cx+1][cy]['visited']: neighbors.append(((cx+1, cy), 1, 3))
            # Bottom
            if cy < self.grid_rows - 1 and not grid[cx][cy+1]['visited']: neighbors.append(((cx, cy+1), 2, 0))
            # Left
            if cx > 0 and not grid[cx-1][cy]['visited']: neighbors.append(((cx-1, cy), 3, 1))

            if neighbors:
                stack.append((cx, cy))
                (nx, ny), wall_idx, opposite_wall_idx = neighbors[self.np_random.integers(len(neighbors))]
                
                grid[cx][cy]['walls'][wall_idx] = False
                grid[nx][ny]['walls'][opposite_wall_idx] = False
                
                # Create a gate where a wall was removed
                gate_shape = self.np_random.integers(0, 3)
                self.gates.append((cx, cy, nx, ny, gate_shape))
                
                grid[nx][ny]['visited'] = True
                stack.append((nx, ny))
        
        # Store wall line segments for rendering
        for x in range(self.grid_cols):
            for y in range(self.grid_rows):
                if grid[x][y]['walls'][0]: # Top
                    self.walls.append(((x * self.cell_width, y * self.cell_height), ((x+1) * self.cell_width, y * self.cell_height)))
                if grid[x][y]['walls'][1]: # Right
                    self.walls.append((((x+1) * self.cell_width, y * self.cell_height), ((x+1) * self.cell_width, (y+1) * self.cell_height)))

        # Place artifacts
        num_to_place = self.num_artifacts_total - self.artifacts_collected_count
        player_cell_x = -1
        player_cell_y = -1
        if self.player_pos is not None and self.cell_width > 0 and self.cell_height > 0:
            player_cell_x = int(self.player_pos[0] / self.cell_width)
            player_cell_y = int(self.player_pos[1] / self.cell_height)
        
        possible_cells = []
        for x in range(self.grid_cols):
            for y in range(self.grid_rows):
                if (x, y) != (player_cell_x, player_cell_y):
                    possible_cells.append((x, y))
        
        self.np_random.shuffle(possible_cells)
        
        for i in range(min(num_to_place, len(possible_cells))):
            ax, ay = possible_cells[i]
            self.artifacts.append(self._get_cell_center(ax, ay))


    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_background_details()
        self._render_walls_and_gates()
        self._render_artifacts()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_details(self):
        # Subtle grid lines
        for x in range(self.grid_cols + 1):
            px = x * self.cell_width
            pygame.draw.line(self.screen, (30, 35, 50), (px, 0), (px, self.SCREEN_HEIGHT), 1)
        for y in range(self.grid_rows + 1):
            py = y * self.cell_height
            pygame.draw.line(self.screen, (30, 35, 50), (0, py), (self.SCREEN_WIDTH, py), 1)

    def _render_walls_and_gates(self):
        # Render gates first
        for cx, cy, nx, ny, shape in self.gates:
            pos1 = self._get_cell_center(cx, cy)
            pos2 = self._get_cell_center(nx, ny)
            mid_pos = (pos1 + pos2) / 2
            self._draw_shape(self.screen, shape, mid_pos, self.PLAYER_SIZE * 0.8, self.GATE_COLORS[shape])
        
        # Render solid walls
        for start, end in self.walls:
            pygame.draw.line(self.screen, self.COLOR_WALL, start, end, 3)

    def _render_artifacts(self):
        pulse = (math.sin(self.steps * 0.1) + 1) / 2  # 0 to 1
        glow_alpha = 50 + pulse * 50
        
        for pos in self.artifacts:
            # Draw glow
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 12, (*self.COLOR_ARTIFACT_GLOW, int(glow_alpha)))
            # Draw artifact (a simple diamond shape)
            points = [
                (pos[0], pos[1] - 7), (pos[0] + 5, pos[1]),
                (pos[0], pos[1] + 7), (pos[0] - 5, pos[1])
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ARTIFACT)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ARTIFACT)

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        size = self.PLAYER_SIZE * self.player_transform_progress

        # Draw glow
        glow_radius = int(size * 1.8)
        if glow_radius > 0:
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

        # Draw player shape
        self._draw_shape(self.screen, self.player_shape, self.player_pos, size, self.COLOR_PLAYER)
        
        # Draw current shape icon next to player
        icon_pos = (pos[0] + 20, pos[1] - 20)
        self._draw_shape(self.screen, self.player_shape, icon_pos, 5, self.GATE_COLORS[self.player_shape])

    def _draw_shape(self, surface, shape, pos, size, color):
        if size <= 0: return
        pos_i = (int(pos[0]), int(pos[1]))

        if shape == self.SHAPE_CIRCLE:
            pygame.gfxdraw.filled_circle(surface, pos_i[0], pos_i[1], int(size), color)
            pygame.gfxdraw.aacircle(surface, pos_i[0], pos_i[1], int(size), color)
        elif shape == self.SHAPE_SQUARE:
            rect = pygame.Rect(pos_i[0] - size, pos_i[1] - size, size * 2, size * 2)
            pygame.draw.rect(surface, color, rect)
        elif shape == self.SHAPE_TRIANGLE:
            points = [
                (pos_i[0], pos_i[1] - size * 1.15),
                (pos_i[0] - size, pos_i[1] + size * 0.7),
                (pos_i[0] + size, pos_i[1] + size * 0.7)
            ]
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)

    def _render_ui(self):
        # Timer
        time_ratio = max(0, self.time_remaining / (self.INITIAL_TIME_SECONDS * self.FPS))
        timer_color = (
            int(self.COLOR_TIMER_BAD[0] + (self.COLOR_TIMER_GOOD[0] - self.COLOR_TIMER_BAD[0]) * time_ratio),
            int(self.COLOR_TIMER_BAD[1] + (self.COLOR_TIMER_GOOD[1] - self.COLOR_TIMER_BAD[1]) * time_ratio),
            int(self.COLOR_TIMER_BAD[2] + (self.COLOR_TIMER_GOOD[2] - self.COLOR_TIMER_BAD[2]) * time_ratio)
        )
        time_text = f"TIME: {self.time_remaining / self.FPS:.1f}"
        time_surf = self.font_ui.render(time_text, True, timer_color)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        
        # Artifacts
        artifact_text = f"ARTIFACTS: {self.artifacts_collected_count}/{self.num_artifacts_total}"
        artifact_surf = self.font_ui.render(artifact_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(artifact_surf, (self.SCREEN_WIDTH - artifact_surf.get_width() - 10, 35))

    # --- Particle System ---

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['vel'] *= 0.95  # friction
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.pop(i)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            size = int(life_ratio * 4)
            if size > 0:
                color = (*p['color'], int(life_ratio * 255))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

    # --- Gymnasium Interface Compliance ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "artifacts_collected": self.artifacts_collected_count,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play Example ---
    # This block requires a graphical display. 
    # It will not run in a headless environment.
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        env = GameEnv()
        obs, info = env.reset()
        done = False
        
        render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Shape Shifter Labyrinth")
        clock = pygame.time.Clock()

        total_reward = 0
        total_steps = 0
        
        movement_action = 0
        space_action = 0
        
        print("\n--- Manual Control ---")
        print(GameEnv.user_guide)
        print("R: Reset")
        print("Q: Quit")
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        total_steps = 0
                        done = False

            keys = pygame.key.get_pressed()
            movement_action = 0
            if keys[pygame.K_UP]: movement_action = 1
            elif keys[pygame.K_DOWN]: movement_action = 2
            elif keys[pygame.K_LEFT]: movement_action = 3
            elif keys[pygame.K_RIGHT]: movement_action = 4
            
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement_action, space_action, shift_action]
            
            if not done:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                total_steps += 1
                done = terminated or truncated

            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if done:
                print(f"Episode finished in {total_steps} steps. Final Score: {total_reward:.2f}")
                # Wait for reset
            
            clock.tick(GameEnv.FPS)
            
        env.close()
    except pygame.error as e:
        print(f"Pygame error (likely no display found): {e}")
        print("Skipping manual play example.")