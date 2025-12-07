import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Shift to rotate pipe. Space to place pipe."
    )

    game_description = (
        "Connect the green source to the red drain by placing pipe segments. "
        "Create the longest continuous flow to maximize your score."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 8
    GRID_OFFSET_X, GRID_OFFSET_Y = 40, 40
    CELL_SIZE = 40
    UI_WIDTH = 200

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_SOURCE = (60, 220, 120)
    COLOR_DRAIN = (220, 60, 60)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_CURSOR_INVALID = (255, 50, 50)
    COLOR_PIPE = (180, 190, 200)
    COLOR_FLOW = (80, 150, 255)
    COLOR_TEXT = (230, 240, 250)
    COLOR_TEXT_SHADOW = (10, 15, 20)

    # Pipe Types (0: empty, 1: straight, 2: elbow, 3: T-shape, 4: cross)
    PIPE_I, PIPE_L, PIPE_T, PIPE_X = 1, 2, 3, 4
    PIPE_NAMES = {PIPE_I: "I-Pipe", PIPE_L: "L-Pipe", PIPE_T: "T-Pipe", PIPE_X: "X-Pipe"}
    
    # Directions (N, E, S, W)
    # Using complex numbers for easy rotation: val * 1j rotates 90 deg counter-clockwise
    DIRECTIONS = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
    DIR_VECTORS = {'N': -1j, 'E': 1, 'S': 1j, 'W': -1}
    VEC_DIRECTIONS = {v: k for k, v in DIR_VECTORS.items()}
    OPPOSITES = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    
    # Pipe connection definitions for each rotation (0-3)
    PIPE_CONNECTIONS = {
        PIPE_I: [['N', 'S'], ['E', 'W'], ['N', 'S'], ['E', 'W']],
        PIPE_L: [['N', 'E'], ['E', 'S'], ['S', 'W'], ['W', 'N']],
        PIPE_T: [['N', 'E', 'S'], ['E', 'S', 'W'], ['S', 'W', 'N'], ['W', 'N', 'E']],
        PIPE_X: [['N', 'E', 'S', 'W']] * 4,
    }

    MAX_STEPS = 500
    PARTICLE_SPEED = 2
    PARTICLE_LIFETIME = 200

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        # FIX: A display mode must be set for operations like .convert_alpha() to work,
        # even when running in headless mode.
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)

        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.rotations = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.flow_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.particles = []

        # We don't need to call reset here because the validation will do it.
        # However, to ensure all attributes are initialized before any other method
        # might be called, it's good practice.
        self.reset()
        # self.validate_implementation() # This is for debugging, not needed in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.loss_reason = ""
        
        self.grid.fill(0)
        self.rotations.fill(0)
        
        self._place_source_drain()

        self.pipe_inventory = self.np_random.integers(1, 5, size=20).tolist()
        self.current_pipe_idx = 0
        self.current_pipe_rotation = 0
        
        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self.invalid_move_flash = 0
        self.particles.clear()

        self._update_flow()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        action_taken = False

        # 1. Handle cursor movement
        if movement > 0:
            dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][movement]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_SIZE - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_SIZE - 1)
            action_taken = True
        
        # 2. Handle rotation
        if shift_pressed:
            self.current_pipe_rotation = (self.current_pipe_rotation + 1) % 4
            reward -= 0.1 # Small penalty for rotation
            action_taken = True

        # 3. Handle placement
        if space_pressed:
            action_taken = True
            if self._place_pipe():
                flow_len_before = np.sum(self.flow_grid)
                self._update_flow()
                flow_len_after = np.sum(self.flow_grid)
                reward += (flow_len_after - flow_len_before) * 1.0 # Reward for extending flow
                
                if self.flow_grid[self.drain_pos[1], self.drain_pos[0]]:
                    self.win = True
                    self.game_over = True
                    terminated = True
                    reward += 100 # Win bonus
            else:
                self.invalid_move_flash = 5 # Flash cursor for 5 frames
                reward -= 0.5 # Penalty for invalid move

        if not action_taken:
            reward -= 0.01 # Penalty for no-op

        # Check for termination conditions
        if not self.game_over:
            if self.steps >= self.MAX_STEPS:
                self.game_over = True
                terminated = True
                self.loss_reason = "Time Limit Reached"
                reward -= 50
            elif not self._has_valid_moves():
                self.game_over = True
                terminated = True
                self.loss_reason = "No Valid Moves"
                reward -= 100
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_source_drain(self):
        side_source = self.np_random.integers(0, 4)
        side_drain = (side_source + self.np_random.integers(1, 4)) % 4
        
        def get_pos(side):
            if side == 0: # Top
                return (self.np_random.integers(1, self.GRID_SIZE - 1), 0)
            elif side == 1: # Right
                return (self.GRID_SIZE - 1, self.np_random.integers(1, self.GRID_SIZE - 1))
            elif side == 2: # Bottom
                return (self.np_random.integers(1, self.GRID_SIZE - 1), self.GRID_SIZE - 1)
            else: # Left
                return (0, self.np_random.integers(1, self.GRID_SIZE - 1))

        self.source_pos = get_pos(side_source)
        self.drain_pos = get_pos(side_drain)
        
        # Ensure source and drain are not the same
        while self.source_pos == self.drain_pos:
            self.drain_pos = get_pos(side_drain)


    def _place_pipe(self):
        x, y = self.cursor_pos
        if not self.pipe_inventory:
            return False
        if self.grid[y, x] != 0:
            return False
        
        pipe_type = self.pipe_inventory[self.current_pipe_idx]
        if self._is_placement_valid(pipe_type, self.current_pipe_rotation, (x, y)):
            self.grid[y, x] = pipe_type
            self.rotations[y, x] = self.current_pipe_rotation
            self.pipe_inventory.pop(self.current_pipe_idx)
            if self.pipe_inventory:
                self.current_pipe_idx = self.current_pipe_idx % len(self.pipe_inventory)
            self.current_pipe_rotation = 0
            return True
        return False

    def _is_placement_valid(self, pipe_type, rotation, pos):
        x, y = pos
        connections = self.PIPE_CONNECTIONS[pipe_type][rotation]
        
        is_connected_to_flow = False
        
        # A pipe is valid if it connects to the source or any existing flowing pipe
        for direction in connections:
            dx, dy = self.DIRECTIONS[direction]
            nx, ny = x + dx, y + dy
            
            # Check connection to source
            if (nx, ny) == self.source_pos:
                is_connected_to_flow = True
                continue

            # Check connection to an existing flowing pipe
            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                if self.flow_grid[ny, nx] == 1:
                    neighbor_pipe = self.grid[ny, nx]
                    neighbor_rot = self.rotations[ny, nx]
                    neighbor_connections = self.PIPE_CONNECTIONS[neighbor_pipe][neighbor_rot]
                    if self.OPPOSITES[direction] in neighbor_connections:
                        is_connected_to_flow = True
        
        return is_connected_to_flow

    def _has_valid_moves(self):
        if not self.pipe_inventory:
            return False
        
        empty_cells = np.argwhere(self.grid == 0)
        unique_pipes = set(self.pipe_inventory)

        for y, x in empty_cells:
            for pipe_type in unique_pipes:
                for r in range(4):
                    if self._is_placement_valid(pipe_type, r, (x, y)):
                        return True
        return False

    def _update_flow(self):
        self.flow_grid.fill(0)
        q = deque()
        visited = {self.source_pos}
        
        # Find initial pipes connected to the source
        for direction, (dx, dy) in self.DIRECTIONS.items():
            nx, ny = self.source_pos[0] + dx, self.source_pos[1] + dy
            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                pipe_type = self.grid[ny, nx]
                if pipe_type != 0:
                    pipe_rot = self.rotations[ny, nx]
                    connections = self.PIPE_CONNECTIONS[pipe_type][pipe_rot]
                    if self.OPPOSITES[direction] in connections:
                        if (nx, ny) not in visited:
                            q.append((nx, ny))
                            visited.add((nx,ny))

        while q:
            x, y = q.popleft()
            self.flow_grid[y, x] = 1
            
            pipe_type = self.grid[y, x]
            pipe_rot = self.rotations[y, x]
            connections = self.PIPE_CONNECTIONS[pipe_type][pipe_rot]

            for direction in connections:
                dx, dy = self.DIRECTIONS[direction]
                nx, ny = x + dx, y + dy

                if (nx, ny) in visited or not (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE):
                    continue
                
                # Check connection to drain
                if (nx, ny) == self.drain_pos:
                    self.flow_grid[ny, nx] = 1 # Mark drain as having flow
                    visited.add((nx,ny))
                    continue

                neighbor_pipe = self.grid[ny, nx]
                if neighbor_pipe != 0:
                    neighbor_rot = self.rotations[ny, nx]
                    neighbor_connections = self.PIPE_CONNECTIONS[neighbor_pipe][neighbor_rot]
                    if self.OPPOSITES[direction] in neighbor_connections:
                        visited.add((nx,ny))
                        q.append((nx, ny))

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
            "pipes_left": len(self.pipe_inventory),
            "cursor_pos": self.cursor_pos.tolist(),
            "win": self.win
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_SIZE * self.CELL_SIZE))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE, y))

        # Update and draw particles
        self._update_and_draw_particles()

        # Draw source and drain
        self._draw_source_drain(self.source_pos, self.COLOR_SOURCE)
        self._draw_source_drain(self.drain_pos, self.COLOR_DRAIN)
        
        # Draw placed pipes
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                pipe_type = self.grid[y, x]
                if pipe_type != 0:
                    rotation = self.rotations[y, x]
                    has_flow = self.flow_grid[y, x] == 1
                    self._draw_pipe((x, y), pipe_type, rotation, has_flow)

        # Draw cursor
        cx, cy = self.cursor_pos
        rect = pygame.Rect(self.GRID_OFFSET_X + cx * self.CELL_SIZE, self.GRID_OFFSET_Y + cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        cursor_color = self.COLOR_CURSOR
        if self.invalid_move_flash > 0:
            cursor_color = self.COLOR_CURSOR_INVALID
            self.invalid_move_flash -= 1

        pygame.draw.rect(self.screen, cursor_color, rect, 3, border_radius=4)
        
        # Draw preview of pipe at cursor
        if self.pipe_inventory and self.grid[cy, cx] == 0:
            pipe_type = self.pipe_inventory[self.current_pipe_idx]
            self._draw_pipe((cx, cy), pipe_type, self.current_pipe_rotation, False, is_preview=True)

    def _draw_source_drain(self, pos, color):
        x, y = pos
        rect = pygame.Rect(
            self.GRID_OFFSET_X + x * self.CELL_SIZE + 5,
            self.GRID_OFFSET_Y + y * self.CELL_SIZE + 5,
            self.CELL_SIZE - 10,
            self.CELL_SIZE - 10
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
    def _draw_pipe(self, pos, pipe_type, rotation, has_flow, is_preview=False):
        x, y = pos
        center_x = self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
        
        color = self.COLOR_FLOW if has_flow else self.COLOR_PIPE
        if is_preview:
            color = (*color, 100) # Use alpha for preview
            
        connections = self.PIPE_CONNECTIONS[pipe_type][rotation]
        
        pipe_width = 8 if has_flow else 6
        
        # Draw center circle for T and X pipes
        if pipe_type in [self.PIPE_T, self.PIPE_X]:
            if is_preview:
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, pipe_width, color)
            else:
                pygame.draw.circle(self.screen, color, (center_x, center_y), pipe_width)

        for direction in connections:
            dx, dy = self.DIRECTIONS[direction]
            end_x = center_x + dx * self.CELL_SIZE // 2
            end_y = center_y + dy * self.CELL_SIZE // 2
            
            if is_preview:
                # To draw with alpha, create a temporary surface with SRCALPHA flag.
                temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(temp_surf, color, (center_x, center_y), (end_x, end_y), pipe_width*2)
                self.screen.blit(temp_surf, (0,0))
            else:
                 pygame.draw.line(self.screen, color, (center_x, center_y), (end_x, end_y), pipe_width*2)

    def _update_and_draw_particles(self):
        # Spawn new particles at source if connected
        for direction, (dx, dy) in self.DIRECTIONS.items():
            nx, ny = self.source_pos[0] + dx, self.source_pos[1] + dy
            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.flow_grid[ny, nx]:
                if self.np_random.random() < 0.3:
                    center_x = self.GRID_OFFSET_X + self.source_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
                    center_y = self.GRID_OFFSET_Y + self.source_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
                    self.particles.append({
                        'pos': np.array([center_x, center_y], dtype=float),
                        'vel': np.array([dx, dy], dtype=float) * self.PARTICLE_SPEED,
                        'life': self.PARTICLE_LIFETIME
                    })
        
        # Update and draw
        new_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            
            if p['life'] > 0:
                grid_x = int((p['pos'][0] - self.GRID_OFFSET_X) / self.CELL_SIZE)
                grid_y = int((p['pos'][1] - self.GRID_OFFSET_Y) / self.CELL_SIZE)

                if 0 <= grid_x < self.GRID_SIZE and 0 <= grid_y < self.GRID_SIZE and self.flow_grid[grid_y, grid_x]:
                    new_particles.append(p)
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, (*self.COLOR_FLOW, 200))
                    pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, (*self.COLOR_FLOW, 200))

                    # Logic to change velocity at pipe junctions
                    cell_center_x = self.GRID_OFFSET_X + grid_x * self.CELL_SIZE + self.CELL_SIZE // 2
                    cell_center_y = self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
                    if np.linalg.norm(p['pos'] - np.array([cell_center_x, cell_center_y])) < self.PARTICLE_SPEED:
                        p['pos'] = np.array([cell_center_x, cell_center_y], dtype=float) # Snap to center
                        
                        pipe_type = self.grid[grid_y, grid_x]
                        rotation = self.rotations[grid_y, grid_x]
                        connections = self.PIPE_CONNECTIONS[pipe_type][rotation]
                        
                        in_dir_vec = -p['vel'] / np.linalg.norm(p['vel'])
                        in_dir_complex = round(in_dir_vec[0]) + round(in_dir_vec[1]) * 1j
                        in_dir = self.VEC_DIRECTIONS.get(in_dir_complex, 'N')
                        
                        out_dirs = [d for d in connections if d != in_dir]
                        if out_dirs:
                            next_dir = self.np_random.choice(out_dirs)
                            dv = self.DIR_VECTORS[next_dir]
                            p['vel'] = np.array([dv.real, dv.imag]) * self.PARTICLE_SPEED

        self.particles = new_particles

    def _render_ui(self):
        ui_x = self.GRID_OFFSET_X * 2 + self.GRID_SIZE * self.CELL_SIZE
        
        # Draw Score
        self._draw_text(f"SCORE: {int(self.score)}", (ui_x, 40), self.font_medium)
        
        # Draw Pipes Left
        self._draw_text(f"PIPES: {len(self.pipe_inventory)}", (ui_x, 80), self.font_medium)
        
        # Draw Next Pipe preview
        self._draw_text("NEXT PIPE:", (ui_x, 150), self.font_medium)
        if self.pipe_inventory:
            pipe_type = self.pipe_inventory[self.current_pipe_idx]
            self._draw_preview_pipe((ui_x + 60, 220), pipe_type, self.current_pipe_rotation)
            pipe_name = self.PIPE_NAMES.get(pipe_type, "Unknown")
            self._draw_text(pipe_name, (ui_x, 280), self.font_small)

        # Draw Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            self._draw_text(message, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 40), self.font_large, center=True)
            if self.loss_reason:
                 self._draw_text(self.loss_reason, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20), self.font_medium, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        shadow_surf = font.render(text, True, shadow_color)
        text_surf = font.render(text, True, color)
        
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
            
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _draw_preview_pipe(self, center_pos, pipe_type, rotation):
        center_x, center_y = center_pos
        pipe_width = 12
        
        connections = self.PIPE_CONNECTIONS[pipe_type][rotation]
        
        if pipe_type in [self.PIPE_T, self.PIPE_X]:
            pygame.draw.circle(self.screen, self.COLOR_PIPE, (center_x, center_y), pipe_width // 2)

        for direction in connections:
            dx, dy = self.DIRECTIONS[direction]
            end_x = center_x + dx * self.CELL_SIZE // 1.5
            end_y = center_y + dy * self.CELL_SIZE // 1.5
            pygame.draw.line(self.screen, self.COLOR_PIPE, (center_x, center_y), (end_x, end_y), pipe_width)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will create a visible window for playing.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Re-initialize pygame with a visible display
    pygame.display.quit()
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pipe Dream")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        # Use pygame.event.get() for responsive key presses
        action_taken_this_frame = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action_taken_this_frame = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift = 1
        
        if action_taken_this_frame:
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Win: {info['win']}")
                # Render final frame
                frame = np.transpose(obs, (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame)
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                # Wait for a moment before allowing reset
                pygame.time.wait(2000)
                obs, info = env.reset()

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate for human play

    env.close()