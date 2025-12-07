import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:04:33.400808
# Source Brief: brief_00795.md
# Brief Index: 795
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
    A Gymnasium environment for a musical puzzle game.
    The agent navigates a procedurally generated dungeon, clearing obstacles
    by inputting correct melodic sequences.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Navigate a musical dungeon and clear obstacles by replaying their melodic sequences."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move. Press Shift to cycle through notes and Space to play a note at an adjacent obstacle."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 12
    CELL_SIZE = 32

    # Colors (Vibrant, high contrast)
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_OBSTACLE = (80, 90, 110)
    COLOR_EXIT = (255, 0, 128)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SUCCESS = (0, 255, 128)
    COLOR_FAIL = (255, 50, 50)
    NOTE_COLORS = [
        (50, 150, 255),   # Blue
        (255, 200, 50),    # Yellow
        (100, 255, 100),   # Green
        (255, 100, 100),   # Red
    ]
    NUM_NOTES = len(NOTE_COLORS)

    MAX_EPISODE_STEPS = 5000
    LEVEL_MOVE_LIMIT = 500

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
        self.font_ui = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_game_over = pygame.font.SysFont('Consolas', 50, bold=True)

        # --- State Variables ---
        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 0
        
        self.player_pos = [0, 0]
        self.grid = []
        self.obstacles = {}
        self.exit_pos = [0, 0]
        
        self.selected_note_id = 0
        self.current_input_sequence = []
        self.target_obstacle_pos = None
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.particles = []
        self.text_popups = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.LEVEL_MOVE_LIMIT
        
        self._generate_level()

        self.selected_note_id = 0
        self.current_input_sequence = []
        self.target_obstacle_pos = None

        self.prev_space_held = False
        self.prev_shift_held = False

        self.particles = []
        self.text_popups = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        action_taken = False

        # --- Action Handling ---
        # 1. Note Selection (Shift)
        if shift_pressed:
            self.selected_note_id = (self.selected_note_id + 1) % self.NUM_NOTES
            # SFX: Note select tick
            action_taken = True

        # 2. Movement
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

            if self._is_valid_move(new_pos):
                self.player_pos = new_pos
                self.current_input_sequence = []
                self.target_obstacle_pos = None
                action_taken = True
                # SFX: Player move step
                self._add_particles(self.player_pos, 5, self.COLOR_PLAYER, life=10, trail=True)

        # 3. Note Input (Space)
        if space_pressed:
            adjacent_obstacle = self._get_adjacent_obstacle()
            if adjacent_obstacle:
                self.target_obstacle_pos = adjacent_obstacle
                self.current_input_sequence.append(self.selected_note_id)
                action_taken = True
                
                obstacle_pattern = self.obstacles[self.target_obstacle_pos]
                
                # Check for match/mismatch
                if len(self.current_input_sequence) == len(obstacle_pattern):
                    if self.current_input_sequence == obstacle_pattern:
                        # Success!
                        reward += 1.0
                        self.score += 10
                        del self.obstacles[self.target_obstacle_pos]
                        self.grid[self.target_obstacle_pos[1]][self.target_obstacle_pos[0]] = 0
                        self._add_particles(self.target_obstacle_pos, 50, self.COLOR_SUCCESS, life=40)
                        self.text_popups.append([f"+10", self.target_obstacle_pos, 30, self.COLOR_SUCCESS])
                        # SFX: Success chime
                    else:
                        # Failure
                        reward -= 0.1
                        self.score -= 1
                        self._add_particles(self.target_obstacle_pos, 20, self.COLOR_FAIL, life=20)
                        self.text_popups.append([f"-1", self.target_obstacle_pos, 30, self.COLOR_FAIL])
                        # SFX: Failure buzz
                    
                    self.current_input_sequence = []
                    self.target_obstacle_pos = None
                else:
                    # SFX: Note play sound
                    pass # Continue sequence
        
        if action_taken:
            self.moves_remaining -= 1

        self._update_world()
        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        # --- Termination Check ---
        terminated = False
        if tuple(self.player_pos) == self.exit_pos:
            reward += 10.0
            self.score += 100
            if self.moves_remaining > 0:
                reward += 50.0
                self.score += self.moves_remaining
            terminated = True
            self.game_over = True
            # SFX: Level complete fanfare

        if self.moves_remaining <= 0 or self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over = True
            # SFX: Game over sound

        self.steps += 1

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_world(self):
        # Update particles
        new_particles = []
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
            if p[2] > 0:
                new_particles.append(p)
        self.particles = new_particles
        
        # Update text popups
        new_popups = []
        for t in self.text_popups:
            t[2] -= 1
            if t[2] > 0:
                new_popups.append(t)
        self.text_popups = new_popups


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_exit()
        self._render_obstacles()
        self._render_particles()
        self._render_player()
        self._render_input_feedback()
        self._render_text_popups()
        self._render_ui()

        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_remaining}

    def _grid_to_screen(self, grid_pos):
        x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2 + grid_pos[0] * self.CELL_SIZE
        y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2 + grid_pos[1] * self.CELL_SIZE
        return int(x), int(y)

    # --- Rendering Methods ---

    def _render_background(self):
        offset_x, offset_y = self._grid_to_screen((0, 0))
        for r in range(self.GRID_HEIGHT + 1):
            y = offset_y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (offset_x, y), (offset_x + self.GRID_WIDTH * self.CELL_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = offset_x + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, offset_y), (x, offset_y + self.GRID_HEIGHT * self.CELL_SIZE))

    def _render_exit(self):
        screen_pos = self._grid_to_screen(self.exit_pos)
        center_x = screen_pos[0] + self.CELL_SIZE // 2
        center_y = screen_pos[1] + self.CELL_SIZE // 2
        
        # Pulsing glow effect
        radius = self.CELL_SIZE * 0.4 + math.sin(self.steps * 0.1) * 3
        for i in range(10, 0, -1):
            alpha = 150 - i * 15
            color = (*self.COLOR_EXIT, alpha)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(radius + i), color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(radius), self.COLOR_EXIT)

    def _render_obstacles(self):
        note_radius = self.CELL_SIZE // 8
        for pos, pattern in self.obstacles.items():
            screen_pos = self._grid_to_screen(pos)
            rect = pygame.Rect(screen_pos[0], screen_pos[1], self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=4)
            
            # Draw melodic pattern
            num_notes = len(pattern)
            for i, note_id in enumerate(pattern):
                x = screen_pos[0] + self.CELL_SIZE // 2 + (i - (num_notes - 1) / 2) * (note_radius * 2.5)
                y = screen_pos[1] + self.CELL_SIZE // 2
                pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), note_radius, self.NOTE_COLORS[note_id])
                pygame.gfxdraw.aacircle(self.screen, int(x), int(y), note_radius, (255,255,255))

    def _render_player(self):
        screen_pos = self._grid_to_screen(self.player_pos)
        center_x = screen_pos[0] + self.CELL_SIZE // 2
        center_y = screen_pos[1] + self.CELL_SIZE // 2
        radius = self.CELL_SIZE // 3
        
        # Selected note color glow
        glow_color = self.NOTE_COLORS[self.selected_note_id]
        for i in range(8, 0, -1):
            alpha = 100 - i * 12
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius + i, (*glow_color, alpha))
            
        # Player body
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_GRID)

    def _render_input_feedback(self):
        if self.target_obstacle_pos and self.current_input_sequence:
            screen_pos = self._grid_to_screen(self.target_obstacle_pos)
            note_radius = self.CELL_SIZE // 8
            num_notes = len(self.current_input_sequence)
            
            for i, note_id in enumerate(self.current_input_sequence):
                x = screen_pos[0] + self.CELL_SIZE // 2 + (i - (num_notes - 1) / 2) * (note_radius * 2.5)
                y = screen_pos[1] + self.CELL_SIZE * 0.75 # Below the pattern
                color = self.NOTE_COLORS[note_id]
                pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), note_radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(x), int(y), note_radius, self.COLOR_PLAYER)

    def _render_particles(self):
        for pos, vel, life, color, trail in self.particles:
            grid_pos = [pos[0], pos[1]]
            if not trail:
                grid_pos = self._grid_to_screen(pos)
                grid_pos = [grid_pos[0] + self.CELL_SIZE/2, grid_pos[1] + self.CELL_SIZE/2]

            radius = int(max(0, life / 5))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(grid_pos[0]), int(grid_pos[1]), radius, color)

    def _render_text_popups(self):
        for text, pos, life, color in self.text_popups:
            screen_pos = self._grid_to_screen(pos)
            alpha = min(255, int(life * 255 / 20))
            text_surf = self.font_ui.render(text, True, color)
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(screen_pos[0] + self.CELL_SIZE//2, screen_pos[1] - (30 - life)))
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        moves_text = self.font_ui.render(f"MOVES: {self.moves_remaining}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(moves_text, moves_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if tuple(self.player_pos) == self.exit_pos:
            text = "LEVEL COMPLETE"
            color = self.COLOR_SUCCESS
        else:
            text = "GAME OVER"
            color = self.COLOR_FAIL
            
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        overlay.blit(text_surf, text_rect)
        self.screen.blit(overlay, (0, 0))

    # --- Game Logic ---

    def _generate_level(self):
        # 1: Grid of walls
        self.grid = [[1 for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.obstacles = {}

        # 2: Carve path with Randomized DFS
        start_x, start_y = (1, self.GRID_HEIGHT // 2)
        self.player_pos = [start_x, start_y]
        stack = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        self.grid[start_y][start_x] = 0

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                # Carve path
                self.grid[ny][nx] = 0
                self.grid[cy + (ny - cy) // 2][cx + (nx - cx) // 2] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # 3: Set exit
        self.exit_pos = (self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2)
        self.grid[self.exit_pos[1]][self.exit_pos[0]] = 0 # Ensure exit is clear

        # 4: Populate obstacles and patterns
        pattern_length = min(self.NUM_NOTES, 2 + (self.level - 1) // 5)
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] == 1:
                    pos = (c, r)
                    self.obstacles[pos] = random.choices(range(self.NUM_NOTES), k=pattern_length)
    
    def _is_valid_move(self, pos):
        x, y = pos
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
            return False
        if self.grid[y][x] == 1:
            return False
        return True

    def _get_adjacent_obstacle(self):
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            check_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            if check_pos in self.obstacles:
                return check_pos
        return None

    def _add_particles(self, grid_pos, count, color, life, trail=False):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            
            if trail:
                pos = self._grid_to_screen(grid_pos)
                pos = [pos[0] + self.CELL_SIZE/2, pos[1] + self.CELL_SIZE/2]
                self.particles.append([list(pos), vel, random.randint(life//2, life), color, True])
            else:
                self.particles.append([list(grid_pos), vel, random.randint(life//2, life), color, False])

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    
    # The main script can create a display, but the env itself is headless
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Musical Dungeon")
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    
    while running:
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            # Optional: Add a delay and reset on game over
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        clock.tick(30) # Run at 30 FPS for smooth feel

    pygame.quit()