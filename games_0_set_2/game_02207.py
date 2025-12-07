
# Generated: 2025-08-28T04:06:04.890801
# Source Brief: brief_02207.md
# Brief Index: 2207

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Shift to examine adjacent objects. Press Space to interact."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape the procedurally generated Midnight Manor. Solve puzzles and avoid traps to find the exit before your health runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Game world
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 9
    TILE_WIDTH_ISO, TILE_HEIGHT_ISO = 64, 32
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 100

    # Tile types
    T_EMPTY, T_WALL, T_PUZZLE, T_TRAP, T_EXIT = 0, 1, 2, 3, 4

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_FLOOR = (25, 30, 50)
    COLOR_FLOOR_GRID = (35, 40, 65)
    COLOR_WALL = (60, 65, 90)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_SHADOW = (0, 0, 0, 100)
    COLOR_EXIT = (120, 0, 180)
    COLOR_TRAP_INACTIVE = (200, 50, 50)
    COLOR_TRAP_ACTIVE = (255, 80, 80)
    COLOR_PUZZLE_UNSOLVED = (0, 100, 255)
    COLOR_PUZZLE_SOLVED = (0, 255, 100)
    COLOR_INTERACT_HINT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 150)
    COLOR_HEALTH_BAR = (200, 0, 0)
    COLOR_HEALTH_BAR_BG = (50, 0, 0)

    MAX_STEPS = 1000
    MAX_HEALTH = 10

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
        
        self.ui_font = pygame.font.SysFont("Consolas", 16, bold=True)
        self.hint_font = pygame.font.SysFont("Consolas", 12)

        # These will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_health = 0
        self.grid = None
        self.puzzle_state = {}
        self.visited_tiles = set()
        self.rng = None
        self.last_reward = 0
        self.examining_pos = None
        self.interaction_feedback = {} # pos: (color, timer)
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.MAX_HEALTH
        self.last_reward = 0
        self.interaction_feedback = {}
        
        self._generate_room()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.examining_pos = None

        movement = action[0]
        space_pressed = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Handle Actions ---
        # 1. Examination (Shift)
        if shift_held:
            self._handle_examine()
            
        # 2. Interaction (Space)
        if space_pressed:
            reward += self._handle_interaction()

        # 3. Movement
        moved = self._handle_movement(movement)
        if moved:
            if tuple(self.player_pos) not in self.visited_tiles:
                reward += 0.1 # Reward for exploration
                self.visited_tiles.add(tuple(self.player_pos))

        self.score += reward
        self.last_reward = reward
        
        terminated = self._check_termination()
        if terminated and not self.game_over: # Apply terminal rewards only once
            self.game_over = True
            if self.player_health <= 0:
                reward -= 100
            elif self._is_on_tile_type(self.T_EXIT): # Check if termination was due to exit
                 reward += 100
            self.score += reward - self.last_reward # Adjust score with terminal reward
            self.last_reward = reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_examine(self):
        target_pos = self._get_adjacent_interactive_pos()
        if target_pos:
            self.examining_pos = target_pos

    def _handle_interaction(self):
        target_pos = self._get_adjacent_interactive_pos()
        if not target_pos:
            return 0
        
        tx, ty = target_pos
        tile_type = self.grid[ty][tx]
        reward = 0

        if tile_type == self.T_TRAP:
            # sound: trap_spring.wav
            self.player_health = max(0, self.player_health - 2)
            reward = -2
            self.interaction_feedback[target_pos] = (self.COLOR_TRAP_ACTIVE, 15)
        elif tile_type == self.T_PUZZLE:
            # sound: puzzle_interact.wav
            if not self.puzzle_state.get(target_pos, False):
                self.puzzle_state[target_pos] = True
                reward = 5
                self.interaction_feedback[target_pos] = (self.COLOR_PUZZLE_SOLVED, 15)
        elif tile_type == self.T_EXIT:
            # sound: door_open.wav
            # Termination logic will handle the large reward
            pass
        
        return reward

    def _handle_movement(self, movement_action):
        px, py = self.player_pos
        next_pos = list(self.player_pos)

        if movement_action == 1: # Up
            next_pos[1] -= 1
        elif movement_action == 2: # Down
            next_pos[1] += 1
        elif movement_action == 3: # Left
            next_pos[0] -= 1
        elif movement_action == 4: # Right
            next_pos[0] += 1
        else: # No-op
            return False

        nx, ny = next_pos
        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny][nx] != self.T_WALL:
            self.player_pos = next_pos
            # sound: footstep.wav
            return True
        return False

    def _get_adjacent_interactive_pos(self):
        px, py = self.player_pos
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Up, Down, Left, Right
            nx, ny = px + dx, py + dy
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                tile_type = self.grid[ny][nx]
                if tile_type in [self.T_PUZZLE, self.T_TRAP, self.T_EXIT]:
                    return (nx, ny)
        return None

    def _check_termination(self):
        if self.player_health <= 0:
            return True
        if self._is_on_tile_type(self.T_EXIT):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False
    
    def _is_on_tile_type(self, tile_type):
        px, py = self.player_pos
        return self.grid[py][px] == tile_type

    def _generate_room(self):
        self.grid = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), self.T_EMPTY, dtype=int)
        
        # Place walls on borders
        self.grid[0, :] = self.T_WALL
        self.grid[-1, :] = self.T_WALL
        self.grid[:, 0] = self.T_WALL
        self.grid[:, -1] = self.T_WALL
        
        # Get all possible empty locations
        empty_tiles = []
        for y in range(1, self.GRID_HEIGHT - 1):
            for x in range(1, self.GRID_WIDTH - 1):
                empty_tiles.append((x, y))

        self.rng.shuffle(empty_tiles)
        
        # Place player
        self.player_pos = list(empty_tiles.pop())
        self.visited_tiles = {tuple(self.player_pos)}

        # Place exit
        exit_pos = empty_tiles.pop()
        self.grid[exit_pos[1]][exit_pos[0]] = self.T_EXIT
        
        # Place puzzle
        puzzle_pos = empty_tiles.pop()
        self.grid[puzzle_pos[1]][puzzle_pos[0]] = self.T_PUZZLE
        self.puzzle_state = {puzzle_pos: False}

        # Place traps
        num_traps = min(len(empty_tiles), 1 + self.steps // 50)
        for _ in range(num_traps):
            trap_pos = empty_tiles.pop()
            self.grid[trap_pos[1]][trap_pos[0]] = self.T_TRAP

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * (self.TILE_WIDTH_ISO / 2)
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * (self.TILE_HEIGHT_ISO / 2)
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, surface, grid_x, grid_y, color, height=0):
        sx, sy = self._iso_to_screen(grid_x, grid_y)
        sy -= int(height)
        points = [
            (sx, sy - self.TILE_HEIGHT_ISO / 2),
            (sx + self.TILE_WIDTH_ISO / 2, sy),
            (sx, sy + self.TILE_HEIGHT_ISO / 2),
            (sx - self.TILE_WIDTH_ISO / 2, sy),
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, (*color[:3], 150)) # Anti-aliased border

    def _draw_iso_cube(self, surface, grid_x, grid_y, color, height):
        sx, sy = self._iso_to_screen(grid_x, grid_y)
        
        # Top face
        top_points = [
            (sx, sy - height - self.TILE_HEIGHT_ISO / 2),
            (sx + self.TILE_WIDTH_ISO / 2, sy - height),
            (sx, sy - height + self.TILE_HEIGHT_ISO / 2),
            (sx - self.TILE_WIDTH_ISO / 2, sy - height),
        ]
        
        # Calculate shadow/highlight colors
        darker_color = tuple(max(0, c - 40) for c in color[:3])
        darkest_color = tuple(max(0, c - 60) for c in color[:3])

        # Right face
        right_points = [
            (sx, sy + self.TILE_HEIGHT_ISO / 2),
            (sx + self.TILE_WIDTH_ISO / 2, sy),
            (sx + self.TILE_WIDTH_ISO / 2, sy - height),
            (sx, sy - height + self.TILE_HEIGHT_ISO / 2),
        ]
        pygame.gfxdraw.filled_polygon(surface, right_points, darker_color)

        # Left face
        left_points = [
            (sx, sy + self.TILE_HEIGHT_ISO / 2),
            (sx - self.TILE_WIDTH_ISO / 2, sy),
            (sx - self.TILE_WIDTH_ISO / 2, sy - height),
            (sx, sy - height + self.TILE_HEIGHT_ISO / 2),
        ]
        pygame.gfxdraw.filled_polygon(surface, left_points, darkest_color)

        # Draw top face last
        pygame.gfxdraw.filled_polygon(surface, top_points, color)
        pygame.gfxdraw.aapolygon(surface, top_points, tuple(min(255, c + 50) for c in color[:3]))
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Update feedback timers
        new_feedback = {}
        for pos, (color, timer) in self.interaction_feedback.items():
            if timer > 0:
                new_feedback[pos] = (color, timer - 1)
        self.interaction_feedback = new_feedback

        # Render grid from back to front
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Draw floor tile
                self._draw_iso_tile(self.screen, x, y, self.COLOR_FLOOR)
                
                tile_type = self.grid[y][x]
                
                # Draw player shadow
                if (x, y) == tuple(self.player_pos):
                    sx, sy = self._iso_to_screen(x, y)
                    shadow_surf = pygame.Surface((self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO), pygame.SRCALPHA)
                    pygame.draw.ellipse(shadow_surf, self.COLOR_PLAYER_SHADOW, [0, 0, self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO])
                    self.screen.blit(shadow_surf, (sx - self.TILE_WIDTH_ISO//2, sy - self.TILE_HEIGHT_ISO//2))

                # Draw objects
                if tile_type == self.T_WALL:
                    self._draw_iso_cube(self.screen, x, y, self.COLOR_WALL, self.TILE_HEIGHT_ISO)
                elif tile_type == self.T_EXIT:
                    self._draw_iso_cube(self.screen, x, y, self.COLOR_EXIT, self.TILE_HEIGHT_ISO * 0.75)
                elif tile_type == self.T_TRAP:
                    color = self.COLOR_TRAP_INACTIVE
                    if (x,y) in self.interaction_feedback:
                        color = self.interaction_feedback[(x,y)][0]
                    self._draw_iso_tile(self.screen, x, y, color, height=2)
                elif tile_type == self.T_PUZZLE:
                    is_solved = self.puzzle_state.get((x, y), False)
                    color = self.COLOR_PUZZLE_SOLVED if is_solved else self.COLOR_PUZZLE_UNSOLVED
                    if (x,y) in self.interaction_feedback and not is_solved:
                        color = self.interaction_feedback[(x,y)][0]
                    self._draw_iso_cube(self.screen, x, y, color, self.TILE_HEIGHT_ISO * 0.25)
                
                # Draw player
                if (x, y) == tuple(self.player_pos):
                    self._draw_iso_cube(self.screen, x, y, self.COLOR_PLAYER, self.TILE_HEIGHT_ISO * 0.5)
        
        # Render examination hints
        if self.examining_pos:
            ex, ey = self.examining_pos
            tile_type = self.grid[ey][ex]
            hint_text = ""
            if tile_type == self.T_TRAP: hint_text = "[Looks dangerous!]"
            elif tile_type == self.T_PUZZLE: hint_text = "[???]" if not self.puzzle_state.get((ex,ey)) else "[Solved]"
            elif tile_type == self.T_EXIT: hint_text = "[The Way Out]"
            
            if hint_text:
                text_surf = self.hint_font.render(hint_text, True, self.COLOR_INTERACT_HINT)
                sx, sy = self._iso_to_screen(ex, ey)
                text_rect = text_surf.get_rect(center=(sx, sy - self.TILE_HEIGHT_ISO))
                self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Health bar
        bar_width = 200
        bar_height = 20
        health_ratio = self.player_health / self.MAX_HEALTH
        
        bg_rect = pygame.Rect(10, 10, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect)
        
        health_rect = pygame.Rect(10, 10, int(bar_width * health_ratio), bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, health_rect)
        pygame.draw.rect(self.screen, self.COLOR_INTERACT_HINT, bg_rect, 1)

        health_text = self.ui_font.render(f"HEALTH: {self.player_health}/{self.MAX_HEALTH}", True, self.COLOR_INTERACT_HINT)
        self.screen.blit(health_text, (15, 12))

        # Score and Steps
        score_text = self.ui_font.render(f"SCORE: {self.score:.1f}", True, self.COLOR_INTERACT_HINT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - 150, 10))
        
        steps_text = self.ui_font.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_INTERACT_HINT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - 150, 30))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "player_pos": self.player_pos,
            "puzzles_solved": sum(self.puzzle_state.values())
        }

    def close(self):
        pygame.quit()

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

# Example of how to run the environment
if __name__ == "__main__":
    import os
    # Set a dummy video driver to run pygame headlessly for the main process
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    
    # --- To play the game manually ---
    # Un-comment the following block to control the game with your keyboard.
    # Note: This requires a display. You may need to remove the os.environ line above.
    """
    import sys
    # os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS" etc.
    # del os.environ['SDL_VIDEODRIVER'] # Or this one

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Midnight Manor")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    while not terminated:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the display screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Wait for the next action
        pygame.time.wait(100) # control game speed for manual play

    env.close()
    pygame.quit()
    sys.exit()
    """

    # --- To run a simple agent ---
    obs, info = env.reset()
    total_reward = 0
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i+1) % 100 == 0:
            print(f"Step {i+1}: Action: {action}, Reward: {reward:.2f}, Info: {info}")
        if terminated:
            print(f"Episode finished after {i+1} steps. Final Score: {info['score']:.2f}")
            break
    env.close()