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
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. "
        "Each move costs 1 turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a maze to the green exit. You have a limited number of moves. "
        "Red zones give points but risk a penalty. Blue zones cost points but are safe."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_WALL = (70, 80, 90)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_EXIT = (0, 255, 100)
    COLOR_RISK = (255, 50, 50)
    COLOR_SAFE = (50, 100, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)

    # Maze & Screen Dimensions
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    CELL_SIZE = 36
    MAZE_WIDTH = GRID_SIZE * CELL_SIZE
    MAZE_HEIGHT = GRID_SIZE * CELL_SIZE
    OFFSET_X = (SCREEN_WIDTH - MAZE_WIDTH) // 2
    OFFSET_Y = (SCREEN_HEIGHT - MAZE_HEIGHT) // 2

    # Game Parameters
    INITIAL_MOVES = 50
    MAX_STEPS = 500  # Failsafe episode termination
    NUM_RISK_ZONES = 5
    NUM_SAFE_ZONES = 3
    TRAP_CHANCE = 0.25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables
        self.maze = []
        self.player_pos = (0, 0)
        self.exit_pos = (self.GRID_SIZE - 1, self.GRID_SIZE - 1)
        self.risk_zones = []
        self.safe_zones = []
        self.moves_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_trap_feedback = 0
        self.last_reward_feedback = 0
        
        # This will be properly initialized in reset()
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.INITIAL_MOVES
        self.game_over = False
        self.player_pos = (0, 0)
        self.last_trap_feedback = 0
        self.last_reward_feedback = 0

        # Generate a new maze and place zones
        self._generate_maze()
        self._place_special_zones()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.steps += 1
        reward = 0
        terminated = False
        
        # A turn is consumed regardless of action (even no-op)
        self.moves_remaining -= 1
        reward -= 1  # Cost for taking a turn

        # --- Process Movement ---
        if movement != 0:
            current_x, current_y = self.player_pos
            next_x, next_y = current_x, current_y

            if movement == 1:  # Up
                next_y -= 1
            elif movement == 2:  # Down
                next_y += 1
            elif movement == 3:  # Left
                next_x -= 1
            elif movement == 4:  # Right
                next_x += 1

            if self._is_valid_move((current_x, current_y), (next_x, next_y)):
                self.player_pos = (next_x, next_y)

        # --- Process Zone Interactions ---
        if self.player_pos in self.risk_zones:
            reward += 2
            self.score += 2
            self.last_reward_feedback = 2
            if self.np_random.random() < self.TRAP_CHANCE:
                reward -= 5
                self.score -= 5
                self.last_trap_feedback = self.steps # for visual feedback
        elif self.player_pos in self.safe_zones:
            reward -= 1
            self.score -= 1
            self.last_reward_feedback = -1
        else:
            self.last_reward_feedback = 0

        # --- Check Termination Conditions ---
        if self.player_pos == self.exit_pos:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
        elif self.moves_remaining <= 0:
            reward -= 10
            self.score -= 10
            terminated = True
            self.game_over = True
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True


        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

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
            "moves_remaining": self.moves_remaining,
            "player_pos": self.player_pos,
        }

    def _render_text(self, text, font, color, pos, shadow=False):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        # Moves Remaining
        self._render_text(
            f"Moves: {self.moves_remaining}", self.font_large, self.COLOR_TEXT, (15, 10), shadow=True
        )
        # Score
        score_text = f"Score: {self.score}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        self._render_text(
            score_text, self.font_large, self.COLOR_TEXT, (self.SCREEN_WIDTH - score_surf.get_width() - 15, 10), shadow=True
        )

        # Trap feedback
        if self.last_trap_feedback > 0 and self.steps - self.last_trap_feedback < 30:
            alpha = max(0, 255 - (self.steps - self.last_trap_feedback) * 15)
            feedback_surf = self.font_large.render("TRAP! (-5)", True, self.COLOR_RISK)
            feedback_surf.set_alpha(alpha)
            pos_x = self.SCREEN_WIDTH // 2 - feedback_surf.get_width() // 2
            pos_y = self.SCREEN_HEIGHT - 50
            self.screen.blit(feedback_surf, (pos_x, pos_y))
            
    def _render_game(self):
        # --- Draw Maze Elements ---
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                px, py = self.OFFSET_X + x * self.CELL_SIZE, self.OFFSET_Y + y * self.CELL_SIZE
                
                # Draw zones first to be underneath walls
                if (x, y) in self.risk_zones:
                    pulse = (math.sin(self.steps * 0.3) + 1) / 2  # 0 to 1
                    alpha = 50 + pulse * 100
                    pygame.gfxdraw.box(self.screen, (px, py, self.CELL_SIZE, self.CELL_SIZE), (*self.COLOR_RISK, int(alpha)))
                
                if (x, y) in self.safe_zones:
                    pulse = (math.sin(self.steps * 0.2) + 1) / 2
                    radius = int(self.CELL_SIZE * 0.7 * (0.8 + pulse * 0.2))
                    alpha = 40 + pulse * 40
                    pygame.gfxdraw.filled_circle(self.screen, px + self.CELL_SIZE//2, py + self.CELL_SIZE//2, radius, (*self.COLOR_SAFE, int(alpha)))
                    pygame.gfxdraw.aacircle(self.screen, px + self.CELL_SIZE//2, py + self.CELL_SIZE//2, radius, (*self.COLOR_SAFE, int(alpha)))

                # Draw exit
                if (x, y) == self.exit_pos:
                    pygame.draw.rect(self.screen, self.COLOR_EXIT, (px + 4, py + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8))
                    
                # Draw walls
                if self.maze[y][x]['N']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.CELL_SIZE, py), 3)
                if self.maze[y][x]['S']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.CELL_SIZE), (px + self.CELL_SIZE, py + self.CELL_SIZE), 3)
                if self.maze[y][x]['W']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + self.CELL_SIZE), 3)
                if self.maze[y][x]['E']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.CELL_SIZE, py), (px + self.CELL_SIZE, py + self.CELL_SIZE), 3)

        # --- Draw Player ---
        player_px = self.OFFSET_X + self.player_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        player_py = self.OFFSET_Y + self.player_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        radius = self.CELL_SIZE // 2 - 6
        
        # Glow effect
        pulse = (math.sin(self.steps * 0.25) + 1) / 2
        glow_radius = radius + 3 + pulse * 3
        glow_alpha = 100 - pulse * 40
        pygame.gfxdraw.filled_circle(self.screen, player_px, player_py, int(glow_radius), (*self.COLOR_PLAYER, int(glow_alpha)))
        
        # Player circle
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (player_px, player_py), radius)
        pygame.draw.circle(self.screen, self.COLOR_BG, (player_px, player_py), radius - 3)

    def _generate_maze(self):
        # Initialize grid with all walls up
        self.maze = [{'N': True, 'S': True, 'E': True, 'W': True} for _ in range(self.GRID_SIZE * self.GRID_SIZE)]
        self.maze = [list(row) for row in np.reshape(self.maze, (self.GRID_SIZE, self.GRID_SIZE))]
        
        visited = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        stack = [(0, 0)]
        visited[0, 0] = True

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            # Check neighbors
            if cy > 0 and not visited[cy - 1, cx]: neighbors.append(('N', (cx, cy - 1)))
            if cy < self.GRID_SIZE - 1 and not visited[cy + 1, cx]: neighbors.append(('S', (cx, cy + 1)))
            if cx > 0 and not visited[cy, cx - 1]: neighbors.append(('W', (cx - 1, cy)))
            if cx < self.GRID_SIZE - 1 and not visited[cy, cx + 1]: neighbors.append(('E', (cx + 1, cy)))

            if neighbors:
                # Use self.np_random for reproducibility
                direction_idx = self.np_random.integers(len(neighbors))
                direction, (nx, ny) = neighbors[direction_idx]
                
                # Knock down walls
                if direction == 'N':
                    self.maze[cy][cx]['N'] = False
                    self.maze[ny][nx]['S'] = False
                elif direction == 'S':
                    self.maze[cy][cx]['S'] = False
                    self.maze[ny][nx]['N'] = False
                elif direction == 'W':
                    self.maze[cy][cx]['W'] = False
                    self.maze[ny][nx]['E'] = False
                elif direction == 'E':
                    self.maze[cy][cx]['E'] = False
                    self.maze[ny][nx]['W'] = False
                
                visited[ny, nx] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        # This guarantees a solvable maze from (0,0) to any other cell.

    def _place_special_zones(self):
        self.risk_zones.clear()
        self.safe_zones.clear()
        
        # Get all possible floor cells, excluding start and end
        possible_cells = []
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if (x, y) != (0, 0) and (x, y) != self.exit_pos:
                    possible_cells.append((x, y))
        
        # Shuffle and pick zones
        self.np_random.shuffle(possible_cells)
        
        num_risk = min(self.NUM_RISK_ZONES, len(possible_cells))
        self.risk_zones = possible_cells[:num_risk]
        
        remaining_cells = possible_cells[num_risk:]
        num_safe = min(self.NUM_SAFE_ZONES, len(remaining_cells))
        self.safe_zones = remaining_cells[:num_safe]

    def _is_valid_move(self, from_pos, to_pos):
        fx, fy = from_pos
        tx, ty = to_pos

        # Check boundaries
        if not (0 <= tx < self.GRID_SIZE and 0 <= ty < self.GRID_SIZE):
            return False

        # Check walls
        if tx > fx:  # Moving Right
            return not self.maze[fy][fx]['E']
        if tx < fx:  # Moving Left
            return not self.maze[fy][fx]['W']
        if ty > fy:  # Moving Down
            return not self.maze[fy][fx]['S']
        if ty < fy:  # Moving Up
            return not self.maze[fy][fx]['N']
        
        return False # Should not happen

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will create a window and render the game
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Create a window to display the game
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Runner")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while not terminated and not truncated:
        movement = 0 # No-op by default
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    total_reward = 0
                    print("\n--- Game Reset ---")
                
        # If a movement key was pressed, step the environment
        if movement != 0:
            action = [movement, 0, 0] # space/shift are not used
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            terminated = term
            truncated = trunc
            print(f"Move: {info['steps']}, Action: {movement}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Moves Left: {info['moves_remaining']}")

            if terminated or truncated:
                print(f"--- Episode Finished ---")
                print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

        # Render the observation to the screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()