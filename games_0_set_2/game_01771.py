import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set the SDL video driver to "dummy" to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the ninja. "
        "Collect the numbers from 1 to 10 in order."
    )

    # Short, user-facing description of the game
    game_description = (
        "A minimalist puzzle game. Guide your ninja to collect numbers in ascending order "
        "before you run out of moves. Plan your path carefully!"
    )

    # Frames only advance when an action is received
    auto_advance = False

    # --- Constants ---
    # Game parameters
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    MAX_MOVES = 15
    NUM_TARGETS = 10
    
    # Colors (Vibrant, high-contrast)
    COLOR_BG = (44, 62, 80)  # #2c3e50
    COLOR_GRID = (52, 73, 94)  # #34495e
    COLOR_NINJA = (236, 240, 241)  # #ecf0f1
    COLOR_NINJA_GLOW = (236, 240, 241, 50)
    COLOR_TEXT_DEFAULT = (189, 195, 199)  # #bdc3c7
    COLOR_TEXT_TARGET = (46, 204, 113)  # #2ecc71
    COLOR_TEXT_COLLECTED = (127, 140, 141) # #7f8c8d
    COLOR_TEXT_UI = (236, 240, 241) # #ecf0f1
    COLOR_WIN = (46, 204, 113)
    COLOR_LOSE = (231, 76, 60) # #e74c3c

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width = self.GRID_WIDTH * self.CELL_SIZE
        self.screen_height = self.GRID_HEIGHT * self.CELL_SIZE
        
        # EXACT spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_game = pygame.font.Font(None, 32)
        self.font_ui = pygame.font.Font(None, 28)
        self.font_endgame = pygame.font.Font(None, 72)
        
        # State variables (initialized in reset)
        self.ninja_pos = None
        self.number_positions = None
        self.current_target = None
        self.moves_left = None
        self.collected_numbers = None
        self.game_over = None
        self.win = None
        self.score = None
        self.steps = None
        self.rng = None
        self.particles = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.rng is None:
            self.rng = np.random.default_rng(seed)

        # Construct a solvable level to avoid infinite loops during generation
        generation_attempts = 0
        while True:
            generation_attempts += 1
            if generation_attempts > 100: # Failsafe
                raise RuntimeError("Failed to generate a solvable level after 100 attempts.")
            
            try:
                occupied_cells = set()
                
                # 1. Place Ninja
                ninja_start_pos = (
                    self.rng.integers(0, self.GRID_WIDTH),
                    self.rng.integers(0, self.GRID_HEIGHT)
                )
                occupied_cells.add(ninja_start_pos)
                current_pos = ninja_start_pos
                
                # 2. Plan the path for targets
                # Create 10 path segments with a total Manhattan distance of MAX_MOVES (15).
                segment_lengths = [1] * self.NUM_TARGETS
                remaining_dist = self.MAX_MOVES - self.NUM_TARGETS
                for _ in range(remaining_dist):
                    idx = self.rng.integers(0, self.NUM_TARGETS)
                    segment_lengths[idx] += 1
                self.rng.shuffle(segment_lengths)
                
                temp_number_positions = {}

                # 3. Create the path segment by segment
                for i in range(1, self.NUM_TARGETS + 1):
                    dist_to_travel = segment_lengths[i - 1]
                    
                    path_segment = [current_pos]
                    temp_pos = current_pos
                    
                    # 3a. Walk to find the next target position
                    for _ in range(dist_to_travel):
                        possible_moves = []
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            next_x, next_y = temp_pos[0] + dx, temp_pos[1] + dy
                            
                            if not (0 <= next_x < self.GRID_WIDTH and 0 <= next_y < self.GRID_HEIGHT):
                                continue
                            if (next_x, next_y) in occupied_cells or (next_x, next_y) in path_segment:
                                continue
                            possible_moves.append((next_x, next_y))
                        
                        if not possible_moves:
                            raise ValueError("Path generation got stuck.")
                        
                        move_idx = self.rng.integers(len(possible_moves))
                        temp_pos = possible_moves[move_idx]
                        path_segment.append(temp_pos)

                    # 3b. Set the target and update occupied cells
                    next_target_pos = temp_pos
                    temp_number_positions[i] = next_target_pos
                    
                    for pos in path_segment[1:]:
                        occupied_cells.add(pos)
                        
                    current_pos = next_target_pos

                # Generation was successful
                self.ninja_pos = pygame.Vector2(ninja_start_pos)
                self.number_positions = {k: pygame.Vector2(v) for k, v in temp_number_positions.items()}
                break

            except ValueError:
                # This attempt failed, retry
                continue

        # Initialize game state
        self.current_target = 1
        self.moves_left = self.MAX_MOVES
        self.collected_numbers = set()
        self.game_over = False
        self.win = False
        self.score = 0
        self.steps = 0
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # Only process movement actions (1-4)
        if movement > 0:
            self.steps += 1
            old_pos = pygame.Vector2(self.ninja_pos)
            
            if movement == 1: # Up
                self.ninja_pos.y -= 1
            elif movement == 2: # Down
                self.ninja_pos.y += 1
            elif movement == 3: # Left
                self.ninja_pos.x -= 1
            elif movement == 4: # Right
                self.ninja_pos.x += 1

            # Clamp position to grid boundaries
            self.ninja_pos.x = max(0, min(self.GRID_WIDTH - 1, self.ninja_pos.x))
            self.ninja_pos.y = max(0, min(self.GRID_HEIGHT - 1, self.ninja_pos.y))
            
            # If a move was actually made
            if old_pos != self.ninja_pos:
                self.moves_left -= 1
                
                # Reward for moving towards/away from the target
                if self.current_target <= self.NUM_TARGETS:
                    target_pos = self.number_positions[self.current_target]
                    dist_before = abs(old_pos.x - target_pos.x) + abs(old_pos.y - target_pos.y)
                    dist_after = abs(self.ninja_pos.x - target_pos.x) + abs(self.ninja_pos.y - target_pos.y)
                    if dist_after < dist_before:
                        reward += 1  # Moved closer
                    else:
                        reward -= 1  # Moved away or parallel

            # Check for number collection
            if self.ninja_pos.x in [pos.x for pos in self.number_positions.values()] and \
               self.ninja_pos.y in [pos.y for pos in self.number_positions.values()]:
                for num, pos in self.number_positions.items():
                    if self.ninja_pos == pos and num not in self.collected_numbers:
                        if num == self.current_target:
                            reward += 10
                            self.collected_numbers.add(num)
                            self.current_target += 1
                            self._create_particles(pos, self.COLOR_TEXT_TARGET)
                        else:
                            reward -= 2
                            self._create_particles(pos, self.COLOR_LOSE)

        # Check for termination conditions
        if self.current_target > self.NUM_TARGETS:
            self.win = True
            self.game_over = True
            reward += 100
        elif self.moves_left <= 0:
            self.game_over = True
            reward -= 10
        
        self.score += reward
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _render_game(self):
        # Draw grid
        for x in range(0, self.screen_width, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_width, y))

        # Draw numbers
        for num, pos in self.number_positions.items():
            if num in self.collected_numbers:
                color = self.COLOR_TEXT_COLLECTED
            elif num == self.current_target and not self.game_over:
                color = self.COLOR_TEXT_TARGET
            else:
                color = self.COLOR_TEXT_DEFAULT
            
            text_surf = self.font_game.render(str(num), True, color)
            text_rect = text_surf.get_rect(center=self._grid_to_pixel(pos))
            self.screen.blit(text_surf, text_rect)

        # Update and draw particles
        self._update_particles()
        for p in self.particles:
            p_pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.aacircle(self.screen, p_pos[0], p_pos[1], int(p['radius']), p['color'])
            pygame.gfxdraw.filled_circle(self.screen, p_pos[0], p_pos[1], int(p['radius']), p['color'])

        # Draw ninja
        if self.ninja_pos:
            ninja_pixel_pos = self._grid_to_pixel(self.ninja_pos)
            x, y = int(ninja_pixel_pos[0]), int(ninja_pixel_pos[1])
            radius = self.CELL_SIZE // 3
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius + 4, self.COLOR_NINJA_GLOW)
            # Main body
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_NINJA)
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, self.COLOR_NINJA)

    def _render_ui(self):
        # Moves left
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT_UI)
        self.screen.blit(moves_text, (10, 10))

        # Target number
        target_str = f"Target: {self.current_target}" if not self.game_over else "Target: -"
        target_text = self.font_ui.render(target_str, True, self.COLOR_TEXT_TARGET)
        target_rect = target_text.get_rect(topright=(self.screen_width - 10, 10))
        self.screen.blit(target_text, target_rect)
        
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT_UI)
        score_rect = score_text.get_rect(midtop=(self.screen_width // 2, 10))
        self.screen.blit(score_text, score_rect)

        # End game message
        if self.game_over:
            if self.win:
                end_text_str = "VICTORY!"
                end_color = self.COLOR_WIN
            else:
                end_text_str = "OUT OF MOVES"
                end_color = self.COLOR_LOSE
            
            end_surf = self.font_endgame.render(end_text_str, True, end_color)
            end_rect = end_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            
            # Simple shadow/outline for readability
            shadow_surf = self.font_endgame.render(end_text_str, True, self.COLOR_BG)
            self.screen.blit(shadow_surf, end_rect.move(3, 3))
            self.screen.blit(end_surf, end_rect)
    
    def _grid_to_pixel(self, grid_pos):
        x = grid_pos.x * self.CELL_SIZE + self.CELL_SIZE / 2
        y = grid_pos.y * self.CELL_SIZE + self.CELL_SIZE / 2
        return (x, y)

    def _create_particles(self, grid_pos, color):
        pixel_pos = self._grid_to_pixel(grid_pos)
        for _ in range(20):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pixel_pos),
                'vel': vel,
                'radius': self.rng.uniform(2, 5),
                'lifespan': self.rng.integers(15, 31),
                'color': color
            })

    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['vel'] *= 0.95 # friction
            p['radius'] -= 0.1
            if p['lifespan'] > 0 and p['radius'] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def close(self):
        pygame.quit()

# Example usage for manual play and visualization
if __name__ == '__main__':
    # This block will run with a visible window, as it re-initializes pygame
    # after the headless environment is set up. Note that this may not work
    # on all systems, but is provided for local testing.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame window for rendering
    pygame.display.set_caption("Ninja Number Maze")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    
    action = env.action_space.sample()
    action.fill(0) # Start with a no-op

    running = True
    while running:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if terminated:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                        action.fill(0)
                    continue

                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    action.fill(0)
                    continue
                else:
                    # Don't step on other key presses
                    continue

                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
                action.fill(0) # Reset action after one step
                if terminated:
                    print("Game Over! Press 'R' to restart.")

        # Drawing
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()