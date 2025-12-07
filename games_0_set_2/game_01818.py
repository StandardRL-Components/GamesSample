import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move through the maze. Each step consumes time. "
        "Collect yellow diamonds for bonus time."
    )

    game_description = (
        "Navigate a procedurally generated isometric maze against the clock. "
        "Collect checkpoints for more time and reach the green exit to advance through 3 stages of increasing difficulty."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_FLOOR = (40, 50, 70)
        self.COLOR_WALL_TOP = (100, 110, 130)
        self.COLOR_WALL_SIDE = (80, 90, 110)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 100, 100, 50)
        self.COLOR_CHECKPOINT = (255, 220, 50)
        self.COLOR_CHECKPOINT_GLOW = (255, 220, 50, 60)
        self.COLOR_EXIT = (50, 255, 100)
        self.COLOR_EXIT_GLOW = (50, 255, 100, 60)
        self.COLOR_TEXT = (255, 255, 255)

        # --- Game State ---
        self.stage_configs = [
            {"size": (10, 10), "checkpoints": 3},
            {"size": (15, 15), "checkpoints": 5},
            {"size": (20, 20), "checkpoints": 7},
        ]
        self.max_stages = len(self.stage_configs)
        
        # Initialize state variables to be populated in reset
        self.maze = None
        self.player_pos = None
        self.checkpoints = []
        self.exit_pos = None
        self.current_stage = 0
        self.time_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.player_trail = []
        self.particles = []
        self.last_dist_to_checkpoint = 0.0

        # Isometric projection parameters
        self.tile_width = 24
        self.tile_height = 12
        self.wall_height = 18
        self.origin_x = 0
        self.origin_y = 0

        # self.reset() is called by the environment wrapper, no need to call it here.
        # self.validate_implementation() # This can cause issues in some testing frameworks

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.current_stage = 1
        
        self._setup_stage()

        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        config = self.stage_configs[self.current_stage - 1]
        maze_w, maze_h = config["size"]
        
        self.maze = self._generate_maze(maze_w, maze_h)
        self.player_pos = np.array([1, 1], dtype=float) # Start at top-left cell
        self.exit_pos = np.array([2 * maze_w - 1, 2 * maze_h - 1])
        
        self._place_checkpoints(config["checkpoints"], maze_w, maze_h)

        self.time_remaining = 60
        self.player_trail = []
        self.particles = []
        self.last_dist_to_checkpoint = self._get_dist_to_nearest_checkpoint()

    def _generate_maze(self, width, height):
        # Maze is represented as a grid where 0 is path and 1 is wall.
        # Size is (2*height+1) x (2*width+1) to represent walls between cells.
        maze = np.ones((2 * height + 1, 2 * width + 1), dtype=np.uint8)
        
        # Start DFS from cell (0,0)
        stack = [(0, 0)]
        visited = {(0, 0)}
        maze[1, 1] = 0

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # FIX: Correctly select a random neighbor and unpack its coordinates.
                # The original code had a TypeError from trying to unpack a single integer.
                neighbor_index = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[neighbor_index]
                
                # Carve path
                maze[2 * y + 1 + (ny - y), 2 * x + 1 + (nx - x)] = 0
                maze[2 * ny + 1, 2 * nx + 1] = 0
                
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        
        # Ensure exit is open
        maze[2 * height - 1, 2 * width - 1] = 0
        maze[2*height, 2*width-1] = 0 # Open bottom wall for exit
        return maze

    def _place_checkpoints(self, num_checkpoints, maze_w, maze_h):
        self.checkpoints = []
        possible_locs = []
        for r in range(1, 2 * maze_h, 2):
            for c in range(1, 2 * maze_w, 2):
                if (self.maze[r, c] == 0 and 
                    not np.array_equal([c, r], self.player_pos) and
                    not np.array_equal([c, r], self.exit_pos)):
                    possible_locs.append(np.array([c, r]))
        
        if not possible_locs: return
        
        indices = self.np_random.choice(len(possible_locs), size=min(num_checkpoints, len(possible_locs)), replace=False)
        self.checkpoints = [possible_locs[i] for i in indices]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        self.steps += 1

        moved = self._handle_movement(movement)
        
        if moved:
            self.time_remaining -= 1
            
            # Distance-based reward
            new_dist = self._get_dist_to_nearest_checkpoint()
            if new_dist < self.last_dist_to_checkpoint:
                reward += 0.1 # Closer to checkpoint
            elif new_dist > self.last_dist_to_checkpoint:
                reward -= 0.01 # Further from checkpoint
            self.last_dist_to_checkpoint = new_dist

        # Check for checkpoint collection
        collected_checkpoint = None
        for i, cp in enumerate(self.checkpoints):
            if np.array_equal(self.player_pos.astype(int), cp):
                collected_checkpoint = i
                break
        
        if collected_checkpoint is not None:
            # sfx: checkpoint_get
            self.checkpoints.pop(collected_checkpoint)
            reward += 10
            self.time_remaining += 10
            self._create_particles(self.player_pos, self.COLOR_CHECKPOINT, 15)
            self.last_dist_to_checkpoint = self._get_dist_to_nearest_checkpoint()

        # Check for exit
        terminated = False
        if np.array_equal(self.player_pos.astype(int), self.exit_pos):
            # sfx: stage_clear
            reward += 50
            if self.current_stage == self.max_stages:
                reward += 100
                self.game_over = True
                terminated = True
                self.win_message = "YOU WIN!"
            else:
                self.current_stage += 1
                self._setup_stage()

        # Check for game over by time
        if self.time_remaining <= 0:
            reward = -100
            self.game_over = True
            terminated = True
            self.win_message = "TIME UP!"

        self.score += reward
        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        self.player_trail.append(self.player_pos.copy())
        if len(self.player_trail) > 10:
            self.player_trail.pop(0)
        
        if movement == 0:
            return False

        pos = self.player_pos.astype(int)
        
        if movement == 1: # Up (N)
            target_wall = self.maze[pos[1] - 1, pos[0]]
            delta = np.array([0, -2])
        elif movement == 2: # Down (S)
            target_wall = self.maze[pos[1] + 1, pos[0]]
            delta = np.array([0, 2])
        elif movement == 3: # Left (W)
            target_wall = self.maze[pos[1], pos[0] - 1]
            delta = np.array([-2, 0])
        elif movement == 4: # Right (E)
            target_wall = self.maze[pos[1], pos[0] + 1]
            delta = np.array([2, 0])
        else:
            return False

        if target_wall == 0:
            self.player_pos += delta
            return True
        return False

    def _get_dist_to_nearest_checkpoint(self):
        if not self.checkpoints:
            return np.linalg.norm(self.exit_pos - self.player_pos)
        
        distances = [np.linalg.norm(cp - self.player_pos) for cp in self.checkpoints]
        return min(distances)

    def _iso_to_screen(self, x, y, z=0):
        screen_x = self.origin_x + (x - y) * self.tile_width / 2
        screen_y = self.origin_y + (x + y) * self.tile_height / 2 - z
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        maze_h, maze_w = self.maze.shape
        self.origin_x = self.screen.get_width() / 2
        self.origin_y = self.screen.get_height() / 2 - (maze_h * self.tile_height / 4) + 20

        # Draw from back to front for correct occlusion
        for r in range(maze_h):
            for c in range(maze_w):
                # Draw floor
                if self.maze[r, c] == 0:
                    p1 = self._iso_to_screen(c, r)
                    p2 = self._iso_to_screen(c + 1, r)
                    p3 = self._iso_to_screen(c + 1, r + 1)
                    p4 = self._iso_to_screen(c, r + 1)
                    pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3, p4), self.COLOR_FLOOR)
                # Draw walls
                elif self.maze[r, c] == 1:
                    # Top face
                    p1 = self._iso_to_screen(c, r, self.wall_height)
                    p2 = self._iso_to_screen(c + 1, r, self.wall_height)
                    p3 = self._iso_to_screen(c + 1, r + 1, self.wall_height)
                    p4 = self._iso_to_screen(c, r + 1, self.wall_height)
                    pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3, p4), self.COLOR_WALL_TOP)
                    pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3, p4), self.COLOR_WALL_TOP)

                    # Side faces (only draw if neighbor is a path)
                    # South face
                    if r + 1 < maze_h and self.maze[r + 1, c] == 0:
                        p1 = self._iso_to_screen(c, r + 1, self.wall_height)
                        p2 = self._iso_to_screen(c + 1, r + 1, self.wall_height)
                        p3 = self._iso_to_screen(c + 1, r + 1, 0)
                        p4 = self._iso_to_screen(c, r + 1, 0)
                        pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3, p4), self.COLOR_WALL_SIDE)
                    # East face
                    if c + 1 < maze_w and self.maze[r, c + 1] == 0:
                        p1 = self._iso_to_screen(c + 1, r, self.wall_height)
                        p2 = self._iso_to_screen(c + 1, r + 1, self.wall_height)
                        p3 = self._iso_to_screen(c + 1, r + 1, 0)
                        p4 = self._iso_to_screen(c + 1, r, 0)
                        pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3, p4), self.COLOR_WALL_SIDE)
        
        # Draw entities on top of maze
        self._render_entities()

    def _render_entities(self):
        # Draw Exit
        exit_center = self._iso_to_screen(self.exit_pos[0] + 0.5, self.exit_pos[1] + 0.5)
        pygame.gfxdraw.filled_circle(self.screen, exit_center[0], exit_center[1], 12, self.COLOR_EXIT_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, exit_center[0], exit_center[1], 8, self.COLOR_EXIT)
        
        # Draw Checkpoints
        for cp in self.checkpoints:
            cp_center = self._iso_to_screen(cp[0] + 0.5, cp[1] + 0.5)
            p1 = (cp_center[0], cp_center[1] - 8)
            p2 = (cp_center[0] + 8, cp_center[1])
            p3 = (cp_center[0], cp_center[1] + 8)
            p4 = (cp_center[0] - 8, cp_center[1])
            pygame.gfxdraw.filled_polygon(self.screen, (p1,p2,p3,p4), self.COLOR_CHECKPOINT_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, (p1,p2,p3,p4), self.COLOR_CHECKPOINT)

        # Draw Player Trail
        for i, pos in enumerate(self.player_trail):
            alpha = int(100 * (i / len(self.player_trail)))
            trail_color = (*self.COLOR_PLAYER, alpha)
            trail_pos = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5, -2)
            pygame.gfxdraw.filled_circle(self.screen, trail_pos[0], trail_pos[1], 4, trail_color)

        # Draw Player
        player_center = self._iso_to_screen(self.player_pos[0] + 0.5, self.player_pos[1] + 0.5, -2)
        pygame.gfxdraw.filled_circle(self.screen, player_center[0], player_center[1], 8, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, player_center[0], player_center[1], 5, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center[0], player_center[1], 5, self.COLOR_PLAYER)

        # Draw Particles
        for p in self.particles:
            pos_x, pos_y, life, radius, color = p
            p_center = self._iso_to_screen(pos_x + 0.5, pos_y + 0.5, -2)
            alpha = int(255 * (life / 20.0))
            p_color = (*color[:3], alpha)
            pygame.gfxdraw.filled_circle(self.screen, p_center[0], p_center[1], int(radius), p_color)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append([
                pos[0], pos[1], # position
                20, # lifetime
                self.np_random.random() * 5 + 2, # radius
                color
            ])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[2] -= 1 # decrement life
            p[3] += 0.2 # increase radius

    def _render_ui(self):
        # Time
        time_text = self.font_large.render(f"TIME: {int(self.time_remaining)}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.screen.get_width() - time_text.get_width() - 10, 10))

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.current_stage}/{self.max_stages}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.screen.get_width() - stage_text.get_width() - 10, 50))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Game Over / Win Message
        if self.game_over and self.win_message:
            msg_surf = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "time_remaining": self.time_remaining,
            "checkpoints_left": len(self.checkpoints),
        }
    
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To play, you might need to unset the dummy video driver
    # comment out: os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # and install pygame: pip install pygame
    
    # Re-enable video driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Isometric Maze Runner")
    screen = pygame.display.set_mode((640, 400))
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        # Process events once per frame
        event_occurred = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                event_occurred = True
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if event_occurred and action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
            if terminated:
                print("Game Over! Resetting in 3 seconds...")
                # Render final state before waiting
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                
                pygame.time.wait(3000)
                obs, info = env.reset()
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate

    env.close()