
# Generated: 2025-08-27T20:07:41.634901
# Source Brief: brief_02358.md
# Brief Index: 2358

import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Use arrow keys to move the selected robot. "
        "A no-op action (e.g., waiting) selects the next robot. "
        "Rescue all 7 robots in under 30 moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a team of colorful robots through a procedurally generated maze "
        "to their rescue zone, minimizing moves while taking calculated risks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.TILE_SIZE = self.WIDTH // self.GRID_W
        self.NUM_ROBOTS = 7
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_WALL = (40, 50, 60)
        self.COLOR_PATH = (60, 70, 80)
        self.COLOR_RESCUE = (60, 180, 75, 100) # Green with alpha
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.ROBOT_COLORS = [
            (230, 25, 75), (255, 225, 25), (0, 130, 200),
            (245, 130, 48), (145, 30, 180), (70, 240, 240),
            (240, 50, 230)
        ]
        self.COLOR_SELECTION = (255, 255, 255)
        self.COLOR_TRAP = (220, 50, 50)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_used = 0
        self.robots = []
        self.maze = np.array([])
        self.rescue_zone_pos = (0,0)
        self.selected_robot_idx = 0
        self.particles = []
        self.win_message = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Seed python's random for maze generation
        if seed is not None:
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_used = 0
        self.win_message = ""
        self.particles = []

        # --- Generate World ---
        self._generate_maze()
        self.rescue_zone_pos = (self.GRID_W - 2, self.GRID_H - 2)
        
        # --- Place Robots ---
        self.robots = []
        # Use a consistent source of randomness for robot placement
        possible_starts = np.argwhere(self.maze == 0).tolist()
        self.np_random.shuffle(possible_starts)
        
        # Ensure rescue zone isn't a starting spot
        if list(self.rescue_zone_pos) in possible_starts:
            possible_starts.remove(list(self.rescue_zone_pos))

        for i in range(self.NUM_ROBOTS):
            if not possible_starts: break # Avoid error if not enough start points
            pos = possible_starts.pop()
            self.robots.append({
                'pos': tuple(pos),
                'status': 'active', # 'active', 'rescued', 'trapped'
                'color': self.ROBOT_COLORS[i],
                'id': i
            })
        
        self.selected_robot_idx = 0
        self._cycle_selection_to_active()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = 0.0
        
        # --- Handle Action ---
        if movement == 0: # No-op: cycle selection
            self._cycle_selection_to_active()
        else: # Movement action
            self.moves_used += 1
            reward -= 0.1
            
            if self.robots: # Ensure robots list is not empty
                robot = self.robots[self.selected_robot_idx]
                if robot['status'] == 'active':
                    dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
                    target_pos = (robot['pos'][0] + dx, robot['pos'][1] + dy)
                    
                    if self._is_valid_move(target_pos):
                        robot['pos'] = target_pos

                        # Check for rescue
                        if robot['pos'] == self.rescue_zone_pos:
                            robot['status'] = 'rescued'
                            reward += 10
                            self._create_particles(robot['pos'], robot['color'], 30)
                            self._cycle_selection_to_active()
            
        # --- Update Score ---
        self.score += reward

        # --- Check Termination Conditions ---
        robots_rescued = sum(1 for r in self.robots if r['status'] == 'rescued')
        active_robots = [r for r in self.robots if r['status'] == 'active']

        # 1. Win Condition
        if robots_rescued == self.NUM_ROBOTS:
            self.game_over = True
            self.score += 50
            reward += 50
            self.win_message = "ALL ROBOTS RESCUED!"

        # 2. Loss Conditions (if not already won)
        if not self.game_over:
            # a. Out of moves
            if self.moves_used >= self.MAX_MOVES:
                self.game_over = True
                self.win_message = "OUT OF MOVES"

            # b. Robot is trapped
            if not any(r['status'] == 'active' for r in self.robots):
                # If no robots are active but not all are rescued, it's a loss
                if robots_rescued < self.NUM_ROBOTS:
                    self.game_over = True
                    self.win_message = "ROBOTS TRAPPED!"
            else:
                for r in active_robots:
                    if self._is_trapped(r):
                        r['status'] = 'trapped'
                        # A single trapped robot now just becomes inactive, not a game over
                        self._cycle_selection_to_active()

                # Check if all remaining robots are trapped
                if all(r['status'] != 'active' for r in self.robots) and robots_rescued < self.NUM_ROBOTS:
                    self.game_over = True
                    self.score -= 10
                    reward -= 10
                    self.win_message = "ALL ROBOTS TRAPPED!"
        
        # 3. Hard step limit
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        terminated = self.game_over and not truncated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.MAX_MOVES - self.moves_used,
            "robots_rescued": sum(1 for r in self.robots if r['status'] == 'rescued'),
        }

    # --- Helper & Rendering Methods ---

    def _cycle_selection_to_active(self):
        if not self.robots or all(r['status'] != 'active' for r in self.robots):
            return # No active robots to select
        
        current_idx = self.selected_robot_idx
        for _ in range(self.NUM_ROBOTS):
            current_idx = (current_idx + 1) % self.NUM_ROBOTS
            if self.robots[current_idx]['status'] == 'active':
                self.selected_robot_idx = current_idx
                return

    def _is_valid_move(self, target_pos):
        tx, ty = target_pos
        # Check bounds
        if not (0 <= tx < self.GRID_W and 0 <= ty < self.GRID_H):
            return False
        # Check walls
        if self.maze[tx, ty] == 1:
            return False
        # Check other robots
        for r in self.robots:
            if r['status'] != 'rescued' and r['pos'] == target_pos:
                return False
        return True

    def _is_trapped(self, robot):
        x, y = robot['pos']
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            if self._is_valid_move((x + dx, y + dy)):
                return False
        return True

    def _generate_maze(self):
        self.maze = np.ones((self.GRID_W, self.GRID_H), dtype=np.uint8)
        stack = deque()
        
        start_x, start_y = (self.np_random.integers(1, self.GRID_W//2) * 2, 
                            self.np_random.integers(1, self.GRID_H//2) * 2)
        self.maze[start_x, start_y] = 0
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and self.maze[nx, ny] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                self.maze[nx, ny] = 0
                self.maze[cx + (nx - cx) // 2, cy + (ny - cy) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _render_game(self):
        # Update and render particles
        self._update_and_render_particles()

        # Render maze and rescue zone
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if self.maze[x, y] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_PATH, rect)
        
        # Render rescue zone
        rz_rect = pygame.Rect(self.rescue_zone_pos[0] * self.TILE_SIZE, 
                              self.rescue_zone_pos[1] * self.TILE_SIZE, 
                              self.TILE_SIZE, self.TILE_SIZE)
        s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_RESCUE)
        self.screen.blit(s, rz_rect.topleft)

        # Render robots
        for i, robot in enumerate(self.robots):
            if robot['status'] == 'rescued':
                continue
            
            px, py = (robot['pos'][0] + 0.5) * self.TILE_SIZE, (robot['pos'][1] + 0.5) * self.TILE_SIZE
            radius = int(self.TILE_SIZE * 0.35)
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, robot['color'])
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, robot['color'])

            # Selection highlight
            if i == self.selected_robot_idx and not self.game_over and robot['status'] == 'active':
                pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
                sel_radius = radius + int(2 + pulse * 3)
                alpha = int(150 + pulse * 105)
                pygame.gfxdraw.aacircle(self.screen, int(px), int(py), sel_radius, self.COLOR_SELECTION + (alpha,))
                pygame.gfxdraw.aacircle(self.screen, int(px), int(py), sel_radius + 1, self.COLOR_SELECTION + (alpha // 2,))
            
            # Trapped indicator
            if robot['status'] == 'trapped':
                pygame.draw.line(self.screen, self.COLOR_TRAP, (px - radius, py - radius), (px + radius, py + radius), 3)
                pygame.draw.line(self.screen, self.COLOR_TRAP, (px - radius, py + radius), (px + radius, py - radius), 3)

    def _render_ui(self):
        # --- Info Text ---
        moves_text = f"Moves: {self.moves_used}/{self.MAX_MOVES}"
        rescued_text = f"Rescued: {sum(1 for r in self.robots if r['status'] == 'rescued')}/{self.NUM_ROBOTS}"
        
        moves_surf = self.font_medium.render(moves_text, True, self.COLOR_UI_TEXT)
        rescued_surf = self.font_medium.render(rescued_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(moves_surf, (10, 10))
        self.screen.blit(rescued_surf, (self.WIDTH - rescued_surf.get_width() - 10, 10))

        # --- Game Over Text ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_text_surf = self.font_large.render(self.win_message, True, self.COLOR_UI_TEXT)
            text_rect = end_text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text_surf, text_rect)

    def _create_particles(self, grid_pos, color, count):
        px, py = (grid_pos[0] + 0.5) * self.TILE_SIZE, (grid_pos[1] + 0.5) * self.TILE_SIZE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 41),
                'color': color
            })

    def _update_and_render_particles(self):
        # Use a temporary surface for particles to handle alpha correctly
        particle_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        
        remaining_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] > 0:
                remaining_particles.append(p)
                alpha = int(255 * (p['life'] / 40))
                color = p['color'] + (alpha,)
                size = int(p['life'] / 10)
                pygame.draw.circle(particle_surf, color, (int(p['pos'][0]), int(p['pos'][1])), max(1, size))
        
        self.particles = remaining_particles
        self.screen.blit(particle_surf, (0,0))


    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'human' to see the game being played.
    # Set to 'rgb_array' for training.
    render_mode = "human" 

    # The environment is always created in 'rgb_array' mode for headless operation.
    # The main loop here handles the rendering for 'human' mode.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    if render_mode == "human":
        # We need to unset the dummy driver and re-init pygame for display
        os.environ.pop("SDL_VIDEODRIVER", None)
        pygame.quit()
        pygame.init()
        human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption(env.game_description)
        
        terminated = False
        truncated = False
        while not terminated and not truncated:
            # Convert numpy array back to a Pygame Surface for display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()

            action = np.array([0, 0, 0]) # Default to no-op
            
            # Simple event loop for human control
            should_step = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    should_step = False # To exit loop immediately
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                        should_step = True
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                        should_step = True
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                        should_step = True
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                        should_step = True
                    elif event.key == pygame.K_SPACE: # Use space for no-op/cycle
                        action[0] = 0
                        should_step = True
                    elif event.key == pygame.K_r: # Reset
                        obs, info = env.reset(seed=random.randint(0, 10000))
                        should_step = False
                    
                    if should_step:
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
                        if terminated or truncated:
                            # Render final frame
                            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                            human_screen.blit(surf, (0, 0))
                            pygame.display.flip()
                            print("Game Over! Press R to restart or Q to quit.")
            
            if terminated or truncated:
                # Wait for R or Q after game over
                wait_for_input = True
                while wait_for_input:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                            terminated = True # to exit outer loop
                            wait_for_input = False
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                            obs, info = env.reset(seed=random.randint(0, 10000))
                            terminated = False
                            truncated = False
                            wait_for_input = False


        env.close()

    else: # rgb_array mode test
        obs, info = env.reset(seed=42)
        print("Initial Info:", info)
        
        for i in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}, Info={info}")
            if terminated or truncated:
                print("--- Episode Finished ---")
                obs, info = env.reset(seed=43+i)
                print("New Episode Info:", info)
        
        env.close()